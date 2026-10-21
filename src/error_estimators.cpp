/*
 * @Description: Implementations of the current-density jump and SPR patch-recovery
 *   error estimators.
 * @Author: Lewen Qiu, Zhengguang Liu, Hongbo Yao (original DC3DPAFEM module);
 *   adapted for DC3DPAINV by Lewen Qiu and Huang Chen
 * @Date: 2026-10-21
 */

// Copyright (c) 2026, Lewen Qiu and Huang Chen.
// Portions copyright (c) 2023, Lewen Qiu, Zhengguang Liu,
// Hongbo Yao, and Jingtian Tang.
// This file is part of the DC3DPAINV program. DC3DPAINV is free
// software with source code available at
// https://github.com/luowen7592/DC3DPAINV. You can redistribute it
// or modify it under the terms of the BSD-3 license. See the
// LICENSE file for details.

#include "error_estimators.h"

#include <iostream>
#include <time.h>

Estimator::Estimator(

    std::shared_ptr<ParMesh> pmesh_,

    std::shared_ptr<ParFiniteElementSpace> pfes_,

    std::vector<std::shared_ptr<ParGridFunction>> up_,

    std::shared_ptr<ParGridFunction> w_)
    :
      pmesh(pmesh_), pfes(pfes_), up(up_), w(w_) {}

Estimator::~Estimator() {}

nJEstimator::nJEstimator(

    std::shared_ptr<ParMesh> pmesh_,

    std::shared_ptr<ParFiniteElementSpace> pfes_,

    std::vector<std::shared_ptr<ParGridFunction>> up_,

    std::shared_ptr<ParGridFunction> w_,
    std::vector<DenseMatrix> &att_cond)
    :
      Estimator(pmesh_, pfes_, up_, w_), cond_att(att_cond) {

}

nJEstimator::~nJEstimator() {}

double nJEstimator::n_dot_J(bool if_loc1, bool if_shared_tet2,
                            FaceElementTransformations *face_elem_trans,
                            ElementTransformation *face_trans,
                            ElementTransformation *tet_trans,
                            const FiniteElement *tet_fe, int tet_id,
                            const IntegrationPoint &ip,

                            std::shared_ptr<ParGridFunction> u) {

  int ndof = tet_fe->GetDof();
  int dim = tet_fe->GetDim();
  myassert(dim == 3);

  Vector normalJ(dim);
  normalJ = 0.0;
  CalcOrtho(face_trans->Jacobian(), normalJ);

  Vector normal(dim);
  normal = 0.0;
  normal.Set(1.0 / normalJ.Norml2(),
             normalJ);

  IntegrationPoint tet_ip;

  if (if_loc1) {

    face_elem_trans->Loc1.Transform(ip, tet_ip);
  } else {
    face_elem_trans->Loc2.Transform(ip, tet_ip);
  }
  tet_trans->SetIntPoint(&tet_ip);
  DenseMatrix tet_shape(ndof, dim);

  tet_fe->CalcPhysDShape(*tet_trans, tet_shape);
  Array<int> tet_vdofs;

  Vector tU;

  if (!if_shared_tet2) {
    pfes->GetElementVDofs(tet_id, tet_vdofs);
    u->GetSubVector(tet_vdofs, tU);
  } else {
    pfes->GetFaceNbrElementVDofs(tet_id, tet_vdofs);
    u->FaceNbrData().GetSubVector(tet_vdofs, tU);

  }

  myassert(tet_vdofs.Size() == ndof);
  myassert(tU.Size() == ndof);
  Vector tempE(3);
  tempE = 0.0;

  for (int j = 0; j < ndof; j++) {
    for (int c = 0; c < 3; c++) {
      double shape = tet_shape(j, c);
      tempE(c) += tU[j] * shape;
    }
  }

  Vector tempJe(3);
  tempJe = 0.0;
  DenseMatrix tet_cond = cond_att[tet_trans->Attribute - 1];
  tet_cond.AddMult(tempE, tempJe);

  double nJ = 0.0;
  nJ = normal * tempJe;

  return nJ;
}

void nJEstimator::compute_face_err(bool if_shared_tet2,
                                   FaceElementTransformations *face_elem_trans,
                                   double *face_err_u, double &face_err_w) {
  ElementTransformation *face_trans = face_elem_trans->Face;
  int tet1_id;
  const FiniteElement *tet1_fe;
  ElementTransformation *tet1_trans;
  int tet2_id;
  const FiniteElement *tet2_fe;
  ElementTransformation *tet2_trans;

  if (!if_shared_tet2) {
    tet1_id = face_elem_trans->Elem1No;
    tet1_fe = pfes->GetFE(tet1_id);
    tet1_trans = face_elem_trans->Elem1;
    tet2_id = face_elem_trans->Elem2No;
    tet2_fe = pfes->GetFE(tet2_id);
    tet2_trans = face_elem_trans->Elem2;
  } else {
    tet1_id = face_elem_trans->Elem1No;
    tet1_fe = pfes->GetFE(tet1_id);
    tet1_trans = face_elem_trans->Elem1;

    tet2_id = face_elem_trans->Elem2No - pmesh->GetNE();

    tet2_fe = pfes->GetFaceNbrFE(tet2_id);
    tet2_trans = face_elem_trans->Elem2;

  }

  const IntegrationRule *face_ir =
      &IntRules.Get(face_trans->GetGeometryType(), face_trans->Order());

  for (int q = 0; q < face_ir->GetNPoints(); ++q) {

    const IntegrationPoint &ip = face_ir->IntPoint(q);
    face_trans->SetIntPoint(&ip);
    double WJ = face_trans->Weight() * ip.weight;

    double nJ_tet1_u = 0.0;
    double nJ_tet1_w = 0.0;

    double nJ_tet2_u = 0.0;
    double nJ_tet2_w = 0.0;

    int N_s = up.size();

    for (int s = 0; s < N_s; s++) {

      nJ_tet1_u = this->n_dot_J(true, false, face_elem_trans, face_trans,
                                tet1_trans, tet1_fe, tet1_id, ip, up[s]);

      nJ_tet2_u =
          this->n_dot_J(false, if_shared_tet2, face_elem_trans, face_trans,
                        tet2_trans, tet2_fe, tet2_id, ip, up[s]);

      double nJu = std::abs(nJ_tet1_u) - std::abs(nJ_tet2_u);
      face_err_u[s] += 0.5 * nJu * nJu * WJ;
    }

    nJ_tet1_w = this->n_dot_J(true, false, face_elem_trans, face_trans,
                              tet1_trans, tet1_fe, tet1_id, ip, w);

    nJ_tet2_w = this->n_dot_J(false, if_shared_tet2, face_elem_trans,
                              face_trans, tet2_trans, tet2_fe, tet2_id, ip, w);

    double nJw = std::abs(nJ_tet1_w) - std::abs(nJ_tet2_w);

    face_err_w += 0.5 * nJw * nJw * WJ;
  }
}

void nJEstimator::get_error_estimates(bool if_solve_dual, Vector &errors) {

  for (int s = 0; s < up.size(); s++) {

    up[s]->ExchangeFaceNbrData();
  }
  w->ExchangeFaceNbrData();

  int n_faces = pmesh->GetNumFaces();

  for (int f = 0; f < n_faces; f++) {

    FaceElementTransformations *face_elem_trans =
        pmesh->GetInteriorFaceTransformations(f);
    if (face_elem_trans == NULL)
    {

      continue;
    }

    int tet1_id = face_elem_trans->Elem1No;

    int tet2_id = face_elem_trans->Elem2No;

    double face_err_w = 0.0;

    double face_err_u[up.size()];

    for (int i = 0; i < up.size(); i++) {
      face_err_u[i] = 0.0;
    }

    bool if_shared_tet2 = false;
    compute_face_err(if_shared_tet2, face_elem_trans, face_err_u, face_err_w);

    double elem_error_face = 0.0;

    for (int s = 0; s < up.size(); s++) {

      if (if_solve_dual == true) {
        elem_error_face += std::sqrt(face_err_u[s]) * std::sqrt(face_err_w);
      } else {
        elem_error_face += std::sqrt(face_err_u[s]);
      }

    }

    errors(tet1_id) += elem_error_face;
    errors(tet2_id) += elem_error_face;

  }

  for (int f = 0; f < pmesh->GetNSharedFaces(); f++) {

    FaceElementTransformations *face_elem_trans =
        pmesh->GetSharedFaceTransformations(f);

    int tet1_id = face_elem_trans->Elem1No;

    double face_err_w = 0.0;

    double face_err_u[up.size()];

    for (int i = 0; i < up.size(); i++) {
      face_err_u[i] = 0.0;
    }

    bool if_shared_tet2 = true;
    compute_face_err(if_shared_tet2, face_elem_trans, face_err_u, face_err_w);

    double elem_error_face = 0.0;

    for (int s = 0; s < up.size(); s++) {

      if (if_solve_dual == true) {
        elem_error_face += std::sqrt(face_err_u[s]) * std::sqrt(face_err_w);
      } else {
        elem_error_face += std::sqrt(face_err_u[s]);
      }

    }

    errors(tet1_id) += elem_error_face;

  }

}

SPREstimator::SPREstimator(

    std::shared_ptr<ParMesh> pmesh_,

    std::shared_ptr<ParFiniteElementSpace> pfes_,

    std::vector<std::shared_ptr<ParGridFunction>> up_,

    std::shared_ptr<ParGridFunction> w_, std::vector<DenseMatrix> &att_cond,
    const unsigned int s)
    :
      Estimator(pmesh_, pfes_, up_, w_), cond_att(att_cond) {

}

void SPREstimator::gradient_patch_recovery(std::vector<DenseVector> &Ui,
                                           DenseVector &W,
                                           std::vector<DenseMatrix> &cond_att,
                                           Vector &local_err,
                                           bool if_solve_dual) {

  int Ne = pmesh->GetNE();

  Ui.push_back(W);
  std::set<unsigned int> computed_elems;

  for (unsigned int i = 0; i < Ne; i++) {

    unsigned int id = i;

    if (computed_elems.find(id) == computed_elems.end()) {

      std::set<unsigned int> patch_id;

      this->get_element_patch(id, patch_id, this->_s, cond_att);

      assert(patch.size() > 0);

      Matrix4D C;
      std::vector<Matrix43D> b, a;

      this->compute_C(patch_id, C);

      this->get_patch_b(patch_id, Ui, b);

      this->solve_a(C, b, a);

      std::map<unsigned int, double> energy_norm;
      this->compute_PI_k(patch_id, Ui, a, cond_att, energy_norm, if_solve_dual);

      std::set<unsigned int>::const_iterator it = patch_id.begin();
      for (; it != patch_id.end(); it++) {

        local_err[(*it)] = energy_norm[(*it)];
        computed_elems.insert((*it));

      }
    }

  }

  Ui.erase(Ui.end());
  std::vector<double> temp(W.size());
  for (int i = 0; i < temp.size(); i++)
    temp[i] = W(i);

  return;
}

void SPREstimator::get_element_patch(unsigned int elem,

                                     std::set<unsigned int> &patch_id,
                                     const unsigned int S,
                                     std::vector<DenseMatrix> &cond_att) {
  assert(elem != NULL);

  patch_id.clear();
  patch_id.insert(elem);
  const unsigned int N = 20;

  std::set<unsigned int> neighbors;
  neighbors.insert(elem);

  for (int i = 0; i < N; i++) {

    std::set<unsigned int>::const_iterator it = neighbors.begin();
    const std::set<unsigned int>::const_iterator end = neighbors.end();
    std::set<unsigned int> temp;
    for (; it != end; ++it) {

      int elem_temp_id = *it;
      Array<int> faces, ori;

      pmesh->GetElementFaces(elem_temp_id, faces, ori);

      for (unsigned int s = 0; s < faces.Size(); s++) {
        int face_id = faces[s];
        int nr_id = -1;
        bool haveNeighbor = false;

        if (pmesh->FaceIsInterior(face_id)) {
          haveNeighbor = true;
          int elem1_id, elem2_id;
          pmesh->GetFaceElements(face_id, &elem1_id, &elem2_id);
          myassert(elem1_id != -1 && elem2_id != -1);
          if (elem_temp_id == elem1_id) {
            nr_id = elem2_id;
          } else {
            myassert(elem_temp_id == elem2_id);
            nr_id = elem1_id;
          }
        }

        if (haveNeighbor) {

          const Element *neighbor = pmesh->GetElement(nr_id);
          const Element *e = pmesh->GetElement(elem);

          DenseMatrix temp1(3, 3), temp2(3, 3);

          temp1 = cond_att[neighbor->GetAttribute() - 1];
          temp2 = cond_att[e->GetAttribute() - 1];

          if (equalDM(temp1, temp2)) {

            patch_id.insert(nr_id);
            temp.insert(nr_id);
          }
        }
      }
    }

    neighbors = temp;

    if (patch_id.size() > S)
      break;
  }

}

void SPREstimator::get_patch_polynomial_c(std::vector<double> &c,
                                          const Point_double &p) {
  const double x = p(0), y = p(1), z = p(2);

  c.clear();
  c.push_back(1);
  c.push_back(x);
  c.push_back(y);
  c.push_back(z);
  return;
}

void SPREstimator::compute_C(std::set<unsigned int> &patch_id, Matrix4D &C) {
  assert(patch.size() >= 4);
  C.setZero();

  std::set<unsigned int>::const_iterator it = patch_id.begin();
  for (; it != patch_id.end(); it++) {

    int elem_id = (*it);

    ElementTransformation *eltrans = pfes->GetElementTransformation(elem_id);
    const FiniteElement *fe = pfes->GetFE(elem_id);

    const IntegrationRule *ir =
        &IntRules.Get(fe->GetGeomType(), fe->GetOrder());
    for (unsigned int q = 0; q < ir->GetNPoints(); q++) {

      const IntegrationPoint &ip = ir->IntPoint(q);
      eltrans->SetIntPoint(&ip);
      double WJ = eltrans->Weight() * ip.weight;

      std::vector<double> c;
      Vector r_temp(3);
      r_temp = 0.0;
      eltrans->Transform(ip, r_temp);
      Point_double r(r_temp[0], r_temp[1], r_temp[2]);

      this->get_patch_polynomial_c(c, r);

      assert(c.size() == 4);

      for (unsigned int i = 0; i < 4; i++)
        for (unsigned int j = 0; j < 4; j++)
          C(i, j) += WJ * c[i] * c[j];
    }
  }

  return;
}

void SPREstimator::get_patch_b(std::set<unsigned int> &patch_id,
                               std::vector<DenseVector> &solution,
                               std::vector<Matrix43D> &b) {

  assert(patch.size() >= 4);
  unsigned int n_solution = solution.size();
  b.resize(n_solution);

  std::set<unsigned int>::const_iterator it = patch_id.begin();

  for (; it != patch_id.end(); it++) {

    int elem_id = (*it);
    ElementTransformation *eltrans = pfes->GetElementTransformation(elem_id);
    const FiniteElement *fe = pfes->GetFE(elem_id);

    const IntegrationRule *ir =
        &IntRules.Get(fe->GetGeomType(), fe->GetOrder());

    std::vector<unsigned int> DOFs;

    int ndof = fe->GetDof();
    int dim = fe->GetDim();
    Array<int> vdofs;

    pfes->GetElementDofs(elem_id, vdofs);

    DOFs.resize(vdofs.Size());
    for (int i = 0; i < vdofs.Size(); i++) {
      DOFs[i] = vdofs[i];
    }
    assert(DOFs.size() == 4);

    for (unsigned int q = 0; q < ir->GetNPoints(); q++) {

      const IntegrationPoint &ip = ir->IntPoint(q);
      eltrans->SetIntPoint(&ip);

      double WJ = eltrans->Weight() * ip.weight;

      DenseMatrix Dshape(ndof, 3);
      Dshape = 0.0;
      fe->CalcPhysDShape(*eltrans, Dshape);

      Point_double dev_shape[4];
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
          dev_shape[i](j) = Dshape(i, j);
        }

      }

      Vector r_temp(3);
      r_temp = 0.0;
      eltrans->Transform(ip, r_temp);
      Point_double r(r_temp[0], r_temp[1], r_temp[2]);

      std::vector<double> c;

      this->get_patch_polynomial_c(c, r);

      for (int n = 0; n < n_solution; n++) {
        Point_double grad_u_h;
        b[n].setZero();

        for (unsigned int i = 0; i < 4; i++)
          grad_u_h = grad_u_h + dev_shape[i] * solution[n](DOFs[i]);

        for (unsigned int i = 0; i < 4; i++)
          for (unsigned int j = 0; j < 3; j++) {
            b[n](i, j) += WJ * c[i] * grad_u_h(j);
          }
      }
    }
  }

  return;
}

void SPREstimator::solve_a(Matrix4D &C, std::vector<Matrix43D> &b,
                           std::vector<Matrix43D> &a) {
  a.resize(b.size());
  for (int i = 0; i < a.size(); i++)
    a[i] = C.partialPivLu().solve(b[i]);

  return;
}

void SPREstimator::compute_P_value(Point_double &r, Matrix43D &a,
                                   Matrix13D &P) {
  P.setZero();
  std::vector<double> c;
  this->get_patch_polynomial_c(c, r);
  assert(c.size() == 4);
  Matrix14D dv_c;
  for (int t = 0; t < 4; t++)
    dv_c(t) = c[t];
  P = dv_c * a;

  return;
}

void SPREstimator::compute_PI_k(std::set<unsigned int> &patch_id,
                                std::vector<DenseVector> &solution,
                                std::vector<Matrix43D> &a,

                                std::vector<DenseMatrix> &cond_att,
                                std::map<unsigned int, double> &energy_norm,
                                bool if_solve_dual) {
  assert(solution.size() == a.size());
  energy_norm.clear();

  std::set<unsigned int>::const_iterator it = patch_id.begin();
  for (; it != patch_id.end(); it++) {

    int elem_id = (*it);
    ElementTransformation *eltrans = pfes->GetElementTransformation(elem_id);
    const FiniteElement *fe = pfes->GetFE(elem_id);

    const IntegrationRule *ir =
        &IntRules.Get(fe->GetGeomType(), fe->GetOrder());

    std::vector<unsigned int> DOFs;

    int ndof = fe->GetDof();
    int dim = fe->GetDim();
    Array<int> vdofs;

    pfes->GetElementDofs(elem_id, vdofs);

    DOFs.resize(vdofs.Size());
    for (int i = 0; i < vdofs.Size(); i++) {
      DOFs[i] = vdofs[i];
    }
    assert(DOFs.size() == 4);

    DenseMatrix temp_sigma = cond_att[eltrans->Attribute - 1];
    Matrix3D sigma_ele;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        sigma_ele(i, j) = temp_sigma(i, j);
      }
    }

    double elem_error = 0.;
    double elem_error_w = 0;
    double elem_error_e[solution.size() - 1];
    for (int i = 0; i < solution.size() - 1; i++)
      elem_error_e[i] = 0.;

    for (unsigned int q = 0; q < ir->GetNPoints(); q++) {

      const IntegrationPoint &ip = ir->IntPoint(q);
      eltrans->SetIntPoint(&ip);
      double WJ = eltrans->Weight() * ip.weight;

      DenseMatrix Dshape(ndof, 3);
      Dshape = 0.0;
      fe->CalcPhysDShape(*eltrans, Dshape);

      Point_double dev_shape[4];
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
          dev_shape[i](j) = Dshape(i, j);
        }
      }

      Vector r_temp(3);
      r_temp = 0.0;
      eltrans->Transform(ip, r_temp);
      Point_double r(r_temp[0], r_temp[1], r_temp[2]);

      DenseVector &W = solution[solution.size() - 1];
      Matrix43D &Wa = a[solution.size() - 1];

      Matrix13D grad_w_P;
      this->compute_P_value(r, Wa, grad_w_P);
      Matrix13D grad_w_h;
      Point_double temp_grad_w_h;
      for (unsigned int t = 0; t < 4; t++)
        temp_grad_w_h = temp_grad_w_h + dev_shape[t] * W(DOFs[t]);
      for (int t = 0; t < 3; t++)
        grad_w_h(t) = temp_grad_w_h(t);

      elem_error_w += ((grad_w_P - grad_w_h) * sigma_ele *
                       (grad_w_P - grad_w_h).transpose())
                          .determinant() *
                      WJ;

      for (int i = 0; i < solution.size() - 1; i++) {
        DenseVector &Ui = solution[i];
        Matrix43D &Uia = a[i];
        Matrix13D grad_e_P;
        this->compute_P_value(r, Uia, grad_e_P);
        Matrix13D grad_e_h;
        Point_double temp_grad_e_h;
        for (unsigned int t = 0; t < 4; t++)
          temp_grad_e_h = temp_grad_e_h + dev_shape[t] * Ui(DOFs[t]);
        for (int t = 0; t < 3; t++)
          grad_e_h(t) = temp_grad_e_h(t);

        elem_error_e[i] += ((grad_e_P - grad_e_h) * sigma_ele *
                            (grad_e_P - grad_e_h).transpose())
                               .determinant() *
                           WJ;
      }
    }

    if (if_solve_dual == true) {
      for (int i = 0; i < solution.size() - 1; i++) {

        elem_error += std::sqrt(elem_error_w) * std::sqrt(elem_error_e[i]);

      }
    } else {
      for (int i = 0; i < solution.size() - 1; i++) {
        elem_error += std::sqrt(elem_error_e[i]);
      }
    }
    energy_norm[elem_id] = elem_error;
  }
  return;
}

void SPREstimator::get_error_estimates(bool if_solve_dual, Vector &errors) {

  int source_num = up.size();
  for (int s = 0; s < source_num; s++) {
    up[s]->ExchangeFaceNbrData();
  }
  w->ExchangeFaceNbrData();

  myassert(errors.Size() == pmesh->GetNE());
  myassert(cond_att.size() > 0);

  int dofs = pmesh->GetNV();
  std::vector<DenseVector> Ui(source_num);

  for (int i = 0; i < source_num; i++) {
    Ui[i].resize(dofs);
    for (int j = 0; j < dofs; j++) {
      Ui[i](j) = (*up[i])(j);
    }
  }
  DenseVector W(dofs);
  for (int j = 0; j < dofs; j++) {
    W[j] = (*w)(j);
  }
  this->gradient_patch_recovery(Ui, W, cond_att, errors, if_solve_dual);
}
