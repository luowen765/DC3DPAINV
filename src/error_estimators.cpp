/*
 * @Description: Implementation of the normal current-density jump estimator used
 *   for forward-mesh adaptivity.
 * @Author: Lewen Qiu, Zhengguang Liu, Hongbo Yao (original DC3DPAFEM module);
 *   adapted for DC3DPAINV by Lewen Qiu and Huang Chen
 * @Date: 2026-10-21
 */

// Copyright (c) 2026, Lewen Qiu and Huang Chen.
// Portions copyright (c) 2023, Lewen Qiu, Zhengguang Liu,
// Hongbo Yao, and Jingtian Tang.
// This file is part of the DC3DPAINV program. DC3DPAINV is free
// software with source code available at
// https://github.com/luowen765/DC3DPAINV. You can redistribute it
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
