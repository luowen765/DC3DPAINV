/*
 * @Description: MFEM extensions for DC3DPAINV: tetrahedral coefficient classes, a
 *   solver wrapper, mesh output, and coefficient classes adapted from
 *   MFEM.
 * @Author: Lewen Qiu (original DC3DPAFEM module); portions adapted from MFEM
 *   (BSD-3); adapted for DC3DPAINV by Lewen Qiu and Huang Chen
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

#include "em.h"

#include "mfem.hpp"
using namespace mfem;

inline void compute_normal(mfem::ElementTransformation &T,
                           mfem::Vector &normal) {
  CalcOrtho(T.Jacobian(), normal);
  normal /= normal.Norml2();

}

inline int get_tetId_bdr(std::shared_ptr<ParMesh> pmesh, int ElementNo) {

  int tet_id = -1;
  int flag = -1;

  mfem::FaceElementTransformations *face_trans =
      pmesh->GetBdrFaceTransformations(ElementNo);

  mfem::ElementTransformation *e1 = face_trans->Elem1;
  if (e1 != NULL) {
    tet_id = e1->Attribute - 1;
    flag *= -1;
  }
  mfem::ElementTransformation *e2 = face_trans->Elem2;
  if (e2 != NULL) {
    tet_id = e2->Attribute - 1;
    flag *= -1;
  }
  myassert(flag == 1 && tet_id != -1);
  return tet_id;
}

double B(Point_double &x, Point_double &x_c, Matrix3D &tet_sigma);
double beta(double B, Point_double &x, Point_double &x_c, Point_double &n);
double Uis(Matrix3D &sigma, Point_double &x, Point_double &x_s,
           int if_s_on_surface);
Point_double gradient_Uis(Matrix3D &sigma, Point_double &x, Point_double &x_s,
                          int if_s_on_surface);

#ifndef _SOLVERSPLUS_h
#define _SOLVERSPLUS_h

class SolversPlus : public CGSolver {

public:
  SolversPlus();
  SolversPlus(MPI_Comm comm_);

  void Mult(const Vector &b, Vector &x);
};

#endif

#ifndef _MESHPLUS_H
#define _MESHPLUS_H

class Meshplus : public Mesh {

public:
  Meshplus();
  Meshplus(ParMesh *pmesh_);
  Meshplus(Mesh *mesh_);

  void PrintVTK_eta(std::ostream &out, Vector &other_attri, int n_value = 1);

  void write_out_node_info_vtk(std::ostream &out, Vector &node_data);
};

#endif

#ifndef MFEM_COEFFICIENTS_H
#define MFEM_COEFFICIENTS_H

class LVCoefficient : public MatrixCoefficient {
private:
  std::vector<DenseMatrix> mat;

public:

  LVCoefficient(std::vector<DenseMatrix> &m

                )
      : MatrixCoefficient(m[0].Height(), m[0].Width()), mat(m)

  {}
  using MatrixCoefficient::Eval;

  virtual void Eval(DenseMatrix &M, ElementTransformation &T,
                    const IntegrationPoint &ip)

      ;
};

class LGamma1Coefficient : public Coefficient {
private:
  std::vector<DenseMatrix> mat;
  Vector x_c;

  std::shared_ptr<ParMesh> pmesh;

public:

  LGamma1Coefficient(std::vector<DenseMatrix> &m, Vector sources_center,

                     std::shared_ptr<ParMesh> pmesh_)
      : mat(m), x_c(sources_center), pmesh(pmesh_) {}
  using Coefficient::Eval;

  virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class RVCoefficient : public VectorCoefficient {
private:
  std::vector<DenseMatrix> mat;
  DenseMatrix sigma0;
  Vector x_i;
  int if_s_on_surface;

public:

  RVCoefficient(std::vector<DenseMatrix> &m, DenseMatrix sigma0_,
                Vector source_i, int if_s_on_surface_

                )
      : VectorCoefficient(3), mat(m), sigma0(sigma0_), x_i(source_i),
        if_s_on_surface(if_s_on_surface_)

  {}
  using VectorCoefficient::Eval;

  virtual void Eval(Vector &V, ElementTransformation &T,
                    const IntegrationPoint &ip);
};

class RGamma0Coefficient : public Coefficient {
private:
  DenseMatrix sigma0;
  Vector x_i;
  int if_s_on_surface;

public:

  RGamma0Coefficient(DenseMatrix sigma0_, Vector source_i, int if_s_on_surface_

                     )
      : sigma0(sigma0_), x_i(source_i), if_s_on_surface(if_s_on_surface_)

  {}
  using Coefficient::Eval;

  virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class RGamma1Coefficient : public Coefficient {
private:
  std::vector<DenseMatrix> mat;
  DenseMatrix sigma0;
  Vector x_i;
  int if_s_on_surface;

  std::shared_ptr<ParMesh> pmesh;

public:

  RGamma1Coefficient(std::vector<DenseMatrix> &m, DenseMatrix sigma0_,
                     Vector source_i, int if_s_on_surface_,

                     std::shared_ptr<ParMesh> pmesh_)
      : mat(m), sigma0(sigma0_), x_i(source_i),
        if_s_on_surface(if_s_on_surface_), pmesh(pmesh_) {}
  using Coefficient::Eval;

  virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

#endif
