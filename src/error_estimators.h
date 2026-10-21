/*
 * @Description: Declarations of the a posteriori error estimators used for
 *   forward-mesh adaptivity: current-density jump and SPR patch recovery.
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

#ifndef _ERROR_ESTIMATORS_H
#define _ERROR_ESTIMATORS_H

#include <cstdlib>

#include <set>
#include <string>
#include <vector>

#include "em.h"
#include "mfem.hpp"

using namespace mfem;

class Estimator {
protected:

  std::shared_ptr<ParMesh> pmesh;

  std::shared_ptr<ParFiniteElementSpace> pfes;

  std::vector<std::shared_ptr<ParGridFunction>> up;

  std::shared_ptr<ParGridFunction> w;

public:
  Estimator(

      std::shared_ptr<ParMesh> pmesh_,

      std::shared_ptr<ParFiniteElementSpace> pfes_,

      std::vector<std::shared_ptr<ParGridFunction>> up_,

      std::shared_ptr<ParGridFunction> w_
      );
  virtual ~Estimator();

public:

  virtual void get_error_estimates(bool if_solve_dual, Vector &errors) = 0;

};

class nJEstimator : public Estimator {
public:
  std::vector<DenseMatrix> &cond_att;
  nJEstimator(

      std::shared_ptr<ParMesh> pmesh_,

      std::shared_ptr<ParFiniteElementSpace> pfes_,

      std::vector<std::shared_ptr<ParGridFunction>> up_,

      std::shared_ptr<ParGridFunction> w_,
      std::vector<DenseMatrix> &att_cond);
  ~nJEstimator();

  double n_dot_J(bool if_loc1, bool if_shared_tet2,
                 FaceElementTransformations *face_elem_trans,
                 ElementTransformation *face_trans,
                 ElementTransformation *tet_trans, const FiniteElement *tet_fe,
                 int tet_id, const IntegrationPoint &ip,

                 std::shared_ptr<ParGridFunction> u);

  void compute_face_err(bool if_shared_tet2,
                        FaceElementTransformations *face_elem_trans,
                        double *face_err_u, double &face_err_w);

  void get_error_estimates(bool if_solve_dual, Vector &errors);
};

class SPREstimator : public Estimator {
public:
  std::vector<DenseMatrix> &cond_att;

  SPREstimator(

      std::shared_ptr<ParMesh> pmesh_,

      std::shared_ptr<ParFiniteElementSpace> pfes_,

      std::vector<std::shared_ptr<ParGridFunction>> up_,

      std::shared_ptr<ParGridFunction> w_,
      std::vector<DenseMatrix> &att_cond,
      const unsigned int s = 10);

  ~SPREstimator(){};

  void gradient_patch_recovery(std::vector<DenseVector> &Ui, DenseVector &W,
                               std::vector<DenseMatrix> &cond_att,
                               Vector &local_err, bool if_solve_dual);

  void get_error_estimates(bool if_solve_dual, Vector &errors);

protected:

  void get_element_patch(unsigned int elem,
                         std::set<unsigned int> &patch_id,
                         const unsigned int s,

                         std::vector<DenseMatrix> &cond_att);

  void get_patch_polynomial_c(std::vector<double> &c,
                              const Point_double &p);

  void get_patch_b(std::set<unsigned int> &patch_id,
                   std::vector<DenseVector> &solution,
                   std::vector<Matrix43D> &b);

  void compute_PI_k(std::set<unsigned int> &patch_id,
                    std::vector<DenseVector> &solution,
                    std::vector<Matrix43D> &a,

                    std::vector<DenseMatrix> &cond_att,
                    std::map<unsigned int, double> &energy_norm,
                    bool if_solve_dual);

  void compute_C(std::set<unsigned int> &patch_id, Matrix4D &C);
  void solve_a(Matrix4D &C, std::vector<Matrix43D> &b,
               std::vector<Matrix43D> &a);
  void compute_P_value(Point_double &r, Matrix43D &a, Matrix13D &P);

private:

  unsigned int _s;
};

#endif
