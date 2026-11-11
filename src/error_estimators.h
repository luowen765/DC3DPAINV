/*
 * @File: src/error_estimators.h
 * @Description: Declaration of the a posteriori error estimator used for
 *   forward-mesh adaptivity: the normal current-density jump.
 * @Date: 2026-10-21
 * @License: BSD-3-Clause (see LICENSE)
 */

// Copyright (c) 2026, Lewen Qiu and Huang Chen.
// Portions copyright (c) 2023, Lewen Qiu, Zhengguang Liu,
// Hongbo Yao, and Jingtian Tang (original DC3DPAFEM module).
// This file is part of the DC3DPAINV program. DC3DPAINV is free
// software with source code available at
// https://github.com/luowen765/DC3DPAINV. You can redistribute it
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

#endif
