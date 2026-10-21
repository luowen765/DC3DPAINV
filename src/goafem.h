/*
 * @Description: Declaration of the parallel goal-oriented adaptive finite-element
 *   forward and adjoint solver.
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

#ifndef _GOAFEM_H
#define _GOAFEM_H

#include "em.h"
#include "mfem.hpp"
#include "parahandler.h"

#include <mpi.h>

#include <iostream>
#include <vector>

using namespace EM;

using namespace mfem;

class GOAFEM {
public:

  std::vector<DenseMatrix> &cond_att;

  std::vector<DenseMatrix> sigma0;

  ParaHandler &para_handler;

  std::shared_ptr<ParMesh> pmesh;

  std::shared_ptr<FiniteElementCollection> pfec;

  std::shared_ptr<ParFiniteElementSpace> pfes;

  std::shared_ptr<ParBilinearForm> a;

  std::vector<std::shared_ptr<ParGridFunction>> up;

  std::vector<std::shared_ptr<ParLinearForm>> f;

  std::shared_ptr<ParGridFunction> w;

  std::shared_ptr<ParLinearForm> l;

  std::vector<std::shared_ptr<ParGridFunction>> xgf;

  std::vector<std::shared_ptr<ParLinearForm>> y;

  Array<int> gamma0_bdr;
  Array<int> gamma1_bdr;

  Array<int> ess_tdof_list;

  std::shared_ptr<MatrixCoefficient> LVolume_coef;

  std::shared_ptr<Coefficient> LGamma1_coef;

  std::shared_ptr<Coefficient> RGamma0_coef;
  std::shared_ptr<Coefficient> RGamma1_coef;

  bool if_solve_dual = false;
  bool if_solve_adjoint = false;

  OperatorHandle A;

  std::vector<Vector> U, F;
  Vector W, L;

  std::vector<Vector> X, Y;

  Vector local_err;
  bool if_refine_mesh = false;

  int order;

  std::string invAlgorithm;

  int N_d;
  int group_nd;
  int N_m;

  int ns = 0;
  int group_ns = 0;

  int group_msn = 0;

  int N_adjoint = 0;

  Array<int>
      global_initial_sources_tets;

  std::vector<REV> up_all;

public:
  GOAFEM(std::vector<DenseMatrix> &att_cond, ParaHandler &para_handler_,

         std::shared_ptr<ParMesh> pmesh, std::string &invAlgorithm_);
  ~GOAFEM();

  void init();
  void reconstruct_FEM_terms();

  void set_dual_adjoint_problem(bool dual_problem_, bool adjoint_problem_);

  void one_forward_iteration(REV &d_model, REV &JT_V, REM &JT,
                             std::vector<REV> &loc_Koef,
                             std::vector<REV> &loc_d_obs,
                             std::vector<RESM> &loc_Wd,
                             std::map<int, std::vector<int>> &inv_forward_map,
                             std::shared_ptr<double> Trans_mat, int N_inv_cell

  );

  void set_sigma0();

  void setup_integral_coefficient();
  void set_bdr();

  void assemble_dual_linearform(Array<int> &local_dual_tets_);

  void solve_with_pcg(OperatorHandle &A, Vector &X, Vector &B, std::string pre,
                      int maxit, std::string problem_type

  );

  void assemble_adjoint_linearform(int adjointId, REV &vec);

  void assemble_KmE(int msId, RESM &Km_E,
                    std::map<int, std::vector<int>> &inv_forward_map,

                    std::shared_ptr<double> Trans_mat, int N_inv_cell,
                    std ::vector<int> &is_source_on_surface);

  void compute_jtv(

    REV & X_,REV & JTV_temp,
                    std::map<int, std::vector<int>> &inv_forward_map,

                    std::shared_ptr<double> Trans_mat, int N_inv_cell,
                    std ::vector<int> &is_source_on_surface);

  void save_jtu_as_one(REV &JT_V_);

  void save_jt_as_one(REM &JT_);

  void group_save_jtu_as_one(REV &JT_V_);
  void group_save_jt_as_one(REM &JT_);

  void save_up_as_one_sources();

  void refine_mesh();

  void error_estimating();

  void update();

  void compute_d_obs(REV &d_model, int world_rank);

  void assemble_adjoint_single_site(bool if_site_M, int tet_id, Vector pt,
                                    int adjointId, double vecI = 0.0);

  void assemble_RV_linearform(std::vector<std::shared_ptr<ParLinearForm>> f_);
};

#endif
