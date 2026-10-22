/*
 * @Description: Declaration of the base class that implements the shared inversion
 *   workflow.
 * @Author: Lewen Qiu, Huang Chen
 * @Date: 2026-10-21
 */

// Copyright (c) 2026, Lewen Qiu and Huang Chen.
// This file is part of the DC3DPAINV program. DC3DPAINV is free
// software with source code available at
// https://github.com/luowen765/DC3DPAINV. You can redistribute it
// or modify it under the terms of the BSD-3 license. See the
// LICENSE file for details.

#ifndef _INV_BASE_H
#define _INV_BASE_H

#include "em.h"

#include "mfem.hpp"
#include "parahandler.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <sys/stat.h>
#include <vector>

using namespace mfem;
using namespace EM;

class InvBase {
public:

  ParaHandler &inv_para;

  int world_rank;
  int world_size;

  std::shared_ptr<double> Trans_mat;

  std::shared_ptr<double> cond_k;

  std::vector<int> inv_att_init;
  std::vector<DenseMatrix> cond_att;

  std::shared_ptr<Mesh> mesh_inv;

  std::shared_ptr<ParMesh> pmesh;

  std::vector<int> forward_inv;

  std::map<int, std::vector<int>> inv_forward_map;

  std::vector<int> last_forward_inv;

  std::vector<int> inv_forward;

  int N_inv_cell;
  int N_m;
  int N_d;
  REV m_ref;

  REV d_model;

  RESM WmTWm;

  std::vector<REV> loc_d_obs;
  std::vector<REV> loc_Kcoef;
  std::vector<RESM> loc_Wd;

  REM JT;
  REV JT_V;

  REV m_k;
  REV g_k;
  REV update_dire;
  REV m_search, m_best, d_best, JTV_best;

  std::vector<double> objective_func;
  std::vector<double> lamda;
  std::vector<double> misfit;
  std::vector<double> roughness;
  std::vector<double>
      N_inv_cells;

  int N_fo_ver_k;
  std::vector<int> N_fo_ver;

  std::string inv_algorithm;

  std::vector<int> if_amr_s;

  std::vector<int> if_inv_amr_s;

  int if_amr_forward = 0;
  int if_amr_inv = 0;
  double initial_misfit;
  double last_misfit;
  double max_misfit_for = 0.0;
  double max_misfit_inv = 0.0;
  int count_inv_iter = 0;
  int judge_inv_amr_min = 0;

  int judge_lambda = 0;
  double lamda_k = 0;

  int if_forward_lineSearch = 0;

  bool pmesh_amr_flag = false;

  int inv_mesh_keep;

  int count_forward_times = 0;
  double count_forward_time;
  int count_amr_for_times = 0;
  int count_amr_inv_times = 0;

  int N_iter_adlbfgs =0;

public:
  InvBase(ParaHandler &inv_para_, int world_rank_, int world_size_,

          std::string inv_algorithm_ = "NULL");

  virtual ~InvBase();
  void init();

  void set_model_parameter_convert(double upper, double lower, double cp,
                                   int &kind, REV &m_k_T,

                                   std::shared_ptr<double> m_k,

                                   std::shared_ptr<double> C);

  void save_cond_mesh(std::string &iter_name);

  void save_JTV_mesh(std::string &iter_name);

  void save_predicted_data(std::string &iter_name,
                           std::string &survery_mode_file);

  void StrongWolfeConditionsLineSearch(int N_iter);

  void update_common_inv_para(int N_iter);

  virtual void update_model_direction(int N_iter) = 0;

  void output_inversion_parameters(std::string inv_reslut_folder);

  void clear();

  virtual void resetInvParas(int N_iter);

  void forward_modeling(int N_iter);

  void compute_err_JTV(Vector &err_model_JTV);

  void refine_inv_mesh(std::string inv_reslut_folder);

  void reset_invpara_REV(REV &para);

  void reset_inv_for_map();
  void transfer_REV(REV &vec);

  void transfer_Cond_Trans();

  void modify_cond_att();

  void save_mesh(std::string inv_reslut_folder, Vector &current_inv_mesh);

  void compute_WmTWm();

  double vector_threshold(Vector &vec, double ratio);
};

#endif
