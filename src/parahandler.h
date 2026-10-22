/*
 * @Description: Declaration of the input manager for control files, electrodes, data
 *   weights, conductivity models, and mesh refinement.
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

#ifndef _PARAHANDLER_H
#define _PARAHANDLER_H

#include <fstream>
#include <map>
#include <math.h>
#include <string>
#include <vector>

#include "em.h"

#include "mfem.hpp"

using namespace EM;
using namespace mfem;
class ParaHandler {
public:

  int if_anisotropy;
  int max_iter;
  int N_M;
  int N_line_search;

  std::string obs_data_file;

  double stop_misfit;
  double noise_floor;

  double init_lambda;
  double atte_lambda;
  double lambda_atte_thre;
  double lambda_floor;
  int lambda_keep;

  double min_cond;
  double max_cond;

  double cp;
  double relative_misfit_for_up;
  double relative_misfit_for_down;
  double relative_misfit_inv_up;
  double relative_misfit_inv_down;

  int GN_PCG_max_iter;
  double GN_PCG_TOL;
  std::string inv_algorithm;

  int adInv_max_iter;
  int invMesh_keep;

  double inv_amr_ratio;

  int N_electrodes;
  std::vector<Point_double> electrodes;
  std::vector<int> is_electrode_on_surface;

  std::vector<Vertex> sources;

  std ::vector<int> is_source_on_surface;

  int sites_number;
  std::vector<Vertex> sites;

  int msn;

  std::vector<int> ms_locations;

  int record_number_total;

  std::vector<std::vector<int>> measure_sources_locId;

  std::vector<std::vector<std::vector<int>>> measure_sites_gather_locId;

  std::map<Point_double, int> c_map;
  std::map<Point_double, int> p_map;

  Array<int> local_sites_tets;

  std::string model_parameters_file;
  int n_regions;

  std::vector<int> inv_marker;
  std::vector<int>
      marker_vec;
  std::map<int, DenseMatrix> region2conductivity;

  std::string mesh_file;

  int N_refine_electrodes_local_inv;

  int N_refine_receivers_local_for;

  int amr_type;
  int max_amr;

  double amr_ratio;

  long int max_dofs;
  std::string linear_solver;

  int pcg_maxit;
  double pcg_primal_tol;
  double pcg_dual_tol;

  int survey_mode;

  REV d_obs;
  RESM Wd;
  int valid_data;

  REV noise;

  int inv_data_type;
  REV K_coef;

  int integration_order;

  int n_group = 0;
  MPI_Comm group_comm;
  std::vector<int> task_source;

  int n_group_s = 0;

  std::vector<int> task_ms;
  int n_group_ms = 0;

  int group_id = 0;
  int group_rank = 0;
  int group_size = 0;

  bool if_parallel_source = false;

public:
  ParaHandler(char *model_file);
  ~ParaHandler();

public:

  void skip_comments(std::istream &in, std::vector<std::string> &para_str_vec,
                     std::string comment_str = "#");

  void read_model_info(char *model_file);

  DenseMatrix cal_conductivity(Vector &main_cond, Vector &_angle);

  DenseMatrix get_elem_conductivity(int marker);

  Vector get_sources_center();

  Vector get_sites_center();

  void refine_locally(std::shared_ptr<Mesh> mesh, std::vector<Vertex> &points,
                      int refine_level);

  void refine_locally(std::shared_ptr<Mesh> mesh,
                      std::vector<Point_double> &points, int refine_level);

  void find_point_tets(

      std::shared_ptr<ParMesh> pmesh, const std::vector<Vertex> &points,
      Array<int> &find_tets);

  void dual_parallel_resource_allocation(int world_rank, int world_size);
};

#endif
