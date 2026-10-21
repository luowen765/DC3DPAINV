/*
 * @Description: Implementation of input parsing and preprocessing for DC3DPAINV.
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

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <sstream>

#include "parahandler.h"

ParaHandler::ParaHandler(char *model_file) {
  this->read_model_info(model_file);
}

ParaHandler::~ParaHandler() {}
void ParaHandler::skip_comments(std::istream &in,
                                std::vector<std::string> &para_str_vec,
                                std::string comment_str) {
  std::string line;
  while (std::getline(in, line)) {
    for (char &c : line)
    {
      if (c == '\t' || c == ',' || c == ';' || c == '\r' || c == '\n') {
        c = ' ';
      }
    }

    line.erase(0, line.find_first_not_of(" "));

    line.erase(line.find_last_not_of(" ") + 1);

    int n_comment_start = line.find_first_of(comment_str);
    if (n_comment_start != std::string::npos) {
      line.erase(n_comment_start);
    }

    if (line.empty())
      continue;

    para_str_vec.push_back(line);
  }
}

void ParaHandler::read_model_info(char *model_file) {
  std::ifstream in_stream(model_file);
  myassert(in_stream.good());
  int para_lines = 45;
  std::vector<std::string> para_str_vec;
  skip_comments(in_stream, para_str_vec);
  myassert(para_str_vec.size() == para_lines);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (world_rank == 0) {
    std::cout << "print input parameters" << std::endl;
    for (int i = 0; i < para_lines; i++) {
      std::cout << para_str_vec[i] << std::endl;
    }
  }

  in_stream.close();
  std::stringstream ss;

  ss << para_str_vec[0];
  ss >> if_anisotropy;
  ss.clear();
  myassert(if_anisotropy == 0 || if_anisotropy == 1 || if_anisotropy == 2);
  ss << para_str_vec[1];
  ss >> max_iter;
  ss.clear();
  myassert(max_iter > 0);
  ss << para_str_vec[2];
  ss >> N_M;
  ss.clear();
  myassert(N_M >= 1);
  ss << para_str_vec[3];
  ss >> N_line_search;
  ss.clear();
  myassert(N_line_search >= 2);

  ss << para_str_vec[4];
  ss >> obs_data_file;
  ss.clear();

  ss << para_str_vec[5];
  ss >> inv_data_type;
  ss.clear();
  myassert(inv_data_type == 0 || inv_data_type == 1);

  ss << para_str_vec[6];
  ss >> stop_misfit;
  ss.clear();
  myassert(stop_misfit >= 1.0);

  ss << para_str_vec[7];
  ss >> noise_floor;
  ss.clear();
  myassert(noise_floor > 0);

  ss << para_str_vec[8];
  ss >> init_lambda;
  ss.clear();
  myassert(init_lambda > 0);

  ss << para_str_vec[9];
  ss >> atte_lambda;
  ss.clear();
  myassert(atte_lambda > 0);

  ss << para_str_vec[10];
  ss >> lambda_atte_thre;
  ss.clear();
  myassert(lambda_atte_thre > 0.0);

  ss << para_str_vec[11];
  ss >> lambda_floor;
  ss.clear();
  myassert(lambda_floor > 0.0);
  ss << para_str_vec[12];
  ss >> lambda_keep;
  ss.clear();
  myassert(lambda_keep >= 1);
  ss << para_str_vec[13];
  ss >> min_cond;
  ss.clear();

  ss << para_str_vec[14];
  ss >> max_cond;
  ss.clear();

  ss << para_str_vec[15];
  ss >> cp;
  ss.clear();
  myassert(cp >= 1);

  ss << para_str_vec[16];
  ss >> GN_PCG_max_iter;
  ss.clear();
  myassert(GN_PCG_max_iter > 0);

  ss << para_str_vec[17];
  ss >> GN_PCG_TOL;
  ss.clear();
  myassert(GN_PCG_TOL > 0);

  ss << para_str_vec[18];
  ss >> inv_algorithm;
  ss.clear();
  myassert(inv_algorithm == "lbfgs" || inv_algorithm == "gn");

  ss << para_str_vec[19];
  ss >> relative_misfit_inv_up;
  ss.clear();
  myassert(relative_misfit_inv_up > 0 && relative_misfit_inv_up < 1);

  ss << para_str_vec[20];
  ss >> relative_misfit_inv_down;
  ss.clear();
  myassert(relative_misfit_inv_down > 0 && relative_misfit_inv_down < 1);

  ss << para_str_vec[21];
  ss >> N_refine_electrodes_local_inv;
  ss.clear();
  myassert(N_refine_electrodes_local_inv >= 0);

  ss << para_str_vec[22];
  ss >> adInv_max_iter;
  ss.clear();
  myassert(adInv_max_iter >= 0);

  ss << para_str_vec[23];
  ss >> invMesh_keep;
  ss.clear();
  myassert(invMesh_keep > 0);

  ss << para_str_vec[24];
  ss >> inv_amr_type;
  ss.clear();
  myassert(inv_amr_type == 0 || inv_amr_type == 1 || inv_amr_type == 2);

  ss << para_str_vec[25];
  ss >> inv_amr_ratio_model;
  ss.clear();

  ss << para_str_vec[26];
  ss >> inv_amr_ratio_data;
  ss.clear();

  ss << para_str_vec[27];
  ss >> h_to_surface;
  ss.clear();
  myassert(h_to_surface >= 0);

  ss << para_str_vec[28];
  ss >> model_parameters_file;
  ss.clear();

  ss << para_str_vec[29];
  ss >> mesh_file;
  ss.clear();

  ss << para_str_vec[30];
  ss >> relative_misfit_for_up;
  ss.clear();
  myassert(relative_misfit_for_up > 0 && relative_misfit_for_up < 1);

  ss << para_str_vec[31];
  ss >> relative_misfit_for_down;
  ss.clear();
  myassert(relative_misfit_for_down > 0 && relative_misfit_for_down < 1);

  ss << para_str_vec[32];
  ss >> N_refine_receivers_local_for;
  ss.clear();
  myassert(N_refine_receivers_local_for >= 0);

  ss << para_str_vec[33];
  ss >> max_amr;
  ss.clear();
  myassert(max_amr >= 0);

  ss << para_str_vec[34];
  ss >> amr_type;
  ss.clear();
  myassert(amr_type == 0 || amr_type == 1);

  ss << para_str_vec[35];
  ss >> amr_ratio;
  ss.clear();
  myassert(amr_ratio > 0);

  ss << para_str_vec[36];
  ss >> max_dofs;
  ss.clear();
  myassert(max_dofs >= 10);

  ss << para_str_vec[37];
  ss >> linear_solver;
  ss.clear();
  myassert(linear_solver == "amg" || linear_solver == "ilu" ||
           linear_solver == "mumps");

  ss << para_str_vec[38];
  ss >> find_points_by;
  ss.clear();
  myassert(find_points_by == "FindPoints");

  ss << para_str_vec[39];
  ss >> pcg_maxit;
  ss.clear();
  myassert(pcg_maxit > 5);

  ss << para_str_vec[40];
  ss >> pcg_primal_tol;
  ss.clear();
  myassert(pcg_primal_tol > 0 && pcg_primal_tol < 1);

  ss << para_str_vec[41];
  ss >> pcg_dual_tol;
  ss.clear();
  myassert(pcg_dual_tol > 0 && pcg_dual_tol < 1 &&
           pcg_dual_tol >= pcg_primal_tol);

  ss << para_str_vec[42];
  ss >> error_esti_way;
  ss.clear();
  myassert(error_esti_way == "jn" || error_esti_way == "zz_mfem" ||
           error_esti_way == "zz_patch" || error_esti_way == "jn_zz");

  ss << para_str_vec[43];
  ss >> integration_order;
  ss.clear();
  myassert(integration_order == 1 || integration_order == 2 ||
           integration_order == 3);

  ss << para_str_vec[44];
  ss >> n_group;
  ss.clear();
  myassert(n_group > 0);

  std::ifstream survey_instream(obs_data_file.c_str());
  myassert(survey_instream.good());

  survey_instream >> N_electrodes;
  myassert(N_electrodes > 0);
  for (int i = 0; i < N_electrodes; i++) {
    Point_double electrode;
    int is_on_earth_surface;
    survey_instream >> electrode(0) >> electrode(1) >> electrode(2) >>
        is_on_earth_surface;
    myassert(is_on_earth_surface == 0 || is_on_earth_surface == 1);
    electrodes.push_back(electrode);
    is_electrode_on_surface.push_back(is_on_earth_surface);
  }
  myassert(electrodes.size() == N_electrodes);

  unsigned int n_data = 0;
  int temp_mode = 0;

  survey_instream >> n_data
      >> temp_mode;

  myassert(n_data > 0);
  myassert(temp_mode == 11 || temp_mode == 12 || temp_mode == 21 ||
           temp_mode == 22 || temp_mode == 41);

  survey_mode = temp_mode;

  double obs_u_temp, noise_u_temp, obs_rho_temp, noise_rho_temp;
  Point_double point_s, point_m, point_s2, point_m2;
  PointPair point_s_pair, point_m_pair;
  std::vector<double> obs_v, noise_v;
  std::set<Point_double> temp_s, temp_m;
  typedef std::pair<std::set<Point_double>::iterator, bool> stlpair;
  std::set<PointPair> temp_s_pair, temp_m_pair;
  typedef std::pair<std::set<PointPair>::iterator, bool> stlpair_Pointpair;
  std::vector<Point_double> temp_point_s;
  std::vector<Point_double> temp_point_m;
  std::vector<std::vector<Point_double>> measure_sources;
  std::vector<std::vector<Point_double>> measure_sites;
  std::vector<std::vector<std::vector<Point_double>>> measure_sites_gather;

  if (inv_data_type == 1)
    K_coef.resize(n_data);

  for (int i = 0; i < n_data; i++) {
    if (survey_mode == 11) {

      int s_id = -1, m_id = -1;
      survey_instream >> s_id >> m_id;
      for (int j = 0; j < 3; j++) {
        point_s(j) = electrodes[s_id - 1](j);
        point_m(j) = electrodes[m_id - 1](j);
      }

      if (inv_data_type == 1) {
        double AM = length_two_point(point_s, point_m);
        double k = 2.0 * PI * AM;

        K_coef(i) = k;
      }

      stlpair pair_s = temp_s.insert(point_s);
      stlpair pair_m = temp_m.insert(point_m);

      if (pair_s.second == true) {
        myassert(temp_point_s.size() == 0);
        temp_point_s.push_back(point_s);
        measure_sources.push_back(temp_point_s);
        temp_point_s.clear();
        if (i > 0) {
          myassert(measure_sites.size() > 0);
          measure_sites_gather.push_back(measure_sites);
        }
        measure_sites.clear();
      }
      myassert(temp_point_m.size() == 0);
      temp_point_m.push_back(point_m);
      measure_sites.push_back(temp_point_m);
      temp_point_m.clear();
      if (i == n_data - 1) {
        myassert(measure_sites.size() > 0);
        measure_sites_gather.push_back(measure_sites);
        measure_sites.clear();
      }

    } else if (survey_mode == 22) {

      int s1_id = -1, s2_id = -1, m1_id = -1, m2_id = -1;
      survey_instream >> s1_id >> s2_id >> m1_id >> m2_id;
      for (int j = 0; j < 3; j++) {
        point_s(j) = electrodes[s1_id - 1](j);
        point_s2(j) = electrodes[s2_id - 1](j);
        point_m(j) = electrodes[m1_id - 1](j);
        point_m2(j) = electrodes[m2_id - 1](j);
      }

      point_s_pair = {point_s, point_s2};
      point_m_pair = {point_m, point_m2};

      if (inv_data_type == 1) {
        double AM = length_two_point(point_s, point_m);
        double BM = length_two_point(point_s2, point_m);
        double AN = length_two_point(point_s, point_m2);
        double BN = length_two_point(point_s2, point_m2);
        double factor = 1.0 / AM - 1.0 / BM - 1.0 / AN + 1.0 / BN;
        double k = 2 * PI * 1.0 / factor;
        K_coef(i) = k;
      }

      stlpair pair_s1 = temp_s.insert(point_s);
      stlpair pair_s2 = temp_s.insert(point_s2);
      stlpair_Pointpair pair_s_pair = temp_s_pair.insert(point_s_pair);
      stlpair pair_m1 = temp_m.insert(point_m);
      stlpair pair_m2 = temp_m.insert(point_m2);
      stlpair_Pointpair pair_m_pair = temp_m_pair.insert(point_m_pair);

      if (pair_s_pair.second == true) {
        myassert(temp_point_s.size() == 0);
        temp_point_s.push_back(point_s);
        temp_point_s.push_back(point_s2);

        measure_sources.push_back(temp_point_s);
        temp_point_s.clear();
        if (i > 0) {
          myassert(measure_sites.size() > 0);
          measure_sites_gather.push_back(measure_sites);
        }
        measure_sites.clear();
      }
      myassert(temp_point_m.size() == 0);
      temp_point_m.push_back(point_m);
      temp_point_m.push_back(point_m2);
      measure_sites.push_back(temp_point_m);
      temp_point_m.clear();
      if (i == n_data - 1) {
        myassert(measure_sites.size() > 0);
        measure_sites_gather.push_back(measure_sites);
        measure_sites.clear();
      }

    } else if (survey_mode == 21) {

      int s1_id = -1, s2_id = -1, m_id = -1;
      survey_instream >> s1_id >> s2_id >> m_id;
      for (int j = 0; j < 3; j++) {
        point_s(j) = electrodes[s1_id - 1](j);
        point_s2(j) = electrodes[s2_id - 1](j);
        point_m(j) = electrodes[m_id - 1](j);
      }

      point_s_pair = {point_s, point_s2};

      if (inv_data_type == 1) {
        double AM = length_two_point(point_s, point_m);
        double BM = length_two_point(point_s2, point_m);
        double factor = 1.0 / AM - 1.0 / BM - 0 + 0;
        double k = 2 * PI * 1.0 / factor;

        K_coef(i) = k;
      }

      stlpair pair_s1 = temp_s.insert(point_s);
      stlpair pair_s2 = temp_s.insert(point_s2);
      stlpair_Pointpair pair_s_pair = temp_s_pair.insert(point_s_pair);
      stlpair pair_m1 = temp_m.insert(point_m);

      if (pair_s_pair.second == true) {
        myassert(temp_point_s.size() == 0);
        temp_point_s.push_back(point_s);
        temp_point_s.push_back(point_s2);
        measure_sources.push_back(temp_point_s);
        temp_point_s.clear();
        if (i > 0) {
          myassert(measure_sites.size() > 0);
          measure_sites_gather.push_back(measure_sites);
        }
        measure_sites.clear();
      }
      myassert(temp_point_m.size() == 0);
      temp_point_m.push_back(point_m);
      measure_sites.push_back(temp_point_m);
      temp_point_m.clear();
      if (i == n_data - 1) {
        myassert(measure_sites.size() > 0);
        measure_sites_gather.push_back(measure_sites);
        measure_sites.clear();
      }

    } else if (survey_mode == 12) {

      int s_id = -1, m1_id = -1, m2_id = -1;
      survey_instream >> s_id >> m1_id >> m2_id;
      for (int j = 0; j < 3; j++) {
        point_s(j) = electrodes[s_id - 1](j);
        point_m(j) = electrodes[m1_id - 1](j);
        point_m2(j) = electrodes[m2_id - 1](j);
      }

      point_m_pair = {point_m, point_m2};

      if (inv_data_type == 1) {
        double AM = length_two_point(point_s, point_m);
        double AN = length_two_point(point_s, point_m2);
        double factor = 1.0 / AM - 1.0 / AN;
        double k = 2 * PI * 1.0 / factor;

        K_coef(i) = k;
      }

      stlpair pair_s1 = temp_s.insert(point_s);
      stlpair pair_m1 = temp_m.insert(point_m);
      stlpair pair_m2 = temp_m.insert(point_m2);
      stlpair_Pointpair pair_m_pair = temp_m_pair.insert(point_m_pair);

      if (pair_s1.second == true) {
        myassert(temp_point_s.size() == 0);
        temp_point_s.push_back(point_s);
        measure_sources.push_back(temp_point_s);
        temp_point_s.clear();
        if (i > 0) {
          myassert(measure_sites.size() > 0);
          measure_sites_gather.push_back(measure_sites);
        }
        measure_sites.clear();
      }
      myassert(temp_point_m.size() == 0);
      temp_point_m.push_back(point_m);
      temp_point_m.push_back(point_m2);
      measure_sites.push_back(temp_point_m);
      temp_point_m.clear();
      if (i == n_data - 1) {
        myassert(measure_sites.size() > 0);
        measure_sites_gather.push_back(measure_sites);
        measure_sites.clear();
      }

    } else {
      std::cout << "fail to get measure_sources, unsopported mode!\n";
      std::abort();
    }

    survey_instream >> obs_u_temp >> noise_u_temp >> obs_rho_temp >>
        noise_rho_temp;

    if (inv_data_type == 0) {
      obs_v.push_back(obs_u_temp);
      noise_v.push_back(noise_u_temp);
    } else {
      myassert(inv_data_type == 1);
      obs_v.push_back(obs_rho_temp);
      noise_v.push_back(noise_rho_temp);
    }

  }

  survey_instream.close();

  for (int i = 0; i < K_coef.size(); i++) {

    if (!std::isfinite(K_coef(i)) || std::abs(K_coef(i)) < 1e-8 ||
        std::abs(K_coef(i)) > 1e8) {

      K_coef[i] = 0.0;
    }
  }

  std::vector<Point_double> tempC(temp_s.begin(), temp_s.end());

  int ns = tempC.size();
  sources.resize(ns);
  is_source_on_surface.resize(ns);
  for (int i = 0; i < tempC.size(); i++) {
    sources[i](0) = tempC[i](0);
    sources[i](1) = tempC[i](1);
    sources[i](2) = tempC[i](2);
    Point_double source;
    source(0) = tempC[i](0);
    source(1) = tempC[i](1);
    source(2) = tempC[i](2);
    auto it = std::find(electrodes.begin(), electrodes.end(), source);
    if (it == electrodes.end()) {
      std::cout << "source = " << source << " not found in electrodes!"
                << std::endl;
    } else {
      int index = std::distance(electrodes.begin(), it);
      is_source_on_surface[i] = is_electrode_on_surface[index];
    }
  }
  int source_number = sources.size();
  myassert(source_number > 0);

  std::vector<Point_double> tempP(temp_m.begin(), temp_m.end());

  sites.resize(tempP.size());
  for (int i = 0; i < tempP.size(); i++) {
    sites[i](0) = tempP[i](0);
    sites[i](1) = tempP[i](1);
    sites[i](2) = tempP[i](2);
    Point_double site;
    site(0) = tempP[i](0);
    site(1) = tempP[i](1);
    site(2) = tempP[i](2);
    auto it = std::find(electrodes.begin(), electrodes.end(), site);
    if (it == electrodes.end()) {
      std::cout << "source = " << site << " not found in electrodes!"
                << std::endl;
    }
  }

  sites_number = sites.size();
  myassert(sites_number > 0);

  for (int i = 0; i < source_number; i++)
    c_map[tempC[i]] = i;
  for (int i = 0; i < sites_number; i++)
    p_map[tempP[i]] = i;

  myassert(obs_v.size() == n_data);
  myassert(noise_v.size() == n_data);
  d_obs.resize(n_data);
  noise.resize(n_data);
  for (int i = 0; i < n_data; i++) {
    d_obs[i] = obs_v[i];
    noise[i] = noise_v[i];
  }

  record_number_total = temp_m_pair.size();

  msn = measure_sources.size();
  myassert(msn > 0);

  myassert(msn == measure_sites_gather.size());

  measure_sources_locId.resize(msn);

  ms_locations.resize(msn);
  measure_sites_gather_locId.resize(msn);

  int s_len, m_len;
  if (survey_mode == 11) {
    s_len = 1;
    m_len = 1;
  } else if (survey_mode == 22) {
    s_len = 2;
    m_len = 2;
  } else if (survey_mode == 21) {
    s_len = 2;
    m_len = 1;
  } else if (survey_mode == 12) {
    s_len = 1;
    m_len = 2;
  } else {
    std::cout << "fail to get measure_sources_locId, unsopported mode!\n";
    std::abort();
  }

  int count_ms_location = 0;
  for (int i = 0; i < msn; i++) {
    measure_sources_locId[i].resize(s_len);
    myassert(measure_sources[i].size() == s_len);
    for (int j = 0; j < s_len; j++) {
      measure_sources_locId[i][j] = c_map[measure_sources[i][j]];
    }

    int n_site = measure_sites_gather[i].size();
    myassert(n_site > 0);
    measure_sites_gather_locId[i].resize(n_site);
    for (int j = 0; j < n_site; j++) {
      measure_sites_gather_locId[i][j].resize(m_len);
      myassert(measure_sites_gather[i][j].size() == m_len);
      for (int k = 0; k < m_len; k++) {
        measure_sites_gather_locId[i][j][k] =
            p_map[measure_sites_gather[i][j][k]];
      }
    }

    ms_locations[i] = count_ms_location;
    count_ms_location += n_site;
  }

  myassert(count_ms_location == n_data);

  Wd.resize(n_data, n_data);
  Wd.setZero();
  valid_data = n_data;
  for (int i = 0; i < n_data; i++) {

    double temp = noise(i);
    if (temp == 1e12) {
      valid_data--;
    }

    Wd.coeffRef(i, i) = 1.0 / (std::abs(temp) + noise_floor);

  }

  std::cout << "  total data: " << n_data << "  valid_data: " << valid_data
            << std::endl;

  std::ifstream cond_in_stream(model_parameters_file.c_str());
  myassert(cond_in_stream.good());

  int N_inv_domain;
  cond_in_stream >> n_regions >> N_inv_domain;
  marker_vec.resize(n_regions);
  inv_marker.resize(N_inv_domain);
  for (int i = 0; i < n_regions; i++) {
    int marker;
    Vector main_cond(3);
    main_cond = 0.0;
    Vector angle(3);
    angle = 0.0;
    cond_in_stream >> marker >> main_cond[0] >> main_cond[1] >> main_cond[2] >>
        angle[0] >> angle[1] >> angle[2];

    if (angle(0) != 0.0 || angle(1) != 0.0 || angle(2) != 0.0) {
      std::cout << "The rotation angles is not considered in this inversion, "
                   "please check it."
                << std::endl;
      std::abort();
    }
    marker_vec.push_back(marker);
    region2conductivity[marker] = cal_conductivity(main_cond, angle);
  }
  for (int i = 0; i < N_inv_domain; i++) {
    cond_in_stream >> inv_marker[i];
  }
  cond_in_stream.close();

  for (int i = 0; i < N_inv_domain; i++) {
    if (std::find(marker_vec.begin(), marker_vec.end(), inv_marker[i]) ==
        marker_vec.end()) {
      std::cout << "inv_marker is not in forward_marker, check it!"
                << std::endl;
      std::abort();
    }
  }

  std::ifstream msh_stream(mesh_file);
  myassert(msh_stream.good());
  msh_stream.close();

}

DenseMatrix ParaHandler::cal_conductivity(Vector &main_cond, Vector &_angle) {

  Vector angle(3);
  angle = 0.0;
  angle[0] = _angle[0] * (EM::PI / 180);
  angle[1] = _angle[1] * (EM::PI / 180);
  angle[2] = _angle[2] * (EM::PI / 180);
  DenseMatrix sigma(3);
  sigma = 0.0;
  sigma(0, 0) = main_cond[0];
  sigma(1, 1) = main_cond[1];
  sigma(2, 2) = main_cond[2];

  Eigen::Matrix3d rotation_matrix1;
  rotation_matrix1 = Eigen::AngleAxisd(angle[0], Eigen::Vector3d::UnitZ()) *
                     Eigen::AngleAxisd(angle[1], Eigen::Vector3d::UnitX()) *
                     Eigen::AngleAxisd(angle[2], Eigen::Vector3d::UnitZ());

  DenseMatrix R_sigma_RT(3);
  R_sigma_RT = 0.0;
  Eigen::Matrix3d sigma_eigen, R_sigma_RT_eigen;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      sigma_eigen(i, j) = sigma(i, j);
    }
  }
  R_sigma_RT_eigen =
      rotation_matrix1 * sigma_eigen * (rotation_matrix1.transpose());
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      R_sigma_RT(i, j) = R_sigma_RT_eigen(i, j);
    }
  }

  return R_sigma_RT;
}

DenseMatrix ParaHandler::get_elem_conductivity(int marker) {
  myassert(!region2conductivity.empty());
  std::map<int, DenseMatrix>::iterator it = region2conductivity.find(marker);
  if (it == region2conductivity.end()) {
    std::cout << "marker = " << marker << std::endl;
  }
  myassert(it != region2conductivity.end());
  return (*it).second;
}

Vector ParaHandler::get_sources_center() {
  Vector gpoint(3);
  gpoint = 0.0;
  for (int i = 0; i < sources.size(); i++) {
    for (int j = 0; j < 3; j++) {
      gpoint[j] += sources[i](j);
    }
  }
  gpoint /= sources.size();
  return gpoint;
}

void ParaHandler::refine_locally(std::shared_ptr<Mesh> mesh,
                                 std::vector<Point_double> &points,
                                 int refine_level) {
  if (refine_level <= 0)

    return;

  std::set<Point_double> temp_P;

  for (int i = 0; i < points.size(); i++) {
    temp_P.insert(points[i]);
  }

  std::vector<Vertex> p_plus;
  typedef std::set<Point_double>::iterator SIT;
  for (SIT sit = temp_P.begin(); sit != temp_P.end(); sit++) {
    Vertex temp;
    temp(0) = (*sit)(0);
    temp(1) = (*sit)(1);
    temp(2) = (*sit)(2);
    p_plus.push_back(temp);
  }

  DenseMatrix point_mat(3, p_plus.size());
  for (int i = 0; i < p_plus.size(); i++) {
    point_mat(0, i) = p_plus[i](0);
    point_mat(1, i) = p_plus[i](1);
    point_mat(2, i) = p_plus[i](2);
  }

  Array<int> sites_tets;
  Array<IntegrationPoint> ips;
  for (int i_r = 0; i_r < refine_level; i_r++) {
    mesh->FindPoints(point_mat, sites_tets, ips);
    for (int i = 0; i < sites_tets.Size(); i++) {
      if (sites_tets[i] == -1 || sites_tets[i] == -2) {
        std::cout << "The code cannot find the tetrahedra that contain the "
                     "source !!!"
                  << std::endl;
        std::abort();
      }
    }
    mesh->GeneralRefinement(sites_tets);
  }
}

void ParaHandler::refine_locally(std::shared_ptr<Mesh> mesh,
                                 std::vector<Vertex> &points,
                                 int refine_level) {

  int n_points = points.size();
  std::vector<Point_double> points_(n_points);
  for (int i = 0; i < n_points; i++) {
    for (int j = 0; j < 3; j++) {
      points_[i](j) = points[i](j);
    }
  }
  refine_locally(mesh, points_, refine_level);
}

void ParaHandler::find_point_tets(

    std::shared_ptr<ParMesh> pmesh, const std::vector<Vector> &points_,
    Array<int> &find_tets, const std::string find_points_by) {

  std::vector<Vertex> points;
  for (int i = 0; i < points_.size(); i++) {
    for (int j = 0; j < 3; j++) {
      points[i](j) = points_[i](j);
    }
  }

  find_point_tets(pmesh, points, find_tets, find_points_by);
}

void ParaHandler::find_point_tets(

    std::shared_ptr<ParMesh> pmesh, const std::vector<Vertex> &points,
    Array<int> &find_tets, const std::string find_points_by) {

  find_tets.DeleteAll();

  int npts = points.size();
  if (find_points_by == "FindPoints") {

    DenseMatrix point_mat(3, npts);
    for (int j = 0; j < npts; j++) {
      point_mat(0, j) = points[j](0);
      point_mat(1, j) = points[j](1);
      point_mat(2, j) = points[j](2);
    }
    Array<int> temp_tet_id;
    Array<IntegrationPoint> ips;
    pmesh->FindPoints(point_mat, temp_tet_id, ips);

    for (int j = 0; j < temp_tet_id.Size(); j++) {
      int tid = temp_tet_id[j];
      if (tid == -1) {
        std::cout
            << "measurements point not found! please check your parameters";

      } else {

        find_tets.Append(tid);
      }
    }
    myassert(find_tets.Size() == npts);
  } else if (find_points_by == "gslib") {

    throw std::runtime_error("please check gslib installed!");

  } else {
    throw std::runtime_error("please check correct way to find points!");
  }

}

Vector ParaHandler::get_sites_center() {
  Vector gpoint(3);
  gpoint = 0.0;
  for (int i = 0; i < sites.size(); i++) {
    for (int j = 0; j < 3; j++) {
      gpoint[j] += sites[i](j);
    }
  }
  gpoint /= sites.size();
  return gpoint;
}

void ParaHandler::dual_parallel_resource_allocation(int world_rank,
                                                    int world_size) {

  myassert(world_size >= n_group);

  std::vector<int> proc_range(n_group + 1);
  std::vector<int> np_per_group(n_group);
  proc_range[0] = 0;
  for (int i = 0; i < n_group; i++) {

    np_per_group[i] =
        world_size / n_group + ((world_size % n_group) > i ? 1 : 0);
    proc_range[i + 1] = proc_range[i] + np_per_group[i];
  }

  for (int i = 0; i < n_group; i++) {
    if (world_rank >= proc_range[i] && world_rank < proc_range[i + 1]) {
      group_id = i;
      break;
    }
  }

  MPI_Comm_split(MPI_COMM_WORLD, group_id, world_rank, &group_comm);
  MPI_Comm_size(group_comm, &group_size);
  MPI_Comm_rank(group_comm, &group_rank);

  int total_ns = sources.size();
  if (world_rank == 0) {
    std::cout << "    Number of sources: " << total_ns << std::endl;
  }

  if (world_rank == 0) {
    std::cout << "    Number of measurement electrodes: " << sites.size() << std::endl;
    std::cout << "    Number of measurement points: " << record_number_total << std::endl;
  }

  if (n_group > 1) {

    myassert(total_ns >= world_size);
    myassert(group_size == 1);

    myassert(max_amr == 0);
    if_parallel_source = true;
  }

  int begin = 0;
  for (int i = 0; i < group_id; i++) {
    begin += total_ns / n_group + ((total_ns % n_group) > i ? 1 : 0);
  }
  int end =
      begin + total_ns / n_group + ((total_ns % n_group) > group_id ? 1 : 0);

  for (int i = begin; i < end; i++) {

    task_source.push_back(i);
  }
  n_group_s = task_source.size();

  if (world_rank == 0) {
    std::cout << "    Number of measurement sources: " << msn << std::endl;
  }

  int begin_ms = 0;

  for (int i = 0; i < group_id; i++) {
    begin_ms += msn / n_group + ((msn % n_group) > i ? 1 : 0);
  }
  int end_ms = begin_ms + msn / n_group + ((msn % n_group) > group_id ? 1 : 0);

  for (int i = begin_ms; i < end_ms; i++) {

    task_ms.push_back(i);
  }
  n_group_ms = task_ms.size();
}
