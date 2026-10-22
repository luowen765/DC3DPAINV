/*
 * @Description: Implementation of the shared inversion workflow: model transformation,
 *   forward and adjoint calls, line search, mesh adaptivity, and output.
 * @Author: Lewen Qiu, Huang Chen
 * @Date: 2026-10-21
 */

// Copyright (c) 2026, Lewen Qiu and Huang Chen.
// This file is part of the DC3DPAINV program. DC3DPAINV is free
// software with source code available at
// https://github.com/luowen765/DC3DPAINV. You can redistribute it
// or modify it under the terms of the BSD-3 license. See the
// LICENSE file for details.

#include "inv_base.h"

#include "goafem.h"

#include "mfemplus.h"
#include <Eigen/PardisoSupport>

InvBase::InvBase(ParaHandler &inv_para_, int world_rank_, int world_size_,

                 std::string inv_algorithm_)
    : inv_para(inv_para_), world_rank(world_rank_), world_size(world_size_),

      inv_algorithm(inv_algorithm_) {
  this->init();

}

InvBase::~InvBase() { this->clear(); }

void InvBase::init() {

  mesh_inv.reset(new Mesh(inv_para.mesh_file.c_str(), 1, 1));
  if (world_rank == 0) {
    std::cout << "  The number of the cells in the initial mesh is: "
              << mesh_inv->GetNE() << std::endl;
    std::cout << "  The number of the nodes in the initial mesh is: "
              << mesh_inv->GetNV() << std::endl;
    count_forward_time = 0.0;
  }

  if (inv_para.inv_data_type == 1) {

    int group_msn = inv_para.n_group_ms;
    loc_Kcoef.resize(group_msn);

    for (int i = 0; i < group_msn; i++) {
      int gobal_s_id = inv_para.task_ms[i];
      int loc_n_data = inv_para.measure_sites_gather_locId[gobal_s_id].size();
      int global_ms_location = inv_para.ms_locations[gobal_s_id];

      loc_Kcoef[i].resize(loc_n_data);

      for (int j = 0; j < loc_n_data; j++) {

        int col = global_ms_location + j;

        loc_Kcoef[i](j) = inv_para.K_coef(col);
      }
    }
  } else {
    myassert(inv_para.inv_data_type == 0);
  }

  N_d = inv_para.d_obs.size();
  d_best.resize(N_d);

  if (world_rank == 0) {
    d_model.resize(N_d);
    d_model.setZero();
  }
}

void InvBase::clear() {

}

void InvBase::set_model_parameter_convert(double upper, double lower, double cp,
                                          int &kind, REV &m_k_T,
                                          std::shared_ptr<double> m_k,
                                          std::shared_ptr<double> C) {
  if ((upper - lower) < 50)
  {
    for (int i = 0; i < m_k_T.size(); i++) {
      if (kind == 1) {
        m_k_T(i) =
            (1.0 / cp) * log((m_k.get()[i] - lower) / (upper - m_k.get()[i]));
      } else {
        double exp_v = exp(cp * m_k_T(i));
        if (std::isinf(exp_v)) {
          m_k.get()[i] = upper - 1e-8;
        } else if (fabs(exp_v) < 1e-8) {
          m_k.get()[i] = lower + 1e-8;
        } else {
          m_k.get()[i] = (lower + upper * exp_v) / (1 + exp_v);
        }
      }
      C.get()[i] = cp * ((upper - m_k.get()[i]) * (m_k.get()[i] - lower)) /
                   (upper - lower);
    }
  } else {
    for (int i = 0; i < m_k_T.size(); i++) {
      if (kind == 1) {
        m_k_T(i) = (1.0 / cp) * log(((1.0 / m_k.get()[i]) - lower) /
                                    (upper - (1.0 / m_k.get()[i])));
      } else {
        double exp_v = exp(cp * m_k_T(i));
        if (std::isinf(exp_v)) {
          m_k.get()[i] = 1.0 / (upper - 1e-8);
        } else if (fabs(exp_v) < 1e-8) {
          m_k.get()[i] = 1.0 / (lower + 1e-8);
        } else {
          m_k.get()[i] = (1 + exp_v) / (lower + upper * exp_v);
        }
      }
      C.get()[i] =
          -cp * m_k.get()[i] * m_k.get()[i] *
          ((upper - (1.0 / m_k.get()[i])) * ((1.0 / m_k.get()[i]) - lower)) /
          (upper - lower);
    }
  }
}

void InvBase::forward_modeling(int N_iter) {

  double tic, toc;
  if (world_rank == 0) {
    tic = MPI_Wtime();
  }

  if (if_amr_forward == 1) {
    if (pmesh_amr_flag == true)
      pmesh.reset();
    pmesh_amr_flag = false;
  }

  bool if_reset_pmesh = (pmesh == NULL);

  if (inv_para.N_refine_receivers_local_for > 0) {
    if (inv_para.N_refine_electrodes_local_inv > 0) {
      if (world_rank == 0)
        std::cout << "Warning: the inv_mesh and for_mesh are refined again at "
                     "the same "
                     "local receivers. "
                  << std::endl;
    }

    if (if_reset_pmesh) {

      myassert(mesh_inv != NULL);
      std::shared_ptr<Mesh> mesh_inv_temp = std::make_shared<Mesh>(*mesh_inv);

      inv_para.refine_locally(mesh_inv_temp, inv_para.sites,
                              inv_para.N_refine_receivers_local_for);

      pmesh.reset(new ParMesh(inv_para.group_comm, *mesh_inv_temp));

      mesh_inv_temp.reset();
    }
  } else {
    if (if_reset_pmesh) {

      myassert(mesh_inv != NULL);

      pmesh.reset(new ParMesh(inv_para.group_comm, *mesh_inv));
    }
  }

  double tempMemory = GetCurrentMemoryUsage();

  if (inv_para.max_amr == 0 && inv_para.adInv_max_iter == 0) {
    if (world_rank != 0) {

      mesh_inv.reset();
    }
  }

  myassert(pmesh != NULL);

  std::unique_ptr<GOAFEM> goafem(
      new GOAFEM(cond_att, inv_para, pmesh, this->inv_algorithm));

  bool amr_inv = (inv_para.adInv_max_iter > 0) &&
                 (count_inv_iter <= inv_para.adInv_max_iter);

  if (if_amr_forward == 1 && if_forward_lineSearch == 0) {
    if (world_rank == 0)
      std::cout << "Performing adaptive forward...\n";
    pmesh_amr_flag = true;
    for (int N_amr = 0; N_amr <= inv_para.max_amr; N_amr++) {

      goafem->reconstruct_FEM_terms();

      if (N_amr < inv_para.max_amr) {
        goafem->set_dual_adjoint_problem(true, false);
        goafem->if_refine_mesh = true;
      } else {
        goafem->set_dual_adjoint_problem(false, true);
        goafem->if_refine_mesh = false;
      }

      this->reset_inv_for_map();

      goafem->one_forward_iteration(d_model, JT_V, JT, loc_Kcoef, loc_d_obs,
                                    loc_Wd, inv_forward_map, Trans_mat,
                                    N_inv_cell);

      goafem->update();
    }
  } else {

    goafem->reconstruct_FEM_terms();
    goafem->set_dual_adjoint_problem(false, true);
    goafem->if_refine_mesh = false;

    if (if_reset_pmesh)
      this->reset_inv_for_map();

    goafem->one_forward_iteration(d_model, JT_V, JT, loc_Kcoef, loc_d_obs,
                                  loc_Wd, inv_forward_map, Trans_mat,
                                  N_inv_cell);
  }

  N_fo_ver_k = 0;
  int local_N_fo_ver_k = goafem->pfes->GetTrueVSize();
  MPI_Allreduce(&local_N_fo_ver_k, &N_fo_ver_k, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
  myassert(N_fo_ver_k > 0);

  if (world_rank == 0) {
    toc = MPI_Wtime();
    count_forward_times++;
    std::cout << "  Forward modeling iteration # and time consumed: "
              << count_forward_times << " : " << toc - tic << " seconds."
              << std::endl;
    count_forward_time = count_forward_time + (toc - tic);
  }
}

void InvBase::save_predicted_data(std::string &iter_name,
                                  std::string &survery_mode_file) {
  std::ostringstream sol_name;
  sol_name << iter_name << "/"
           << "Modeling_data.txt";
  std::ofstream output(sol_name.str().c_str());

  int N_electrodes;
  std::map<Point_double, int> points_map;

  std::ifstream mode_stream(survery_mode_file.c_str());
  myassert(mode_stream.good());

  mode_stream >> N_electrodes;
  output << N_electrodes << std::endl;
  myassert(N_electrodes > 0);

  for (int i = 0; i < N_electrodes; i++) {
    Point_double electrode;
    int is_on_earth_surface;
    mode_stream >> electrode(0) >> electrode(1) >> electrode(2) >>
        is_on_earth_surface;
    output << electrode(0) << " " << electrode(1) << " " << electrode(2) << " "
           << is_on_earth_surface << std::endl;
    points_map[electrode] = i;
  }

  mode_stream.close();

  int valid_data = 0;
  for (int i = 0; i < inv_para.msn; i++) {
    int n_site = inv_para.measure_sites_gather_locId[i].size();
    for (int j = 0; j < n_site; j++) {
      int i_data = i * n_site + j;

      valid_data++;
    }
  }
  output << valid_data << "\t" << inv_para.survey_mode << "\n";

  for (int i = 0; i < inv_para.msn; i++) {
    int n_site = inv_para.measure_sites_gather_locId[i].size();
    for (int j = 0; j < n_site; j++) {
      int i_data = inv_para.ms_locations[i] + j;

      for (int g = 0; g < inv_para.measure_sources_locId[i].size(); g++) {
        int source_id = inv_para.measure_sources_locId[i][g];
        Point_double s_p;
        for (int ii = 0; ii < 3; ii++) {
          s_p(ii) = inv_para.sources[source_id](ii);
        }
        int point_id = points_map[s_p];
        output << point_id + 1 << "\t";
      }
      for (int h = 0; h < inv_para.measure_sites_gather_locId[i][j].size();
           h++) {
        int site_id = inv_para.measure_sites_gather_locId[i][j][h];

        Point_double m_p;
        for (int ii = 0; ii < 3; ii++) {
          m_p(ii) = inv_para.sites[site_id](ii);
        }
        int point_id = points_map[m_p];
        output << point_id + 1 << "\t";
      }

      output << d_model(i_data) << "\n";
    }
  }

  output.close();
}

void InvBase::save_cond_mesh(std::string &iter_name) {

  myassert(mesh_inv != NULL);

  myassert(cond_att.size() == mesh_inv->GetNE());
  Meshplus meshplus = Meshplus(mesh_inv.get());
  if (N_m == N_inv_cell) {
    Vector cond_xx(cond_att.size());
    cond_xx = 0.0;
    for (int i = 0; i < cond_att.size(); i++) {
      cond_xx(i) = cond_att[i](0, 0);
    }
    std::ostringstream vtk_name_inv_1;
    vtk_name_inv_1 << iter_name << "/"
                   << "cond_inv_mesh"
                   << ".vtk";
    std::ofstream vtk_inv_1(vtk_name_inv_1.str().c_str());
    vtk_inv_1.precision(8);

    meshplus.PrintVTK_eta(vtk_inv_1, cond_xx, 2);

  } else if (N_m == 2 * N_inv_cell) {
    Vector cond_xx(cond_att.size());
    Vector cond_zz(cond_att.size());
    for (int i = 0; i < cond_att.size(); i++) {
      cond_xx(i) = cond_att[i](0, 0);
      cond_zz(i) = cond_att[i](2, 2);
    }

    std::ostringstream vtk_name_inv_1, vtk_name_inv_2;
    vtk_name_inv_1 << iter_name << "/"
                   << "cond_h_inv_mesh"
                   << ".vtk";
    vtk_name_inv_2 << iter_name << "/"
                   << "cond_v_inv_mesh"
                   << ".vtk";
    std::ofstream vtk_inv_1(vtk_name_inv_1.str().c_str());
    std::ofstream vtk_inv_2(vtk_name_inv_2.str().c_str());
    vtk_inv_1.precision(8);
    vtk_inv_2.precision(8);
    meshplus.PrintVTK_eta(vtk_inv_1, cond_xx, 2);
    meshplus.PrintVTK_eta(vtk_inv_2, cond_zz, 2);

  } else {
    Vector cond_xx(cond_att.size());
    Vector cond_yy(cond_att.size());
    Vector cond_zz(cond_att.size());
    for (int i = 0; i < cond_att.size(); i++) {
      cond_xx(i) = cond_att[i](0, 0);
      cond_yy(i) = cond_att[i](1, 1);
      cond_zz(i) = cond_att[i](2, 2);
    }

    std::ostringstream vtk_name_inv_1, vtk_name_inv_2, vtk_name_inv_3;
    vtk_name_inv_1 << iter_name << "/"
                   << "cond_xx_inv_mesh"
                   << ".vtk";
    vtk_name_inv_2 << iter_name << "/"
                   << "cond_yy_inv_mesh"
                   << ".vtk";
    vtk_name_inv_3 << iter_name << "/"
                   << "cond_zz_inv_mesh"
                   << ".vtk";
    std::ofstream vtk_inv_1(vtk_name_inv_1.str().c_str());
    std::ofstream vtk_inv_2(vtk_name_inv_2.str().c_str());
    std::ofstream vtk_inv_3(vtk_name_inv_3.str().c_str());
    vtk_inv_1.precision(8);
    vtk_inv_2.precision(8);
    vtk_inv_3.precision(8);
    meshplus.PrintVTK_eta(vtk_inv_1, cond_xx, 2);
    meshplus.PrintVTK_eta(vtk_inv_2, cond_yy, 2);
    meshplus.PrintVTK_eta(vtk_inv_3, cond_zz, 2);

  }
}

void InvBase::StrongWolfeConditionsLineSearch(int N_iter) {

  int stop_loop = 0;

  if_forward_lineSearch = 0;

  int Trans_type = 2, expansion = 2;

  double c_1, c_2, alpha_hi, alpha_lo, alpha, alpha_best, phi_zero, phi, phi_hi,
      phi_lo, phi_best, phi_grad_zero, phi_grad, phi_grad_hi, phi_grad_lo;

  if (world_rank == 0) {

    c_1 = 1e-4, c_2 = 0.9, alpha = 1, alpha_best = alpha;
    phi_zero = (inv_para.Wd * (d_model - inv_para.d_obs)).squaredNorm() +
               lamda[N_iter - 2] * (m_k - m_ref).dot(WmTWm * (m_k - m_ref));
    phi_grad_zero =
        2 * (JT_V + lamda[N_iter - 2] * WmTWm * (m_k - m_ref)).dot(update_dire);
    alpha_lo = 0, phi_lo = phi_zero, phi_grad_lo = phi_grad_zero;
  }

  for (int i = 0; i < inv_para.N_line_search; i++) {

    if (world_rank == 0) {
      m_search = m_k + alpha * update_dire;

      set_model_parameter_convert(inv_para.max_cond, inv_para.min_cond,
                                  inv_para.cp, Trans_type, m_search, cond_k,
                                  Trans_mat);
    }

    this->transfer_Cond_Trans();
    this->modify_cond_att();
    forward_modeling(N_iter);

    if (world_rank == 0) {
      phi = (inv_para.Wd * (d_model - inv_para.d_obs)).squaredNorm() +
            lamda[N_iter - 2] *
                (m_search - m_ref).dot(WmTWm * (m_search - m_ref));
      phi_grad = 2 * (JT_V + lamda[N_iter - 2] * WmTWm * (m_search - m_ref))
                         .dot(update_dire);

      if (i == 0 || (i > 0 && phi < phi_best)) {
        m_best = m_search, d_best = d_model, JTV_best = JT_V, phi_best = phi,
        alpha_best = alpha;
      }

      if ((phi - phi_zero) > c_1 * alpha * phi_grad_zero ||
          (0 < i && phi_lo <= phi)) {
        alpha_hi = alpha, phi_hi = phi, phi_grad_hi = phi_grad;
        stop_loop = 1;
        std::cout
            << "    The condition of sufficient descent is not satisfied, "
               "perform zoom search..."
            << std::endl;
      } else if (std::abs(phi_grad) <= -c_2 * phi_grad_zero) {
        stop_loop = 2;
        std::cout << "    The linear search is successful, the step value is: "
                  << alpha << std::endl;
      } else {
        alpha_hi = alpha_lo, alpha_lo = alpha;
        phi_hi = phi_lo, phi_lo = phi;
        phi_grad_hi = phi_grad_lo, phi_grad_lo = phi_grad;
        if (phi_grad >= 0) {
          stop_loop = 1;
          std::cout
              << "    The gradient of the objective function is larger than "
                 "zero, perform zoom search..."
              << std::endl;
        } else {
          std::cout << "    The curvature condition is not satisfied, continue "
                       "searching..."
                    << std::endl;
        }
      }
      for (int p = 1; p < world_size; p++) {
        MPI_Send(&stop_loop, 1, MPI_INT, p, 200, MPI_COMM_WORLD);
      }
    } else {
      MPI_Recv(&stop_loop, 1, MPI_INT, 0, 200, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }

    if (stop_loop == 2) {

      if (world_rank == 0) {
        m_k = m_search;
      }

      return;
    } else if (stop_loop == 1) {
      if_forward_lineSearch = 1;
      break;
    } else {
      if_forward_lineSearch = 1;

      if (world_rank == 0) {
        alpha = alpha * expansion;
      }
    }
  }

  if (stop_loop == 1) {

    for (int j = 0; j < inv_para.N_line_search; j++) {

      if (world_rank == 0) {
        alpha = (phi_hi - phi_lo) * alpha_lo -
                (alpha_hi * alpha_hi - alpha_lo * alpha_lo) * phi_grad_lo / 2;
        alpha /= (phi_hi - phi_lo) - (alpha_hi - alpha_lo) * phi_grad_lo;
        if (alpha <= std::min(alpha_lo, alpha_hi) ||
            alpha >= std::max(alpha_lo, alpha_hi)) {
          alpha = (alpha_lo + alpha_hi) / 2;
        }
        m_search = m_k + alpha * update_dire;

        set_model_parameter_convert(inv_para.max_cond, inv_para.min_cond,
                                    inv_para.cp, Trans_type, m_search, cond_k,
                                    Trans_mat);
      }

      this->transfer_Cond_Trans();
      this->modify_cond_att();
      forward_modeling(N_iter);

      if (world_rank == 0) {
        phi = (inv_para.Wd * (d_model - inv_para.d_obs)).squaredNorm() +
              lamda[N_iter - 2] *
                  (m_search - m_ref).dot(WmTWm * (m_search - m_ref));
        phi_grad = 2 * (JT_V + lamda[N_iter - 2] * WmTWm * (m_search - m_ref))
                           .dot(update_dire);

        if (phi < phi_best) {
          m_best = m_search, d_best = d_model, JTV_best = JT_V, phi_best = phi,
          alpha_best = alpha;
        }

        if ((phi - phi_zero) > c_1 * alpha * phi_grad_zero || phi_lo <= phi) {
          std::cout
              << "    The condition of sufficient descent is not satisfied, "
                 "continue searching..."
              << std::endl;
          alpha_hi = alpha, phi_hi = phi, phi_grad_hi = phi_grad;
        } else {
          if (std::abs(phi_grad) <= -c_2 * phi_grad_zero) {
            std::cout
                << "    The zoom search is successful, the step value is: "
                << alpha << std::endl;
            stop_loop = 2;
          } else {
            std::cout
                << "    The curvature condition is not satisfied, continue "
                   "searching..."
                << std::endl;
            if (phi_grad * (alpha_hi - alpha_lo) >= 0) {
              alpha_hi = alpha_lo;
              phi_hi = phi_lo;
              phi_grad_hi = phi_grad_lo;
            }
            alpha_lo = alpha;
            phi_lo = phi;
            phi_grad_lo = phi_grad;
          }
        }
        for (int p = 1; p < world_size; p++) {
          MPI_Send(&stop_loop, 1, MPI_INT, p, 200, MPI_COMM_WORLD);
        }
      } else {
        MPI_Recv(&stop_loop, 1, MPI_INT, 0, 200, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
      }

      if (stop_loop == 2) {

        if (world_rank == 0) {
          m_k = m_search;
        }

        return;
      }
    }
  }

  if (world_rank == 0) {
    std::cout << "    The line search did not converge, the best step is: "
              << alpha_best << std::endl;
    d_model = d_best;
    m_k = m_best;
    JT_V = JTV_best;
  }

  return;
}

void InvBase::update_common_inv_para(int N_iter) {

  myassert(world_rank == 0);

  N_inv_cells.push_back(N_inv_cell);

  int k = N_iter;
  double data_fit = 0.0;

  data_fit = sqrt((inv_para.Wd * (d_model - inv_para.d_obs)).squaredNorm() /
                  inv_para.valid_data);

  misfit.push_back(data_fit);
  std::cout << "  Error of data fitting: " << data_fit << std::endl;

  N_fo_ver.push_back(N_fo_ver_k);

  if ((inv_para.adInv_max_iter == 0) ||
      (count_inv_iter >= inv_para.adInv_max_iter)) {
    if_amr_inv = 0;
  } else {
    if (k == 1) {
      initial_misfit = data_fit;
      max_misfit_inv = data_fit;
      myassert(if_amr_inv == 0);
      inv_mesh_keep = inv_para.invMesh_keep;

      judge_inv_amr_min = 1;
    }

    else {
      double change_max_misfit_inv =
          (max_misfit_inv - data_fit) / initial_misfit;

      if (if_amr_inv == 1) {

        max_misfit_inv = data_fit;
      } else {
        myassert(if_amr_inv == 0);
        if (max_misfit_inv - data_fit < 0) {

          max_misfit_inv = data_fit;
        }
      }

      if (judge_inv_amr_min > 0 && judge_inv_amr_min < inv_mesh_keep) {

        if_amr_inv = 0;
        judge_inv_amr_min++;
      } else if (judge_inv_amr_min == inv_mesh_keep) {
        judge_inv_amr_min = 0;
      }

      if (judge_inv_amr_min == 0) {
        myassert(if_amr_inv == 0);

        bool if1 = (change_max_misfit_inv > inv_para.relative_misfit_inv_up);
        bool if2 = (change_max_misfit_inv < inv_para.relative_misfit_inv_down);
        if (if1 || if2) {
          if_amr_inv = 1;
          if (if1)
            std::cout << "    Adaptive inversion: misfit decreasing too fast" << std::endl;
          if (if2)
            std::cout << "    Adaptive inversion: misfit decreasing too slow" << std::endl;
          std::cout << "    change_max_misfit_inv_ratio: "
                    << change_max_misfit_inv
                    << " value: " << change_max_misfit_inv * initial_misfit
                    << std::endl;

          count_amr_inv_times++;

          std::cout << "    count_amr_inv_times: " << count_amr_inv_times
                    << std::endl;

          judge_inv_amr_min = 1;

        }

      }

    }
  }

  if (if_amr_inv == 1) {
    count_inv_iter++;
  }

  if_inv_amr_s.push_back(if_amr_inv);

  if (inv_para.max_amr == 0) {
    if_amr_forward = 0;
  }

  else {

    if (k == 1) {
      initial_misfit = data_fit;
      max_misfit_for = data_fit;
      myassert(if_amr_forward == 0);

    } else {
      double change_max_misfit_for =
          (max_misfit_for - data_fit) / initial_misfit;

      double change_last_misfit = (last_misfit - data_fit) / initial_misfit;

      if (if_amr_forward == 0) {
        if ((max_misfit_for - data_fit) > 0) {

          bool if1 = change_max_misfit_for > inv_para.relative_misfit_for_up;
          bool if2 = change_max_misfit_for < inv_para.relative_misfit_for_down;
          if (if1 || if2) {
            if_amr_forward = 1;
            if (if1)
              std::cout << "    Adaptive forward modeling: misfit decreasing too fast" << std::endl;
            if (if2)
              std::cout << "    Adaptive forward modeling: misfit decreasing too slow" << std::endl;
            count_amr_for_times++;
            std::cout << "    count_amr_for_times: " << count_amr_for_times
                      << std::endl;
          }

          std::cout << "    change_max_misfit_for_ratio: "
                    << change_max_misfit_for
                    << " value: " << change_max_misfit_for * initial_misfit
                    << std::endl;

        } else {

          max_misfit_for = data_fit;
        }
      } else {
        myassert(if_amr_forward == 1);

        max_misfit_for = data_fit;

        if_amr_forward = 0;

      }

    }
  }

  if_amr_s.push_back(if_amr_forward);

  last_misfit = data_fit;

  if (k == 1) {
    lamda_k = inv_para.init_lambda;
    judge_lambda = 1;
  } else {

    if (std::abs(lamda_k - inv_para.lambda_floor) > 0.0) {
      if (judge_lambda > 0 && judge_lambda < inv_para.lambda_keep) {

        judge_lambda += 1;
      } else if (judge_lambda == inv_para.lambda_keep) {
        judge_lambda = 0;
      }

      if (judge_lambda == 0) {

        double decrease_1 = ((misfit[k - 2] - data_fit) / misfit[k - 2] +
                             (misfit[k - 3] - misfit[k - 2]) / misfit[k - 3]) /
                            2.0;
        double decrease_2 = (misfit[k - 2] - data_fit) / misfit[k - 2];
        if (decrease_1 < inv_para.lambda_atte_thre &&
            decrease_2 < inv_para.lambda_atte_thre) {
          lamda_k = lamda_k * inv_para.atte_lambda;
          judge_lambda = 1;
        }
      }
    }
  }

  lamda.push_back(lamda_k);
  std::cout << "  The value of the regularization parameter: " << lamda_k
            << std::endl;

  double roughness_k = (m_k - m_ref).transpose() * WmTWm * (m_k - m_ref);
  double obj_func_k = (inv_para.Wd * (d_model - inv_para.d_obs)).squaredNorm() +
                      lamda_k * roughness_k;
  roughness.push_back(roughness_k);
  objective_func.push_back(obj_func_k);

}

void InvBase::output_inversion_parameters(std::string inv_reslut_folder) {
  std::ostringstream sol_name;
  sol_name << inv_reslut_folder << "/"
           << "Output_Inversion_parameter";
  std::ofstream output(sol_name.str().c_str());
  output << "iter\t"
         << "lamda\t"
         << "misfit\t"
         << "roughness\t"
         << "ob_func\t"
         << "if_amr_f\t"
         << "if_amr_i\t"
         << "N_f_V\t"
         << "N_i_C\n";
  output.precision(8);

  for (int i = 0; i < misfit.size(); i++) {
    output << i + 1 << "\t" << lamda[i] << "\t" << misfit[i] << "\t"
           << roughness[i] << "\t" << objective_func[i] << "\t" << if_amr_s[i]
           << "\t";
    output << if_inv_amr_s[i] << "\t";
    output << N_fo_ver[i] << "\t";
    output << N_inv_cells[i] << "\n";
  }
}

void InvBase::resetInvParas(int N_iter) {
  if (world_rank == 0)
    std::cout << "Initializing the inversion parameters:" << std::endl;

  if (N_iter == 1) {

    int n_e = inv_para.N_refine_electrodes_local_inv;
    inv_para.refine_locally(mesh_inv, inv_para.electrodes, n_e);
    if (world_rank == 0) {
      if (n_e > 0) {
        std::cout
            << "The number of refining electrodes locally for the inv mesh: "
            << n_e << std::endl;

        std::cout
            << "  The number of the cells in the locally refined inv mesh is: "
            << mesh_inv->GetNE() << std::endl;
        std::cout << "  The number of the nodes in the locally refined inv "
                     "mesh is: "
                  << mesh_inv->GetNV() << std::endl;
      }
    }

    cond_att.resize(mesh_inv->GetNE());
    inv_att_init.resize(mesh_inv->GetNE());

    for (int i = 0; i < mesh_inv->GetNE(); i++) {
      int cell_att = mesh_inv->GetAttribute(i);
      cond_att[i] = inv_para.get_elem_conductivity(cell_att);
      inv_att_init[i] = cell_att;

    }

  } else {
    myassert(cond_att.size() > 0 && inv_att_init.size() > 0);
    std::vector<DenseMatrix> last_cond_att = cond_att;
    std::vector<int> last_inv_att_init = inv_att_init;
    cond_att.resize(mesh_inv->GetNE());
    inv_att_init.resize(mesh_inv->GetNE());
    for (int i = 0; i < mesh_inv->GetNE(); i++) {
      cond_att[i] = last_cond_att[mesh_inv->GetAttribute(i) - 1];
      inv_att_init[i] = last_inv_att_init[mesh_inv->GetAttribute(i) - 1];

    }

    last_forward_inv.clear();
    last_forward_inv.resize(forward_inv.size());
    last_forward_inv = forward_inv;
  }

  forward_inv.resize(mesh_inv->GetNE());
  inv_forward.clear();
  N_inv_cell = 0;
  for (unsigned int i = 0; i < mesh_inv->GetNE(); i++) {

    if (std::find(inv_para.inv_marker.begin(), inv_para.inv_marker.end(),
                  inv_att_init[i]) != inv_para.inv_marker.end()) {

      inv_forward.push_back(i);

      forward_inv[i] = N_inv_cell;
      N_inv_cell = N_inv_cell + 1;
    } else {
      forward_inv[i] = -19210723;
    }
  }

  if (inv_para.if_anisotropy == 0) {
    N_m = N_inv_cell;
  } else if (inv_para.if_anisotropy == 1) {
    N_m = N_inv_cell * 2;
  } else {
    N_m = N_inv_cell * 3;
  }

  if (world_rank == 0) {
    std::cout << "  The number of inversion unknowns is: " << N_m << std::endl;
    std::cout << "  The number of inversion elements is: " << N_inv_cell
              << std::endl;

  }

  if (world_rank == 0) {
    JT_V.resize(N_m);

    g_k.resize(N_m), update_dire.resize(N_m);
    m_search.resize(N_m), m_best.resize(N_m), JTV_best.resize(N_m);
  }

  cond_k.reset(new double[N_m]);
  Trans_mat.reset(new double[N_m]);

  if (N_iter == 1) {
    if (world_rank == 0) {
      m_k.resize(N_m);
      m_ref.resize(N_m);

      std::vector<DenseMatrix> inv_cell_init_cond;
      for (unsigned int i = 0; i < mesh_inv->GetNE(); i++) {
        if (std::find(inv_para.inv_marker.begin(), inv_para.inv_marker.end(),
                      inv_att_init[i]) != inv_para.inv_marker.end()) {

          inv_cell_init_cond.push_back(cond_att[i]);
        }
      }

      std::shared_ptr<double> cond_ref(new double[N_m]);

      for (int i = 0; i < N_inv_cell; i++) {
        cond_k.get()[i] = inv_cell_init_cond[i](0, 0);
        cond_ref.get()[i] = inv_cell_init_cond[i](0, 0);
        if (inv_para.if_anisotropy == 1) {
          cond_k.get()[i + N_inv_cell] = inv_cell_init_cond[i](2, 2);
          cond_ref.get()[i + N_inv_cell] = inv_cell_init_cond[i](2, 2);
        }
        if (inv_para.if_anisotropy == 2) {
          cond_k.get()[i + N_inv_cell] = inv_cell_init_cond[i](1, 1);
          cond_ref.get()[i + N_inv_cell] = inv_cell_init_cond[i](1, 1);
          cond_k.get()[i + 2 * N_inv_cell] = inv_cell_init_cond[i](2, 2);
          cond_ref.get()[i + 2 * N_inv_cell] = inv_cell_init_cond[i](2, 2);
        }
      }

      int Trans_type = 1;

      set_model_parameter_convert(inv_para.max_cond, inv_para.min_cond,
                                  inv_para.cp, Trans_type, m_ref, cond_ref,
                                  Trans_mat);

      set_model_parameter_convert(inv_para.max_cond, inv_para.min_cond,
                                  inv_para.cp, Trans_type, m_k, cond_k,
                                  Trans_mat);
    }

  } else {
    if (world_rank == 0) {

      this->reset_invpara_REV(m_k);

      this->reset_invpara_REV(m_ref);

      int Trans_type = 2;
      set_model_parameter_convert(inv_para.max_cond, inv_para.min_cond,
                                  inv_para.cp, Trans_type, m_k, cond_k,
                                  Trans_mat);
    }
  }

  this->transfer_Cond_Trans();

  if (N_iter > 1) {

    this->modify_cond_att();
  }

  this->compute_WmTWm();
  pmesh.reset();
}

void InvBase::refine_inv_mesh(std::string inv_reslut_folder) {

  if (world_rank == 0) {
    std::cout << "Refining inversion mesh...\n";
  }

  if (world_rank != 0) {
    JT_V.resize(N_m);
  }
  this->transfer_REV(JT_V);

  Vector err_inv_mesh(mesh_inv->GetNE());
  this->compute_err_JTV(err_inv_mesh);

  double threshold = vector_threshold(err_inv_mesh, inv_para.inv_amr_ratio);
  if (world_rank == 0) {
    this->save_mesh(inv_reslut_folder + "/" + "Indicator_inv", err_inv_mesh);
  }
  mesh_inv->RefineByError(err_inv_mesh, threshold);

  if (world_rank != 0) {
    m_k.resize(0);
    JT_V.resize(0);
  }
}

void InvBase::compute_err_JTV(Vector &err_model_JTV) {
  err_model_JTV.SetSize(mesh_inv->GetNE());
  err_model_JTV = 0.00001;

  myassert(JT_V.size() == N_m);
  for (int i = 0; i < inv_forward.size(); i++) {
    double tet_V = mesh_inv->GetElementVolume(inv_forward[i]);
    Vector tet_center_point(3);
    mesh_inv->GetElementCenter(inv_forward[i], tet_center_point);

    if (inv_para.if_anisotropy == 0) {
      err_model_JTV[inv_forward[i]] = std::abs(JT_V[i]);

    } else if (inv_para.if_anisotropy == 1) {
      err_model_JTV[inv_forward[i]] =
          0.5 * (std::abs(JT_V[i]) + std::abs(JT_V[i + N_inv_cell]));
    } else {
      err_model_JTV[inv_forward[i]] =
          (std::abs(JT_V[i]) + std::abs(JT_V[i + N_inv_cell]) +
           std::abs(JT_V[i + 2 * N_inv_cell])) /
          3.0;
    }
  }
}

void InvBase::reset_invpara_REV(REV &para) {
  REV last_para = para;

  int n_comp = 1;
  if (inv_para.if_anisotropy == 1)
    n_comp = 2;
  else if (inv_para.if_anisotropy == 2)
    n_comp = 3;
  int last_N_inv_cell = last_para.size() / n_comp;
  para.resize(N_m);

  myassert(inv_forward.size() == N_inv_cell);

  if (inv_para.if_anisotropy == 1)
    myassert(inv_forward.size() * 2 == N_m);
  if (inv_para.if_anisotropy == 2)
    myassert(inv_forward.size() * 3 == N_m);

  for (int i = 0; i < N_inv_cell; i++) {
    int last_forward_id = mesh_inv->GetAttribute(inv_forward[i]) - 1;
    int last_inv_id = last_forward_inv[last_forward_id];
    para[i] = last_para[last_inv_id];

    if (inv_para.if_anisotropy == 1 || inv_para.if_anisotropy == 2) {
      para[i + N_inv_cell] = last_para[last_inv_id + last_N_inv_cell];
    }

    if (inv_para.if_anisotropy == 2) {

      para[i + 2 * N_inv_cell] = last_para[last_inv_id + 2 * last_N_inv_cell];
    }

  }
}

void InvBase::reset_inv_for_map() {
  inv_forward_map.clear();

  typedef std::map<int, std::vector<int>>::iterator IT;
  for (int i = 0; i < pmesh->GetNE(); i++) {

    int init_forwardId = pmesh->GetAttribute(i) - 1;
    int inversionID = forward_inv[init_forwardId];

    if (inversionID != -19210723) {

      IT it = inv_forward_map.find(inversionID);
      if (it == inv_forward_map.end()) {

        inv_forward_map[inversionID].push_back(i);
      } else {

        inv_forward_map[(*it).first].push_back(i);
      }
    }
  }
}

void InvBase::transfer_REV(REV &vec) {

  std::unique_ptr<double> temp(new double[N_m]);
  if (world_rank == 0) {
    myassert(vec.size() == N_m);
    for (int i = 0; i < N_m; i++) {
      temp.get()[i] = vec[i];
    }
    for (int p = 1; p < world_size; p++) {
      MPI_Send(temp.get(), N_m, MPI_DOUBLE, p, 108, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(temp.get(), N_m, MPI_DOUBLE, 0, 108, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);

    myassert(vec.size() == N_m);
    for (int i = 0; i < N_m; i++) {
      vec[i] = temp.get()[i];
    }
  }
}

void InvBase::save_JTV_mesh(std::string &iter_name) {

  myassert(mesh_inv != NULL);
  Meshplus meshplus = Meshplus(mesh_inv.get());

  myassert(JT_V.size() == N_m);
  myassert(JT_V.size() > 0);

  if (N_m == N_inv_cell) {
    Vector JTV_xx(cond_att.size());
    JTV_xx = 0.0;
    for (int i = 0; i < N_inv_cell; i++) {
      JTV_xx(inv_forward[i]) = JT_V(i);
    }
    std::ostringstream vtk_name_inv_2;
    vtk_name_inv_2 << iter_name << "/"
                   << "JTV_inv_mesh"
                   << ".vtk";
    std::ofstream vtk_inv_2(vtk_name_inv_2.str().c_str());
    vtk_inv_2.precision(8);
    meshplus.PrintVTK_eta(vtk_inv_2, JTV_xx, 2);

  } else if (N_m == 2 * N_inv_cell) {
    Vector JTV_xx(cond_att.size());
    Vector JTV_zz(cond_att.size());
    JTV_xx = 0.0, JTV_zz = 0.0;
    for (int i = 0; i < N_inv_cell; i++) {
      JTV_xx(inv_forward[i]) = JT_V(i);
      JTV_zz(inv_forward[i]) = JT_V(i + N_inv_cell);
    }
    std::ostringstream vtk_name_inv_3, vtk_name_inv_4;
    vtk_name_inv_3 << iter_name << "/"
                   << "JTV_h_inv_mesh"
                   << ".vtk";
    vtk_name_inv_4 << iter_name << "/"
                   << "JTV_v_inv_mesh"
                   << ".vtk";
    std::ofstream vtk_inv_3(vtk_name_inv_3.str().c_str());
    std::ofstream vtk_inv_4(vtk_name_inv_4.str().c_str());
    vtk_inv_3.precision(8);
    vtk_inv_4.precision(8);
    meshplus.PrintVTK_eta(vtk_inv_3, JTV_xx, 2);
    meshplus.PrintVTK_eta(vtk_inv_4, JTV_zz, 2);

  } else {
    Vector JTV_xx(cond_att.size());
    Vector JTV_yy(cond_att.size());
    Vector JTV_zz(cond_att.size());
    JTV_xx = 0.0, JTV_yy = 0.0, JTV_zz = 0.0;
    for (int i = 0; i < N_inv_cell; i++) {
      JTV_xx(inv_forward[i]) = JT_V(i);
      JTV_yy(inv_forward[i]) = JT_V(i + N_inv_cell);
      JTV_zz(inv_forward[i]) = JT_V(i + 2 * N_inv_cell);
    }
    std::ostringstream vtk_name_inv_4, vtk_name_inv_5, vtk_name_inv_6;

    vtk_name_inv_4 << iter_name << "/"
                   << "JTV_xx_inv_mesh"
                   << ".vtk";
    vtk_name_inv_5 << iter_name << "/"
                   << "JTV_yy_inv_mesh"
                   << ".vtk";
    vtk_name_inv_6 << iter_name << "/"
                   << "JTV_zz_inv_mesh"
                   << ".vtk";
    std::ofstream vtk_inv_4(vtk_name_inv_4.str().c_str());
    std::ofstream vtk_inv_5(vtk_name_inv_5.str().c_str());
    std::ofstream vtk_inv_6(vtk_name_inv_6.str().c_str());
    vtk_inv_4.precision(8);
    vtk_inv_5.precision(8);
    vtk_inv_6.precision(8);
    meshplus.PrintVTK_eta(vtk_inv_4, JTV_xx, 2);
    meshplus.PrintVTK_eta(vtk_inv_5, JTV_yy, 2);
    meshplus.PrintVTK_eta(vtk_inv_6, JTV_zz, 2);

  }
}

void InvBase::save_mesh(std::string inv_reslut_folder,
                        Vector &current_inv_mesh) {

  myassert(mesh_inv != NULL);
  Meshplus meshplus = Meshplus(mesh_inv.get());

  std::ostringstream vtk_name;
  vtk_name << inv_reslut_folder << "_" << std::to_string(count_inv_iter)
           << ".vtk";
  std::ofstream vtk_ofs(vtk_name.str().c_str());
  vtk_ofs.precision(8);
  myassert(current_inv_mesh.Size() > 0);
  meshplus.PrintVTK_eta(vtk_ofs, current_inv_mesh);
}

void InvBase::transfer_Cond_Trans() {

  if (world_rank == 0) {
    for (int p = 1; p < world_size; p++) {
      MPI_Send(cond_k.get(), N_m, MPI_DOUBLE, p, 110, MPI_COMM_WORLD);
      MPI_Send(Trans_mat.get(), N_m, MPI_DOUBLE, p, 111, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(cond_k.get(), N_m, MPI_DOUBLE, 0, 110, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    MPI_Recv(Trans_mat.get(), N_m, MPI_DOUBLE, 0, 111, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }
}

void InvBase::modify_cond_att() {
  if (inv_para.if_anisotropy == 0) {
    for (int i = 0; i < N_inv_cell; i++) {
      cond_att[inv_forward[i]](0, 0) = cond_k.get()[i];
      cond_att[inv_forward[i]](1, 1) = cond_k.get()[i];
      cond_att[inv_forward[i]](2, 2) = cond_k.get()[i];
    }
  } else if (inv_para.if_anisotropy == 1) {
    for (int i = 0; i < N_inv_cell; i++) {
      cond_att[inv_forward[i]](0, 0) = cond_k.get()[i];
      cond_att[inv_forward[i]](1, 1) = cond_k.get()[i];
      cond_att[inv_forward[i]](2, 2) = cond_k.get()[i + N_inv_cell];
    }
  } else {
    for (int i = 0; i < N_inv_cell; i++) {
      cond_att[inv_forward[i]](0, 0) = cond_k.get()[i];
      cond_att[inv_forward[i]](1, 1) = cond_k.get()[i + N_inv_cell];
      cond_att[inv_forward[i]](2, 2) = cond_k.get()[i + 2 * N_inv_cell];
    }
  }
}

void InvBase::compute_WmTWm() {

  if (world_rank == 0) {
    WmTWm.resize(N_m, N_m);
    WmTWm.setZero();

    if ((inv_para.max_iter % 2) == 0.0) {
      Array<int> node_id(4);
      Element *inv_cell;

      std::map<unsigned int, std::vector<unsigned int>> nodes_tet;
      for (unsigned int i = 0; i < mesh_inv->GetNV(); i++) {
        for (int j = 0; j < inv_forward.size(); j++) {
          inv_cell = mesh_inv->GetElement(inv_forward[j]);
          inv_cell->GetVertices(node_id);
          for (int k = 0; k < 4; k++) {
            if (i == node_id[k]) {
              nodes_tet[i].push_back(j);
            }
          }
        }
      }

      typedef std::map<unsigned int, std::vector<unsigned int>>::iterator IT;
      std::map<int, std::vector<int>> tet_tet;
      for (unsigned int i = 0; i < inv_forward.size(); i++) {
        inv_cell = mesh_inv->GetElement(inv_forward[i]);
        inv_cell->GetVertices(node_id);
        for (int j = 0; j < 4; j++) {
          IT it = nodes_tet.find(node_id[j]);

          std::vector<unsigned int> &node_tet = (*it).second;
          for (int k = 0; k < node_tet.size(); k++) {
            if (node_tet[k] != i) {
              tet_tet[i].push_back(node_tet[k]);
            }
          }
        }

        std::sort(tet_tet[i].begin(), tet_tet[i].end());
        tet_tet[i].erase(std::unique(tet_tet[i].begin(), tet_tet[i].end()),
                         tet_tet[i].end());
      }
      typedef std::map<int, std::vector<int>>::iterator IT2;

      std::vector<Eigen::Triplet<double>> non_zero;
      non_zero.reserve(N_m * 100);
      for (int i = 0; i < inv_forward.size(); i++) {
        IT2 it2 = tet_tet.find(i);
        std::vector<int> &connect_cells = (*it2).second;
        Vector center;
        mesh_inv->GetElementCenter(inv_forward[i], center);
        double tet_V = mesh_inv->GetElementVolume(inv_forward[i]);
        std::vector<double> delta_r(connect_cells.size());
        std::vector<double> tet_Vi(connect_cells.size());
        double all_V_ce = 0;
        for (int ce = 0; ce < connect_cells.size(); ce++) {
          Vector center_ce;
          mesh_inv->GetElementCenter(inv_forward[connect_cells[ce]], center_ce);
          tet_Vi[ce] =
              mesh_inv->GetElementVolume(inv_forward[connect_cells[ce]]);
          delta_r[ce] = sqrt(pow((center[0] - center_ce[0]), 2) +
                             pow((center[1] - center_ce[1]), 2) +
                             pow((center[2] - center_ce[2]), 2));
          all_V_ce = all_V_ce + tet_Vi[ce];
        }
        for (int ce = 0; ce < connect_cells.size(); ce++) {
          double wight = tet_Vi[ce] / all_V_ce;
          double value = (tet_V * wight) / (delta_r[ce] * delta_r[ce]);
          non_zero.push_back(Eigen::Triplet<double>(i, i, value));
          non_zero.push_back(Eigen::Triplet<double>(connect_cells[ce],
                                                    connect_cells[ce], value));
          non_zero.push_back(
              Eigen::Triplet<double>(i, connect_cells[ce], -1.0 * value));
          non_zero.push_back(
              Eigen::Triplet<double>(connect_cells[ce], i, -1.0 * value));
          if (inv_para.if_anisotropy == 1 || inv_para.if_anisotropy == 2) {
            non_zero.push_back(
                Eigen::Triplet<double>(i + N_inv_cell, i + N_inv_cell, value));
            non_zero.push_back(
                Eigen::Triplet<double>(connect_cells[ce] + N_inv_cell,
                                       connect_cells[ce] + N_inv_cell, value));
            non_zero.push_back(Eigen::Triplet<double>(
                i + N_inv_cell, connect_cells[ce] + N_inv_cell, -1.0 * value));
            non_zero.push_back(Eigen::Triplet<double>(
                connect_cells[ce] + N_inv_cell, i + N_inv_cell, -1.0 * value));
            if (inv_para.if_anisotropy == 2) {
              non_zero.push_back(Eigen::Triplet<double>(
                  i + 2 * N_inv_cell, i + 2 * N_inv_cell, value));
              non_zero.push_back(Eigen::Triplet<double>(
                  connect_cells[ce] + 2 * N_inv_cell,
                  connect_cells[ce] + 2 * N_inv_cell, value));
              non_zero.push_back(Eigen::Triplet<double>(
                  i + 2 * N_inv_cell, connect_cells[ce] + 2 * N_inv_cell,
                  -1.0 * value));
              non_zero.push_back(
                  Eigen::Triplet<double>(connect_cells[ce] + 2 * N_inv_cell,
                                         i + 2 * N_inv_cell, -1.0 * value));
            }
          }
        }
      }
      WmTWm.setFromTriplets(non_zero.begin(), non_zero.end());
      non_zero.clear();
    } else {

      std::vector<int> elem1_inv_id_list;
      std::vector<int> elem2_inv_id_list;
      for (int i = 0; i < mesh_inv->GetNFaces(); i++) {
        int elem1_id, elem2_id;
        mesh_inv->GetFaceElements(i, &elem1_id, &elem2_id);
        int elem1_inv_id = -19210723, elem2_inv_id = -19210723;
        if (elem1_id != -1 && elem2_id != -1) {
          elem1_inv_id = forward_inv[elem1_id];
          elem2_inv_id = forward_inv[elem2_id];
        }
        if (elem1_inv_id != -19210723 && elem2_inv_id != -19210723) {
          elem1_inv_id_list.push_back(elem1_inv_id);
          elem2_inv_id_list.push_back(elem2_inv_id);
        }
      }
      int N_face = elem1_inv_id_list.size();

      RESM Wm;
      if (inv_para.if_anisotropy == 0) {
        Wm.resize(N_face, N_m);
      } else if (inv_para.if_anisotropy == 1) {
        Wm.resize(2 * N_face, N_m);
      } else {
        Wm.resize(3 * N_face, N_m);
      }
      Wm.setZero();
      std::vector<Eigen::Triplet<double>> non_zero;
      non_zero.reserve(N_face * 6);
      for (int f = 0; f < N_face; f++) {
        non_zero.push_back(Eigen::Triplet<double>(f, elem1_inv_id_list[f], -1));
        non_zero.push_back(Eigen::Triplet<double>(f, elem2_inv_id_list[f], 1));
        if (inv_para.if_anisotropy == 1 || inv_para.if_anisotropy == 2) {
          non_zero.push_back(Eigen::Triplet<double>(
              f + N_face, N_inv_cell + elem1_inv_id_list[f], -1));
          non_zero.push_back(Eigen::Triplet<double>(
              f + N_face, N_inv_cell + elem2_inv_id_list[f], 1));
          if (inv_para.if_anisotropy == 2) {
            non_zero.push_back(Eigen::Triplet<double>(
                f + 2 * N_face, 2 * N_inv_cell + elem1_inv_id_list[f], -1));
            non_zero.push_back(Eigen::Triplet<double>(
                f + 2 * N_face, 2 * N_inv_cell + elem2_inv_id_list[f], 1));
          }
        }
      }
      Wm.setFromTriplets(non_zero.begin(), non_zero.end());
      WmTWm = Wm.transpose() * Wm;
      non_zero.clear();
    }
  }
}

double InvBase::vector_threshold(Vector &vec, double ratio) {
  int N_cells = vec.Size();
  if (N_cells == 0)
    return 0.0;

  if (ratio <= 0.0)
    return std::numeric_limits<double>::max();
  if (ratio >= 1.0)
    return -std::numeric_limits<double>::max();

  std::vector<double> eta(N_cells);
  for (int i = 0; i < N_cells; i++)
    eta[i] = vec(i);
  sort(std::begin(eta), std::end(eta));

  size_t index =
      std::min(static_cast<size_t>(N_cells - 1),
               static_cast<size_t>(std::round((1.0 - ratio) * N_cells)));
  return eta[index];
}
