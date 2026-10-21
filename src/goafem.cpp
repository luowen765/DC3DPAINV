/*
 * @Description: Implementation of the parallel goal-oriented adaptive finite-element
 *   solver, including forward and adjoint assembly, Jacobian-vector
 *   products, and mesh refinement.
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

#include "goafem.h"

#include "error_estimators.h"

#include "post.h"

#include "mfemplus.h"

#include <algorithm>

#include <cassert>
#include <cmath>

std::vector<double> memoryUsed;
double tempMemory;

GOAFEM::GOAFEM(std::vector<DenseMatrix> &att_cond, ParaHandler &para_handler_,

               std::shared_ptr<ParMesh> pmesh_, std::string &invAlgorithm_)
    : cond_att(att_cond),

      para_handler(para_handler_), pmesh(pmesh_), invAlgorithm(invAlgorithm_)

{
  this->init();
}

GOAFEM::~GOAFEM() {

}

void GOAFEM::init() {
  ns = para_handler.sources.size();
  group_ns = para_handler.n_group_s;
  group_msn = para_handler.n_group_ms;

  N_d = para_handler.d_obs.size();

  group_nd = 0;
  for (int i = 0; i < group_msn; i++) {
    group_nd +=
        para_handler.measure_sites_gather_locId[para_handler.task_ms[i]].size();
  }

  pfec.reset(new H1_FECollection(1, pmesh->Dimension()));
  pfes.reset(new ParFiniteElementSpace(pmesh.get(), pfec.get()));

  U.resize(group_ns);
  F.resize(group_ns);
  up.resize(group_ns);
  f.resize(group_ns);

  if (this->invAlgorithm == "lbfgs") {

    N_adjoint = para_handler.n_group_ms;
  } else {
    myassert(this->invAlgorithm == "gn");

    N_adjoint = para_handler.sites_number;

  }

  X.resize(N_adjoint);
  Y.resize(N_adjoint);
  xgf.resize(N_adjoint);
  y.resize(N_adjoint);

}

void GOAFEM::reconstruct_FEM_terms() {

  a.reset(new ParBilinearForm(pfes.get()));
  for (int i = 0; i < group_ns; i++) {

    up[i].reset(new ParGridFunction(pfes.get()));
    *(up[i]) = 0.0;

    f[i].reset(new ParLinearForm(pfes.get()));
    f[i]->Vector::operator=(0.0);
  }

  w.reset(new ParGridFunction(pfes.get()));
  *w = 0.0;

  l.reset(new ParLinearForm(pfes.get()));

  l->Vector::operator=(0.0);

  for (int j = 0; j < N_adjoint; j++) {

    xgf[j].reset(new ParGridFunction(pfes.get()));
    *(xgf[j]) = 0.0;

    y[j].reset(new ParLinearForm(pfes.get()));
    y[j]->Vector::operator=(0.0);
  }
}

void GOAFEM::set_dual_adjoint_problem(bool dual_problem_,
                                      bool adjoint_problem_) {
  if_solve_dual = dual_problem_;
  if_solve_adjoint = adjoint_problem_;
}

void GOAFEM::one_forward_iteration(
    REV &d_model, REV &JT_V, REM &JT, std::vector<REV> &loc_Koef,
    std::vector<REV> &loc_d_obs, std::vector<RESM> &loc_Wd,
    std::map<int, std::vector<int>> &inv_forward_map,
    std::shared_ptr<double> Trans_mat, int N_inv_cell

) {

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  MPI_Comm mycomm = pmesh->GetComm();
  int group_rank;
  MPI_Comm_rank(mycomm, &group_rank);

  this->set_bdr();

  this->set_sigma0();

  para_handler.find_point_tets(pmesh, para_handler.sites,
                               para_handler.local_sites_tets,
                               para_handler.find_points_by);

  double t2 = MPI_Wtime();

  this->setup_integral_coefficient();

  ess_tdof_list.SetSize(0);

  a->AddDomainIntegrator(new DiffusionIntegrator(*LVolume_coef));

  a->AddBoundaryIntegrator(new MassIntegrator(*LGamma1_coef), gamma1_bdr);

  a->Assemble();

  a->Finalize();

  for (int i = 0; i < para_handler.n_group_s; i++) {
    Vector source_i(3);
    source_i = 0.0;
    for (int j = 0; j < 3; j++) {
      source_i[j] = para_handler.sources[para_handler.task_source[i]](j);
    }
    int if_s_on_surface = para_handler.is_source_on_surface[i];
    RGamma0_coef.reset(new RGamma0Coefficient(
        sigma0[para_handler.task_source[i]], source_i, if_s_on_surface));
    RGamma1_coef.reset(
        new RGamma1Coefficient(cond_att, sigma0[para_handler.task_source[i]],
                               source_i, if_s_on_surface, pmesh));

    f[i]->AddBoundaryIntegrator(new BoundaryLFIntegrator(*RGamma0_coef),
                                gamma0_bdr);

    f[i]->AddBoundaryIntegrator(new BoundaryLFIntegrator(*RGamma1_coef),
                                gamma1_bdr);
    f[i]->Assemble();

  }

  tempMemory = GetCurrentMemoryUsage();

  assemble_RV_linearform(f);

  for (int i = 0; i < group_ns; i++) {

    a->FormLinearSystem(ess_tdof_list, *(up[i]), *(f[i]), A, U[i], F[i]);
  }

  if (if_solve_dual) {
    assemble_dual_linearform(para_handler.local_sites_tets);

    a->FormLinearSystem(ess_tdof_list, *w, *l, A, W, L);
  }

  std::string linear_solver = para_handler.linear_solver;
  std::shared_ptr<MUMPSSolver> mumps;

  if (world_rank == 0) {
    std::cout << "  Solving the primal problem ..." << std::endl;
  }
  if (if_solve_dual) {
    if (world_rank == 0) {
      std::cout << "  And solving the dual problem..." << std::endl;
    }
  }

  double t2_end = MPI_Wtime();
  if (world_rank == 0) {
    std::cout << "    Assemble the primal problem time: " << t2_end - t2
              << std::endl;
  }

  double t3 = MPI_Wtime();

  if (linear_solver == "mumps") {

    mumps.reset(new MUMPSSolver(para_handler.group_comm));

    mumps->SetPrintLevel(0);
    mumps->SetMatrixSymType(MUMPSSolver::MatType::SYMMETRIC_POSITIVE_DEFINITE);

    mumps->SetOperator(*A.Ptr());

    tempMemory = GetCurrentMemoryUsage();
    memoryUsed.push_back(tempMemory);

    for (int i = 0; i < F.size(); i++) {

      mumps->Mult(F[i], U[i]);
    }
    if (if_solve_dual) {
      mumps->Mult(L, W);
    }
  }

  else {
    myassert(linear_solver == "amg" || linear_solver == "ilu");
    for (int i = 0; i < F.size(); i++) {

      solve_with_pcg(A, U[i], F[i], linear_solver, para_handler.pcg_maxit,
                     "primal"

      );
    }
    if (if_solve_dual) {
      solve_with_pcg(A, W, L, linear_solver, para_handler.pcg_maxit, "dual");
    }
  }

  for (int i = 0; i < F.size(); i++) {
    a->RecoverFEMSolution(U[i], *(f[i]), *(up[i]));
  }

  if (if_solve_dual) {
    a->RecoverFEMSolution(W, *l, *w);
  }

  double t3_end = MPI_Wtime();
  if (world_rank == 0) {
    std::cout << "    Solve the primal problem time: " << t3_end - t3
              << std::endl;
  }

  if (world_rank == 0)
    std::cout << "    The number of the forward unknowns: " << pmesh->GetNV()
              << std::endl;

  int local_dofs = pfes->GetTrueVSize();
  int global_dofs = 0;
  MPI_Allreduce(&local_dofs, &global_dofs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if (global_dofs > para_handler.max_dofs) {
    if (world_rank == 0) {
      std::cout << "\nStop due to reached max number of dofs\n";
      std::cout << "Present dofs: " << global_dofs << "\n\n";
    }
    return;
  }
  if (global_dofs > 2147483648) {
    if (world_rank == 0) {
      std::cout << "\nmax_dofs has exited the range of int-type data, please "
                   "modify your code!";
    }
    return;
  }

  if (this->if_solve_adjoint) {
    if (world_rank == 0) {
      std::cout << "  And solving the adjoint problem ..." << std::endl;
    }

    int t_ani = para_handler.if_anisotropy;
    if (t_ani == 0) {
      N_m = N_inv_cell;
    } else if (t_ani == 1) {
      N_m = N_inv_cell * 2;
    } else {
      N_m = N_inv_cell * 3;
    }

    this->compute_d_obs(d_model, world_rank);

    double t4_1 = MPI_Wtime();

    if (this->invAlgorithm == "lbfgs") {

      std::unique_ptr<double> fwData(new double[N_d]);

      if (world_rank == 0) {
        for (int i = 0; i < N_d; i++) {
          fwData.get()[i] = d_model(i);
        }
        for (int p = 1; p < world_size; p++) {

          MPI_Send(fwData.get(), N_d, MPI_DOUBLE, p, 106, MPI_COMM_WORLD);
        }
      } else {

        MPI_Recv(fwData.get(), N_d, MPI_DOUBLE, 0, 106, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
      }

      std::vector<REV> loc_fw_data;

      loc_fw_data.resize(group_msn);

      for (int i = 0; i < group_msn; i++) {
        int global_s_id = para_handler.task_ms[i];
        int loc_n_data =
            para_handler.measure_sites_gather_locId[global_s_id].size();
        int global_ms_location = para_handler.ms_locations[global_s_id];
        loc_fw_data[i].resize(loc_n_data);

        for (int j = 0; j < loc_n_data; j++) {

          int col = global_ms_location + j;
          loc_fw_data[i](j) = fwData.get()[col];
        }
      }

      REV vec;

      for (int i = 0; i < N_adjoint; i++) {

        int loc_n_data = loc_d_obs[i].size();
        myassert(loc_Wd[i].rows() == loc_n_data);
        vec.resize(loc_n_data);
        vec.setZero();

        if (para_handler.inv_data_type == 0) {
          vec = loc_Wd[i].transpose() * loc_Wd[i] *
                (loc_fw_data[i] - loc_d_obs[i]);
        } else {
          myassert(para_handler.inv_data_type == 1);

          myassert(loc_Koef[i].size() == loc_n_data);
          vec = loc_Koef[i].cwiseProduct(loc_Wd[i].transpose() * loc_Wd[i] *
                                         (loc_fw_data[i] - loc_d_obs[i]));
        }

        assemble_adjoint_linearform(i, vec);
      }
    } else if (this->invAlgorithm == "gn") {
      REV vec;
      vec.resize(0);
      vec.setZero();
      for (int i = 0; i < N_adjoint; i++) {
        assemble_adjoint_linearform(i, vec);
      }
    }

    myassert(a != NULL);
    for (int i = 0; i < N_adjoint; i++) {
      a->FormLinearSystem(ess_tdof_list, *(xgf[i]), *(y[i]), A, X[i], Y[i]);
    }

    if (linear_solver == "mumps") {
      myassert(mumps != NULL);
      for (int i = 0; i < N_adjoint; i++) {
        mumps->Mult(Y[i], X[i]);
      }
    }

    else {
      for (int i = 0; i < N_adjoint; i++) {

        solve_with_pcg(A, X[i], Y[i], linear_solver, para_handler.pcg_maxit,
                       "adjoint");

      }
    }

    for (int i = 0; i < N_adjoint; i++) {
      a->RecoverFEMSolution(X[i], *(y[i]), *(xgf[i]));
    }

    if (world_rank == 0) {
      myassert(JT_V.size() == N_m);
      JT_V.setZero();
    }

    if (para_handler.if_parallel_source == true)

      this->save_up_as_one_sources();

    if (this->invAlgorithm == "lbfgs") {

      REV JTV_temp(N_m);
      JTV_temp.setZero();

      int N_dofs = pmesh->GetNV();

      REV X_(N_dofs);

      for (int i = 0; i < group_msn; i++) {

        for (int j = 0; j < N_dofs; j++) {
          X_(j) = (*(xgf[i]))(j);
        }
        RESM Km_E;
        Km_E.resize(N_dofs, N_m);
        int t_ani = para_handler.if_anisotropy;
        if (t_ani == 0) {
          Km_E.resize(N_dofs, N_inv_cell);
        } else if (t_ani == 1) {
          Km_E.resize(N_dofs, 2 * N_inv_cell);
        } else {
          Km_E.resize(N_dofs, 3 * N_inv_cell);
        }
        Km_E.setZero();

        assemble_KmE(para_handler.task_ms[i], Km_E, inv_forward_map, Trans_mat,
                     N_inv_cell, para_handler.is_source_on_surface);

        JTV_temp += Km_E.transpose() * X_;

      }

      save_jtu_as_one(JTV_temp);

      if (para_handler.n_group > 1) {
        group_save_jtu_as_one(JTV_temp);
      }

      if (world_rank == 0) {

        JT_V = JTV_temp;

      }
    } else if (this->invAlgorithm == "gn") {

      int N_dofs = pmesh->GetNV();

      REV X_(N_dofs);

      REM JT_temp(N_m, group_nd);
      JT_temp.setZero();

      tempMemory = GetCurrentMemoryUsage();

      int local_ms_location = 0;

      for (int i = 0; i < group_msn; i++) {
        RESM Km_E;
        Km_E.resize(N_dofs, N_m);
        int t_ani = para_handler.if_anisotropy;
        if (t_ani == 0) {
          Km_E.resize(N_dofs, N_inv_cell);
        } else if (t_ani == 1) {
          Km_E.resize(N_dofs, 2 * N_inv_cell);
        } else {
          Km_E.resize(N_dofs, 3 * N_inv_cell);
        }
        Km_E.setZero();

        assemble_KmE(para_handler.task_ms[i], Km_E, inv_forward_map, Trans_mat,
                     N_inv_cell, para_handler.is_source_on_surface);

        int global_s_id = para_handler.task_ms[i];
        int n_measure =
            para_handler.measure_sites_gather_locId[global_s_id].size();
        for (int j = 0; j < n_measure; j++) {
          if (para_handler.survey_mode == 11 ||
              para_handler.survey_mode == 21) {
            Point_double p;
            for (int hh = 0; hh < 3; hh++) {
              p(hh) = para_handler.sites[para_handler.measure_sites_gather_locId
                                             [global_s_id][j][0]](hh);
            }
            int local_id_in_sites = para_handler.p_map[p];
            for (int k = 0; k < N_dofs; k++) {
              X_(k) = (*(xgf[local_id_in_sites]))(k);
            }

          } else if (para_handler.survey_mode == 22 ||
                     para_handler.survey_mode == 12) {

            Point_double p1, p2;
            for (int hh = 0; hh < 3; hh++) {
              p1(hh) =
                  para_handler.sites[para_handler.measure_sites_gather_locId
                                         [global_s_id][j][0]](hh);
              p2(hh) =
                  para_handler.sites[para_handler.measure_sites_gather_locId
                                         [global_s_id][j][1]](hh);
            }
            int local_id_in_sites_M = para_handler.p_map[p1];
            int local_id_in_sites_N = para_handler.p_map[p2];

            for (int k = 0; k < N_dofs; k++) {
              X_(k) = (*(xgf[local_id_in_sites_M]))(k) -
                      (*(xgf[local_id_in_sites_N]))(k);
            }

          } else {
            std::cout << "unvailable mode!\n";
            std::getchar();
          }

          int data_index = local_ms_location + j;

          if (para_handler.inv_data_type == 0) {
            JT_temp.col(data_index) = Km_E.transpose() * X_;
          } else {
            myassert(para_handler.inv_data_type == 1);
            JT_temp.col(data_index) = (Km_E.transpose() * X_) * loc_Koef[i][j];
          }
        }

        local_ms_location = local_ms_location + n_measure;
      }

      save_jt_as_one(JT_temp);

      if (para_handler.n_group > 1) {
        group_save_jt_as_one(JT_temp);
      }
      if (world_rank == 0) {

        myassert(JT.rows() == N_m);
        myassert(JT.cols() == N_d);

        myassert(JT_temp.rows() == N_m);
        myassert(JT_temp.cols() == N_d);

        JT = JT_temp;

        REV vec(N_d);

        myassert(d_model.size() > 0);
        vec = para_handler.Wd.transpose() * para_handler.Wd *
              (d_model - para_handler.d_obs);
        JT_V = JT * vec;

      }
    }

  }

  up_all.resize(0);

  if (this->if_refine_mesh) {
    if (world_rank == 0) {
      std::cout << "  Start refining mesh..." << std::endl;
    }

    this->refine_mesh();
  }
}

void GOAFEM::set_sigma0() {

  Array<int> local_sources_tets;
  para_handler.find_point_tets(pmesh, para_handler.sources, local_sources_tets,
                               para_handler.find_points_by);

  MPI_Comm mycomm = pmesh->GetComm();
  int group_rank, n_procs;
  MPI_Comm_size(mycomm, &n_procs);
  MPI_Comm_rank(mycomm, &group_rank);

  MPI_Barrier(mycomm);

  std::unique_ptr<int> msg_tets;
  global_initial_sources_tets.SetSize(ns);

  if (group_rank == 0) {

    for (int i = 0; i < ns; i++) {
      int tet_id = local_sources_tets[i];
      if (tet_id != -2) {

        global_initial_sources_tets[i] = pmesh->GetAttribute(tet_id) - 1;
      } else {
        global_initial_sources_tets[i] = 0;
      }
    }
    for (int p = 1; p < n_procs; p++) {
      msg_tets.reset(new int[ns]);

      MPI_Recv(msg_tets.get(), ns, MPI_INT, p, 105, mycomm, MPI_STATUS_IGNORE);

      for (int i = 0; i < ns; i++) {
        global_initial_sources_tets[i] += msg_tets.get()[i];
      }

    }
  } else {

    msg_tets.reset(new int[ns]);
    for (int i = 0; i < ns; i++) {
      int tet_id = local_sources_tets[i];
      if (tet_id != -2) {

        msg_tets.get()[i] = pmesh->GetAttribute(tet_id) - 1;
      } else {
        msg_tets.get()[i] = 0;
      }
    }

    MPI_Send(msg_tets.get(), ns, MPI_INT, 0, 105, mycomm);

  }

  std::unique_ptr<int> sources_tets_g(new int[ns]);

  if (group_rank == 0) {
    for (int j = 0; j < ns; j++) {
      sources_tets_g.get()[j] = global_initial_sources_tets[j];
    }
    for (int p = 1; p < n_procs; p++) {
      MPI_Send(sources_tets_g.get(), ns, MPI_INT, p, 6, mycomm);
    }
  } else {

    MPI_Recv(sources_tets_g.get(), ns, MPI_INT, 0, 6, mycomm,
             MPI_STATUS_IGNORE);

    myassert(global_initial_sources_tets.Size() > 0);
    for (int j = 0; j < ns; j++) {
      global_initial_sources_tets[j] = sources_tets_g.get()[j];
    }
  }

  sigma0.clear();

  sigma0.resize(ns);
  for (int k = 0; k < ns; k++) {

    sigma0[k] = cond_att[global_initial_sources_tets[k]];
  }
}

void GOAFEM::setup_integral_coefficient() {

  LVolume_coef.reset(new LVCoefficient(cond_att));

  Vector sources_center = para_handler.get_sources_center();

  LGamma1_coef.reset(new LGamma1Coefficient(cond_att, sources_center, pmesh));

}

void GOAFEM::set_bdr() {

  myassert(pmesh->bdr_attributes[0] == 1);
  myassert(pmesh->bdr_attributes[1] == 2);

  gamma0_bdr.SetSize(pmesh->bdr_attributes.Size());
  gamma1_bdr.SetSize(pmesh->bdr_attributes.Size());
  gamma0_bdr = 0;
  gamma1_bdr = 0;
  gamma0_bdr[0] = 1;
  gamma1_bdr[1] = 1;
}

void GOAFEM::assemble_dual_linearform(Array<int> &local_dual_tets_) {
  for (int i = 0; i < local_dual_tets_.Size(); i++) {
    int tet_id = local_dual_tets_[i];
    if (tet_id != -2) {
      const FiniteElement *fe = pfes->GetFE(tet_id);

      const IntegrationRule *ir =
          &IntRules.Get(fe->GetGeomType(), fe->GetOrder());

      ElementTransformation *eltrans = pfes->GetElementTransformation(tet_id);
      double elem_volume = pmesh->GetElementVolume(tet_id);
      const int ndof = fe->GetDof();

      Vector shape(ndof);
      shape = 0.0;
      Vector elemvec(ndof);
      elemvec = 0.0;
      for (int q = 0; q < ir->GetNPoints(); ++q) {
        const IntegrationPoint &ip = ir->IntPoint(q);
        eltrans->SetIntPoint(&ip);
        double WJ = eltrans->Weight() * ip.weight;

        fe->CalcPhysShape(*eltrans, shape);
        for (int k = 0; k < ndof; k++)
        {

          elemvec[k] += WJ * elem_volume * shape[k];
        }
      }
      Array<int> vdofs;
      pfes->GetElementDofs(tet_id, vdofs);

      l->AddElementVector(vdofs, elemvec);
    }
  }
}

void GOAFEM::solve_with_pcg(OperatorHandle &A, Vector &X, Vector &B,
                            std::string pre, int maxit, std::string problem_type

) {
  double tol = 0.0;
  if (problem_type == "primal" || problem_type == "adjoint") {
    tol = para_handler.pcg_primal_tol;
  } else if (problem_type == "dual") {
    tol = para_handler.pcg_dual_tol;
  } else {
    std::cout << "please check your problem_type! primal or dual?" << std::endl;
    std::abort();
  }

  std::shared_ptr<Solver> prec;
  if (pre == "amg") {

    std::shared_ptr<HypreBoomerAMG> amg(new HypreBoomerAMG);
    amg->SetPrintLevel(0);
    prec = amg;
  } else if (pre == "ilu") {

    std::shared_ptr<HypreILU> ilu(new HypreILU);
    ilu->SetPrintLevel(0);
    prec = ilu;
  } else {
    std::cout << "Unsupported input linear solver: "
              << para_handler.linear_solver << "\n";
    std::abort();
  }

  SolversPlus cg(para_handler.group_comm);

  cg.SetMaxIter(maxit);
  cg.SetRelTol(tol);
  cg.SetPrintLevel(0);
  if (prec) {
    cg.SetPreconditioner(*prec);
  }
  cg.SetOperator(*A);

  tempMemory = GetCurrentMemoryUsage();
  memoryUsed.push_back(tempMemory);

  cg.Mult(B, X);

}

void GOAFEM::assemble_adjoint_linearform(int adjointId, REV &vec) {

  if (invAlgorithm == "lbfgs") {

    int group_s_id = para_handler.task_ms[adjointId];
    int n_measure = para_handler.measure_sites_gather_locId[group_s_id].size();

    for (int i = 0; i < n_measure; i++) {

      if (para_handler.survey_mode == 11 || para_handler.survey_mode == 21) {

        Vector pt(3);
        Point_double p;
        for (int t = 0; t < 3; t++) {
          pt[t] =
              para_handler
                  .sites[para_handler
                             .measure_sites_gather_locId[group_s_id][i][0]](t);
          p(t) = pt[t];
        }
        int local_id_in_sites = para_handler.p_map[p];
        int tet_id = para_handler.local_sites_tets[local_id_in_sites];

        if (tet_id != -2) {

          assemble_adjoint_single_site(true, tet_id, pt, adjointId, vec[i]);
        }
      } else if (para_handler.survey_mode == 22 ||
                 para_handler.survey_mode == 12) {

        Vector pt1(3), pt2(3);
        Point_double p1, p2;
        for (int t = 0; t < 3; t++) {
          pt1[t] =
              para_handler
                  .sites[para_handler
                             .measure_sites_gather_locId[group_s_id][i][0]](t);

          pt2[t] =
              para_handler
                  .sites[para_handler
                             .measure_sites_gather_locId[group_s_id][i][1]](t);
          p1(t) = pt1[t];
          p2(t) = pt2[t];
        }
        int local_id_in_sites_M = para_handler.p_map[p1];
        int local_id_in_sites_N = para_handler.p_map[p2];
        int tet1_id = para_handler.local_sites_tets[local_id_in_sites_M];
        int tet2_id = para_handler.local_sites_tets[local_id_in_sites_N];

        if (tet1_id != -2) {
          assemble_adjoint_single_site(true, tet1_id, pt1, adjointId, vec[i]);
        }

        if (tet2_id != -2) {
          assemble_adjoint_single_site(false, tet2_id, pt2, adjointId, vec[i]);
        }

      } else {
        std::cout << "unvailable mode!\n";
        std::getchar();
      }
    }
  } else {
    myassert(invAlgorithm == "gn");

    Point_double p;
    Vector pt(3);
    for (int t = 0; t < 3; t++) {
      pt[t] = para_handler.sites[adjointId](t);
      p(t) = pt[t];
    }
    int local_id_in_sites = para_handler.p_map[p];
    int tet_id = para_handler.local_sites_tets[local_id_in_sites];
    if (tet_id != -2) {
      assemble_adjoint_single_site(true, tet_id, pt, adjointId);
    }
  }
}

void GOAFEM::assemble_KmE(
    int msId, RESM &Km_E, std::map<int, std::vector<int>> &inv_forward_map,
    std::shared_ptr<double> Trans_mat, int N_inv_cell,
    std ::vector<int> &is_source_on_surface) {

  typedef std::map<int, std::vector<int>>::iterator IT;
  std::vector<Eigen::Triplet<double>> non_zero;

  non_zero.reserve(3 * 4 * inv_forward_map.size());

  if (para_handler.if_parallel_source == true) {
    myassert(up_all.size() > 0);
  }

  for (IT it = inv_forward_map.begin(); it != inv_forward_map.end(); it++) {
    int inv_id = (*it).first;
    std::vector<int> &dense_grids = (*it).second;
    Array<int> vdofs;

    Array<int> rows(4);

    Array<int> cols(3);

    cols[0] = inv_id, cols[1] = inv_id + N_inv_cell,
    cols[2] = inv_id + 2 * N_inv_cell;

    DenseMatrix per_cell;

    for (int k = 0; k < dense_grids.size(); k++) {
      int tet_id = dense_grids[k];
      const FiniteElement *fe = pfes->GetFE(tet_id);

      double intergrationOrder = 0;
      if (para_handler.integration_order == 1) {
        intergrationOrder = fe->GetOrder();
      } else if (para_handler.integration_order == 2) {
        intergrationOrder = 2 * fe->GetOrder();
      } else {
        intergrationOrder = 2 * fe->GetOrder() + 3;
      }

      const IntegrationRule *ir =
          &IntRules.Get(fe->GetGeomType(), intergrationOrder);

      ElementTransformation *eltrans = pfes->GetElementTransformation(tet_id);

      const int ndof = fe->GetDof();
      const int dim = fe->GetDim();
      per_cell.Clear();
      per_cell.SetSize(ndof, 3);

      DenseMatrix shapeV(ndof, dim);
      pfes->GetElementDofs(tet_id, vdofs);

      if (para_handler.survey_mode == 11 || para_handler.survey_mode == 12) {
        myassert(para_handler.measure_sources_locId[msId].size() == 1);
        int sourceId = para_handler.measure_sources_locId[msId][0];
        Vector up_elem(4);
        up_elem = 0.0;

        if (para_handler.if_parallel_source == true) {
          for (int ii = 0; ii < 4; ii++) {
            up_elem[ii] = up_all[sourceId][vdofs[ii]];
          }
        } else {
          up[sourceId]->GetSubVector(vdofs, up_elem);
        }

        Point_double delta_Uis = 0.0;
        Point_double xi, xp;
        for (int ii = 0; ii < 3; ii++) {
          xi(ii) = para_handler.sources[sourceId](ii);
        }
        Matrix3D sigma_0;
        for (int gg = 0; gg < 3; gg++)
          for (int hh = 0; hh < 3; hh++) {
            {
              sigma_0(gg, hh) = sigma0[sourceId](gg, hh);
            }
          }

        for (int q = 0; q < ir->GetNPoints(); ++q) {
          const IntegrationPoint &ip = ir->IntPoint(q);
          eltrans->SetIntPoint(&ip);
          Vector p(3);
          p = 0.0;
          eltrans->Transform(ip, p);

          for (int ii = 0; ii < 3; ii++) {
            xp(ii) = p[ii];
          }
          delta_Uis =
              gradient_Uis(sigma_0, xp, xi, is_source_on_surface[sourceId]);

          double WJ = eltrans->Weight() * ip.weight;
          fe->CalcPhysDShape(*eltrans, shapeV);

          for (int l = 0; l < ndof; l++) {
            rows[l] = vdofs[l];

            if (para_handler.if_anisotropy == 0) {
              double l1 = -1.0 * (shapeV(l, 0) * delta_Uis(0) +
                                  shapeV(l, 1) * delta_Uis(1) +
                                  shapeV(l, 2) * delta_Uis(2));
              double j1 = 0.0;

              for (int j = 0; j < ndof; j++) {

                j1 +=
                    (shapeV(l, 0) * shapeV(j, 0) + shapeV(l, 1) * shapeV(j, 1) +
                     shapeV(l, 2) * shapeV(j, 2)) *
                    up_elem[j];
              }
              per_cell(l, 0) += WJ * (l1 - j1) * Trans_mat.get()[inv_id];

            } else if (para_handler.if_anisotropy == 1) {
              double l1 = -1.0 * (shapeV(l, 0) * delta_Uis(0) +
                                  shapeV(l, 1) * delta_Uis(1));
              double l2 = -1.0 * (shapeV(l, 2) * delta_Uis(2));
              double j1 = 0.0, j2 = 0.0;

              for (int j = 0; j < ndof; j++) {
                j1 += (shapeV(l, 0) * shapeV(j, 0) +
                       shapeV(l, 1) * shapeV(j, 1)) *
                      up_elem[j];
                j2 += (shapeV(l, 2) * shapeV(j, 2)) * up_elem[j];
              }
              per_cell(l, 0) += WJ * (l1 - j1) * Trans_mat.get()[inv_id];
              per_cell(l, 1) +=
                  WJ * (l2 - j2) * Trans_mat.get()[inv_id + N_inv_cell];

            } else {
              double l1 = -1.0 * (shapeV(l, 0) * delta_Uis(0));
              double l2 = -1.0 * (shapeV(l, 1) * delta_Uis(1));
              double l3 = -1.0 * (shapeV(l, 2) * delta_Uis(2));

              double j1 = 0.0, j2 = 0.0, j3 = 0.0;

              for (int j = 0; j < ndof; j++) {
                j1 += (shapeV(l, 0) * shapeV(j, 0)) * up_elem[j];
                j2 += (shapeV(l, 1) * shapeV(j, 1)) * up_elem[j];
                j3 += (shapeV(l, 2) * shapeV(j, 2)) * up_elem[j];
              }
              per_cell(l, 0) += WJ * (l1 - j1) * Trans_mat.get()[inv_id];
              per_cell(l, 1) +=
                  WJ * (l2 - j2) * Trans_mat.get()[inv_id + N_inv_cell];
              per_cell(l, 2) +=
                  WJ * (l3 - j3) * Trans_mat.get()[inv_id + 2 * N_inv_cell];
            }

          }
        }
      } else if (para_handler.survey_mode == 21 ||
                 para_handler.survey_mode == 22) {
        myassert(para_handler.measure_sources_locId[msId].size() == 2);
        int source1_Id = para_handler.measure_sources_locId[msId][0];
        int source2_Id = para_handler.measure_sources_locId[msId][1];
        Vector up1_elem(4), up2_elem(4);
        up1_elem = 0.0;
        up2_elem = 0.0;
        if (para_handler.if_parallel_source == true) {
          for (int ii = 0; ii < 4; ii++) {

            up1_elem[ii] = up_all[source1_Id][vdofs[ii]];
            up2_elem[ii] = up_all[source2_Id][vdofs[ii]];
          }
        } else {
          up[source1_Id]->GetSubVector(vdofs, up1_elem);
          up[source2_Id]->GetSubVector(vdofs, up2_elem);
        }

        Point_double delta_Uis1, delta_Uis2;
        Point_double xi1, xi2, xp;
        for (int ii = 0; ii < 3; ii++) {
          xi1(ii) = para_handler.sources[source1_Id](ii);
          xi2(ii) = para_handler.sources[source2_Id](ii);
        }
        Matrix3D sigma0_1, sigma0_2;
        for (int gg = 0; gg < 3; gg++)
          for (int hh = 0; hh < 3; hh++) {
            {
              sigma0_1(gg, hh) = sigma0[source1_Id](gg, hh);
              sigma0_2(gg, hh) = sigma0[source2_Id](gg, hh);
            }
          }

        for (int q = 0; q < ir->GetNPoints(); ++q) {
          const IntegrationPoint &ip = ir->IntPoint(q);
          eltrans->SetIntPoint(&ip);
          Vector p(3);
          p = 0.0;
          eltrans->Transform(ip, p);

          for (int ii = 0; ii < 3; ii++) {
            xp(ii) = p[ii];
          }
          delta_Uis1 =
              gradient_Uis(sigma0_1, xp, xi1, is_source_on_surface[source1_Id]);
          delta_Uis2 =
              gradient_Uis(sigma0_2, xp, xi2, is_source_on_surface[source2_Id]);

          double WJ = eltrans->Weight() * ip.weight;
          fe->CalcPhysDShape(*eltrans, shapeV);
          for (int l = 0; l < ndof; l++) {
            rows[l] = vdofs[l];

            if (para_handler.if_anisotropy == 0) {

              double l11 = (-1.0 * (shapeV(l, 0) * delta_Uis1(0) +
                                    shapeV(l, 1) * delta_Uis1(1) +
                                    shapeV(l, 2) * delta_Uis1(2)));
              double l12 = (-1.0 * (shapeV(l, 0) * delta_Uis2(0) +
                                    shapeV(l, 1) * delta_Uis2(1) +
                                    shapeV(l, 2) * delta_Uis2(2)));
              double j11 = 0.0;
              double j12 = 0.0;

              for (int j = 0; j < ndof; j++) {

                double temp =
                    (shapeV(l, 0) * shapeV(j, 0) + shapeV(l, 1) * shapeV(j, 1) +
                     shapeV(l, 2) * shapeV(j, 2));
                j11 += temp * up1_elem[j];
                j12 += temp * up2_elem[j];
              }

              per_cell(l, 0) +=
                  WJ * (l11 - j11 - l12 + j12) * Trans_mat.get()[inv_id];

            } else if (para_handler.if_anisotropy == 1) {

              double l11 = (-1.0 * (shapeV(l, 0) * delta_Uis1(0) +
                                    shapeV(l, 1) * delta_Uis1(1)));
              double l12 = (-1.0 * (shapeV(l, 0) * delta_Uis2(0) +
                                    shapeV(l, 1) * delta_Uis2(1)));
              double l21 = (-1.0 * (shapeV(l, 2) * delta_Uis1(2)));
              double l22 = (-1.0 * (shapeV(l, 2) * delta_Uis2(2)));

              double j11 = 0.0;
              double j12 = 0.0;
              double j21 = 0.0;
              double j22 = 0.0;

              for (int j = 0; j < ndof; j++) {

                double temp1 =
                    (shapeV(l, 0) * shapeV(j, 0) + shapeV(l, 1) * shapeV(j, 1));
                j11 += temp1 * up1_elem[j];
                j12 += temp1 * up2_elem[j];
                double temp2 = (shapeV(l, 2) * shapeV(j, 2));
                j21 += temp2 * up1_elem[j];
                j22 += temp2 * up2_elem[j];
              }
              per_cell(l, 0) +=
                  WJ * (l11 - j11 - l12 + j12) * Trans_mat.get()[inv_id];
              per_cell(l, 1) += WJ * (l21 - j21 - l22 + j22) *
                                Trans_mat.get()[inv_id + +N_inv_cell];

            } else {
              double l11 = (-1.0 * (shapeV(l, 0) * delta_Uis1(0)));
              double l12 = (-1.0 * (shapeV(l, 0) * delta_Uis2(0)));
              double l21 = (-1.0 * (shapeV(l, 1) * delta_Uis1(1)));
              double l22 = (-1.0 * (shapeV(l, 1) * delta_Uis2(1)));
              double l31 = (-1.0 * (shapeV(l, 2) * delta_Uis1(2)));
              double l32 = (-1.0 * (shapeV(l, 2) * delta_Uis2(2)));

              double j11 = 0.0;
              double j12 = 0.0;
              double j21 = 0.0;
              double j22 = 0.0;
              double j31 = 0.0;
              double j32 = 0.0;

              for (int j = 0; j < ndof; j++) {

                double temp1 = (shapeV(l, 0) * shapeV(j, 0));
                j11 += temp1 * up1_elem[j];
                j12 += temp1 * up2_elem[j];
                double temp2 = (shapeV(l, 1) * shapeV(j, 1));
                j21 += temp2 * up1_elem[j];
                j22 += temp2 * up2_elem[j];
                double temp3 = (shapeV(l, 2) * shapeV(j, 2));
                j31 += temp3 * up1_elem[j];
                j32 += temp3 * up2_elem[j];
              }
              per_cell(l, 0) +=
                  WJ * (l11 - j11 - l12 + j12) * Trans_mat.get()[inv_id];
              per_cell(l, 1) += WJ * (l21 - j21 - l22 + j22) *
                                Trans_mat.get()[inv_id + +N_inv_cell];
              per_cell(l, 2) += WJ * (l31 - j31 - l32 + j32) *
                                Trans_mat.get()[inv_id + 2 * N_inv_cell];
            }

          }
        }
      } else {
        std::cout << "Unsopported mode!" << std::endl;
        std::abort();
      }

      for (int ri = 0; ri < 4; ri++) {

        non_zero.push_back(
            Eigen::Triplet<double>(rows[ri], cols[0], per_cell(ri, 0)));
        if (para_handler.if_anisotropy == 1 ||
            para_handler.if_anisotropy == 2) {
          non_zero.push_back(
              Eigen::Triplet<double>(rows[ri], cols[1], per_cell(ri, 1)));
          if (para_handler.if_anisotropy == 2) {
            non_zero.push_back(
                Eigen::Triplet<double>(rows[ri], cols[2], per_cell(ri, 2)));
          }
        }
      }

    }
  }
  if (inv_forward_map.size() != 0) {
    Km_E.setFromTriplets(non_zero.begin(), non_zero.end());
  }
  non_zero.clear();
}

void GOAFEM::save_jtu_as_one(REV &JT_V_) {

  MPI_Status status;
  MPI_Comm mycomm = pmesh->GetComm();
  int group_rank, n_procs;
  MPI_Comm_size(mycomm, &n_procs);
  MPI_Comm_rank(mycomm, &group_rank);

  MPI_Barrier(mycomm);

  std::unique_ptr<double> local_jtu(new double[N_m]);
  for (int i = 0; i < N_m; i++) {
    local_jtu.get()[i] = JT_V_(i);
  }

  if (group_rank == 0) {
    for (int p = 1; p < n_procs; p++) {
      MPI_Recv(local_jtu.get(), N_m, MPI_DOUBLE, p, 2, mycomm, &status);
      for (int i = 0; i < N_m; i++) {
        JT_V_(i) = JT_V_(i) + local_jtu.get()[i];
      }
    }
  } else {
    MPI_Send(local_jtu.get(), N_m, MPI_DOUBLE, 0, 2, mycomm);
  }

}

void GOAFEM::save_jt_as_one(REM &JT_) {

  MPI_Status status;
  MPI_Comm mycomm = pmesh->GetComm();
  int group_rank, n_procs;
  MPI_Comm_size(mycomm, &n_procs);
  MPI_Comm_rank(mycomm, &group_rank);
  MPI_Barrier(mycomm);

  for (int j = 0; j < group_nd; j++) {

    std::unique_ptr<double> local_jt_col(new double[N_m]);
    for (int i = 0; i < N_m; i++) {
      local_jt_col.get()[i] = JT_(i, j);
    }
    if (group_rank == 0) {
      for (int p = 1; p < n_procs; p++) {
        MPI_Recv(local_jt_col.get(), N_m, MPI_DOUBLE, p, 2, mycomm, &status);
        for (int i = 0; i < N_m; i++) {
          JT_(i, j) = JT_(i, j) + local_jt_col.get()[i];
        }
      }
    } else {
      MPI_Send(local_jt_col.get(), N_m, MPI_DOUBLE, 0, 2, mycomm);
    }

  }
}

void GOAFEM::refine_mesh() {

  MPI_Comm group_comm = pmesh->GetComm();

  int group_size, group_rank;

  MPI_Comm_size(group_comm, &group_size);
  MPI_Comm_rank(group_comm, &group_rank);

  this->error_estimating();
  double threshold;
  if (para_handler.amr_type == 0) {
    int N_loc_cell = pmesh->GetNE();
    int N_glo_cell = pmesh->GetGlobalNE();

    double loc_eta[N_loc_cell];
    for (int i = 0; i < N_loc_cell; i++) {
      loc_eta[i] = local_err(i);
    }
    if (group_rank == 0) {
      std::vector<double> global_eta(N_glo_cell);
      for (int i = 0; i < N_loc_cell; i++) {
        global_eta[i] = loc_eta[i];
      }
      int input_begin = N_loc_cell;

      std::unique_ptr<double> eta_oth;
      for (int np = 1; np < group_size; np++) {
        MPI_Recv(&N_loc_cell, 1, MPI_INT, np, 102, group_comm,
                 MPI_STATUS_IGNORE);

        eta_oth.reset(new double[N_loc_cell]);
        MPI_Recv(eta_oth.get(), N_loc_cell, MPI_DOUBLE, np, 103, group_comm,
                 MPI_STATUS_IGNORE);
        for (int i = 0; i < N_loc_cell; i++) {
          global_eta[input_begin + i] = eta_oth.get()[i];
        }
        input_begin += N_loc_cell;

      }
      myassert(input_begin == N_glo_cell);
      sort(std::begin(global_eta), std::end(global_eta));
      threshold = global_eta[round((1 - para_handler.amr_ratio) * N_glo_cell)];

      for (int np = 1; np < group_size; np++) {
        MPI_Send(&threshold, 1, MPI_DOUBLE, np, 104, group_comm);
      }
    } else {
      MPI_Send(&N_loc_cell, 1, MPI_INT, 0, 102, group_comm);
      MPI_Send(&loc_eta, N_loc_cell, MPI_DOUBLE, 0, 103, group_comm);
      MPI_Recv(&threshold, 1, MPI_DOUBLE, 0, 104, group_comm,
               MPI_STATUS_IGNORE);
    }
  } else {
    double local_max_err = local_err.Max();
    double global_max_err;
    MPI_Allreduce(&local_max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX,
                  pmesh->GetComm());
    threshold = para_handler.amr_ratio * global_max_err;
  }

  pmesh->RefineByError(local_err, threshold);

}

void GOAFEM::error_estimating() {

  local_err.SetSize(pmesh->GetNE());
  local_err = 0.0;

  if (para_handler.error_esti_way == "jn") {

    nJEstimator nJ_estimator(pmesh, pfes, up, w, cond_att);

    nJ_estimator.get_error_estimates(if_solve_dual, local_err);
  } else if (para_handler.error_esti_way == "zz_patch") {

    SPREstimator spr_estimator(pmesh, pfes, up, w, cond_att);
    spr_estimator.get_error_estimates(if_solve_dual, local_err);

  } else {
    std::cout << "unsupported estimator!" << std::endl;
  }

}

void GOAFEM::update() {
  pfes->Update();
  a->Update();

  for (int i = 0; i < group_ns; i++) {
    f[i]->Update();
    up[i]->Update();

  }
  l->Update();
  w->Update();

  for (int j = 0; j < group_msn; j++) {
    y[j]->Update();
    xgf[j]->Update();

  }
}

void GOAFEM::compute_d_obs(REV &d_model_, int world_rank) {

  Post post(pmesh, pfes, up, para_handler);

  post.compute_sites_up(cond_att, sigma0, para_handler.is_source_on_surface);
  post.save_as_one(sigma0, para_handler.is_source_on_surface);

  if (para_handler.n_group > 1) {
    post.group_save_as_one();
  }

  if (world_rank == 0) {

    myassert(d_model_.size() > 0);

    post.compute_rho(para_handler.obs_data_file, d_model_);
  }
}

void GOAFEM::assemble_adjoint_single_site(bool if_site_M, int tet_id, Vector pt,
                                          int adjointId, double vecI) {

  if (tet_id != -2) {
    const FiniteElement *fe = pfes->GetFE(tet_id);
    ElementTransformation *eltrans = pfes->GetElementTransformation(tet_id);
    IntegrationPoint ip;
    eltrans->TransformBack(pt, ip);
    eltrans->SetIntPoint(&ip);

    const int ndof = fe->GetDof();

    Vector shape(ndof);
    shape = 0.0;
    Vector elemvec(ndof);
    elemvec = 0.0;
    fe->CalcPhysShape(*eltrans, shape);

    if (invAlgorithm == "lbfgs") {
      if (if_site_M == true) {
        for (int k = 0; k < ndof; k++) {

          elemvec[k] = shape(k) * vecI;

        }
      } else {
        for (int k = 0; k < ndof; k++) {
          elemvec[k] = -1.0 * shape(k) * vecI;
        }
      }

    } else {
      myassert(vecI == 0.0);
      if (if_site_M == true) {
        for (int k = 0; k < ndof; k++) {
          elemvec[k] = shape(k);

        }
      } else {
        for (int k = 0; k < ndof; k++) {
          elemvec[k] = -1.0 * shape(k);
        }
      }
    }
    Array<int> vdofs;

    pfes->GetElementDofs(tet_id, vdofs);

    y[adjointId]->AddElementVector(vdofs, elemvec);
  }
}

void GOAFEM::assemble_RV_linearform(
    std::vector<std::shared_ptr<ParLinearForm>> f_) {

  for (int e = 0; e < pmesh->GetNE(); e++) {
    const FiniteElement *fe = pfes->GetFE(e);

    double intergrationOrder = 0;
    if (para_handler.integration_order == 1) {
      intergrationOrder = fe->GetOrder();
    } else if (para_handler.integration_order == 2) {
      intergrationOrder = 2 * fe->GetOrder();
    } else {
      intergrationOrder = 2 * fe->GetOrder() + 3;
    }

    const IntegrationRule *ir =
        &IntRules.Get(fe->GetGeomType(), intergrationOrder);

    ElementTransformation *eltrans = pfes->GetElementTransformation(e);

    const int ndof = fe->GetDof();
    const int dim = fe->GetDim();

    std::vector<Vector> elemvec_(f_.size());
    for (int i = 0; i < f_.size(); i++) {
      elemvec_[i].SetSize(ndof);
      elemvec_[i] = 0.0;
    }

    for (int q = 0; q < ir->GetNPoints(); ++q) {

      const IntegrationPoint &ip = ir->IntPoint(q);
      eltrans->SetIntPoint(&ip);

      Vector x(3);
      x = 0.0;
      eltrans->Transform(ip, x);

      double WJ = eltrans->Weight() * ip.weight;

      DenseMatrix shapeV(ndof, dim);
      fe->CalcPhysDShape(*eltrans, shapeV);

      for (int i = 0; i < group_ns; i++) {
        Matrix3D sigma_0;
        for (int j = 0; j < 3; j++)
          for (int k = 0; k < 3; k++) {
            {

              sigma_0(j, k) = sigma0[para_handler.task_source[i]](j, k);
            }
          }
        Point_double xp, xi;
        for (int j = 0; j < 3; j++) {
          xp(j) = x(j);

          xi(j) = para_handler.sources[para_handler.task_source[i]](j);
        }
        Point_double grad_U_i_s =
            gradient_Uis(sigma_0, xp, xi, para_handler.is_source_on_surface[i]);

        Vector V(3);

        V = 0.0;

        DenseMatrix temp = sigma0[para_handler.task_source[i]];
        temp -= cond_att[eltrans->Attribute - 1];

        for (int j = 0; j < 3; j++) {
          for (int k = 0; k < 3; k++) {

            V(k) += temp(j, k) * grad_U_i_s(k);
          }
        }

        for (int k = 0; k < ndof; k++)
        {
          elemvec_[i][k] += WJ * (shapeV(k, 0) * V(0) + shapeV(k, 1) * V(1) +
                                  shapeV(k, 2) * V(2));
        }
      }

    }

    Array<int> vdofs;
    pfes->GetElementDofs(e, vdofs);

    for (int i = 0; i < group_ns; i++) {
      f[i]->AddElementVector(vdofs, elemvec_[i]);
    }
  }

}

void GOAFEM::group_save_jtu_as_one(REV &JT_V_) {
  MPI_Barrier(MPI_COMM_WORLD);

  std::unique_ptr<double> local_jtu(new double[N_m]);
  for (int i = 0; i < N_m; i++) {
    local_jtu.get()[i] = JT_V_(i);
  }
  MPI_Status status;
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_rank == 0) {

    for (int p = 1; p < para_handler.n_group; p++) {
      MPI_Recv(local_jtu.get(), N_m, MPI_DOUBLE, p, 2, MPI_COMM_WORLD, &status);
      for (int i = 0; i < N_m; i++) {
        JT_V_(i) = JT_V_(i) + local_jtu.get()[i];
      }
    }
  } else {
    myassert(world_rank > 0);

    if (para_handler.group_rank == 0)
      MPI_Send(local_jtu.get(), N_m, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
  }
}
void GOAFEM::group_save_jt_as_one(REM &JT_) {
  MPI_Barrier(MPI_COMM_WORLD);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Status status;

  std::unique_ptr<double> local_jt_col(new double[N_m]);

  int n_ms_transfer;
  int local_task_ms = -1;

  if (world_rank == 0) {

    REM JT_temp = JT_;
    myassert(JT_temp.rows() == N_m && JT_temp.cols() == group_nd);
    JT_.resize(N_m, N_d);
    JT_.setZero();

    int local_ms_location = 0;
    for (int i = 0; i < group_msn; i++) {
      int global_s_id = para_handler.task_ms[i];

      int n_measure =
          para_handler.measure_sites_gather_locId[global_s_id].size();
      int global_ms_location = para_handler.ms_locations[global_s_id];

      for (int j = 0; j < n_measure; j++) {

        int local_ms_index = local_ms_location + j;
        int global_ms_index = global_ms_location + j;
        JT_.col(global_ms_index) = JT_temp.col(local_ms_index);
      }
      local_ms_location = local_ms_location + n_measure;
    }
    for (int p = 1; p < para_handler.n_group; p++) {
      MPI_Recv(&n_ms_transfer, 1, MPI_INT, p, 101, MPI_COMM_WORLD, &status);

      myassert(n_ms_transfer > 0);

      for (int i = 0; i < n_ms_transfer; i++) {
        MPI_Recv(&local_task_ms, 1, MPI_INT, p, 102, MPI_COMM_WORLD, &status);

        int loc_n_data =
            para_handler.measure_sites_gather_locId[local_task_ms].size();
        int global_ms_location = para_handler.ms_locations[local_task_ms];

        for (int j = 0; j < loc_n_data; j++) {

          MPI_Recv(local_jt_col.get(), N_m, MPI_DOUBLE, p, 110, MPI_COMM_WORLD,
                   &status);

          int global_ms_index = global_ms_location + j;

          for (int k = 0; k < N_m; k++) {
            JT_.col(global_ms_index)[k] = local_jt_col.get()[k];
          }
        }
      }
    }
  } else {
    myassert(world_rank > 0);
    if (para_handler.group_rank == 0) {
      n_ms_transfer = group_ns;
      MPI_Send(&n_ms_transfer, 1, MPI_INT, 0, 101, MPI_COMM_WORLD);

      int local_ms_location = 0;
      for (int i = 0; i < n_ms_transfer; i++) {
        local_task_ms = para_handler.task_ms[i];
        MPI_Send(&local_task_ms, 1, MPI_INT, 0, 102, MPI_COMM_WORLD);

        int loc_n_data =
            para_handler.measure_sites_gather_locId[local_task_ms].size();
        for (int j = 0; j < loc_n_data; j++) {

          int data_index = local_ms_location + j;
          for (int k = 0; k < N_m; k++) {
            local_jt_col.get()[k] = JT_.col(data_index)[k];
          }
        }
        MPI_Send(local_jt_col.get(), N_m, MPI_DOUBLE, 0, 110, MPI_COMM_WORLD);

        local_ms_location = local_ms_location + loc_n_data;
      }
    }
  }
}

void GOAFEM::save_up_as_one_sources() {

  myassert(para_handler.n_group > 1 && para_handler.group_size == 1);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Status status;
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int nv = pmesh->GetNV();

  std::unique_ptr<double> local_up(new double[nv]);
  int ns_transfer;
  int local_task_s = -1;

  up_all.resize(ns);
  for (int i = 0; i < ns; i++) {
    up_all[i].resize(nv);
  }

  if (world_rank == 0) {
    for (int i = 0; i < group_ns; i++) {
      for (int j = 0; j < nv; j++) {
        up_all[para_handler.task_source[i]][j] = (*(up[i]))(j);
      }
    }

    for (int p = 1; p < para_handler.n_group; p++) {
      MPI_Recv(&ns_transfer, 1, MPI_INT, p, 101, MPI_COMM_WORLD, &status);

      myassert(ns_transfer > 0);

      for (int i = 0; i < ns_transfer; i++) {

        MPI_Recv(&local_task_s, 1, MPI_INT, p, 102, MPI_COMM_WORLD, &status);
        MPI_Recv(local_up.get(), nv, MPI_DOUBLE, p, 110, MPI_COMM_WORLD,
                 &status);

        for (int j = 0; j < nv; j++) {
          up_all[local_task_s][j] = local_up.get()[j];
        }
      }
    }
  } else {
    myassert(world_rank > 0);
    myassert(para_handler.group_rank == 0);
    ns_transfer = group_ns;
    MPI_Send(&ns_transfer, 1, MPI_INT, 0, 101, MPI_COMM_WORLD);

    for (int i = 0; i < ns_transfer; i++) {
      local_task_s = para_handler.task_source[i];

      MPI_Send(&local_task_s, 1, MPI_INT, 0, 102, MPI_COMM_WORLD);

      for (int j = 0; j < nv; j++) {
        local_up.get()[j] = (*(up[i]))(j);
      }

      MPI_Send(local_up.get(), nv, MPI_DOUBLE, 0, 110, MPI_COMM_WORLD);
    }
  }

  if (world_rank == 0) {
    for (int p = 1; p < para_handler.n_group; p++) {
      for (int i = 0; i < ns; i++) {
        for (int j = 0; j < nv; j++) {
          local_up.get()[j] = up_all[i][j];
        }
        MPI_Send(local_up.get(), nv, MPI_DOUBLE, p, 111, MPI_COMM_WORLD);
      }
    }
  } else {
    for (int i = 0; i < ns; i++) {
      MPI_Recv(local_up.get(), nv, MPI_DOUBLE, 0, 111, MPI_COMM_WORLD, &status);
      for (int j = 0; j < nv; j++) {
        up_all[i][j] = local_up.get()[j];
      }
    }
  }

}

void GOAFEM::compute_jtv(

    REV &X_, REV &JTV_temp, std::map<int, std::vector<int>> &inv_forward_map,

    std::shared_ptr<double> Trans_mat, int N_inv_cell,
    std ::vector<int> &is_source_on_surface) {

  typedef std::map<int, std::vector<int>>::iterator IT;
  std::vector<Eigen::Triplet<double>> non_zero;

  non_zero.reserve(3 * 4 * inv_forward_map.size());

  if (para_handler.if_parallel_source == true) {
    myassert(up_all.size() > 0);
  }

  myassert(para_handler.survey_mode == 22);
  myassert(para_handler.if_anisotropy == 0);
  int N_dofs = pmesh->GetNV();

  for (int msId = 0; msId < group_msn; msId++) {
    for (int j = 0; j < N_dofs; j++) {
      X_(j) = (*(xgf[msId]))(j);
    }

    myassert(para_handler.measure_sources_locId[msId].size() == 2);

    int source1_Id = para_handler.measure_sources_locId[msId][0];
    int source2_Id = para_handler.measure_sources_locId[msId][1];

    for (IT it = inv_forward_map.begin(); it != inv_forward_map.end(); it++) {
      int inv_id = (*it).first;
      std::vector<int> &dense_grids = (*it).second;
      Array<int> vdofs;

      Array<int> rows(4);

      Array<int> cols(3);

      cols[0] = inv_id, cols[1] = inv_id + N_inv_cell,
      cols[2] = inv_id + 2 * N_inv_cell;

      DenseMatrix per_cell;

      for (int k = 0; k < dense_grids.size(); k++) {
        int tet_id = dense_grids[k];
        const FiniteElement *fe = pfes->GetFE(tet_id);

        double intergrationOrder = 0;
        if (para_handler.integration_order == 1) {
          intergrationOrder = fe->GetOrder();
        } else if (para_handler.integration_order == 2) {
          intergrationOrder = 2 * fe->GetOrder();
        } else {
          intergrationOrder = 2 * fe->GetOrder() + 3;
        }

        const IntegrationRule *ir =
            &IntRules.Get(fe->GetGeomType(), intergrationOrder);

        ElementTransformation *eltrans = pfes->GetElementTransformation(tet_id);
        const int ndof = fe->GetDof();
        const int dim = fe->GetDim();
        per_cell.Clear();
        per_cell.SetSize(ndof, 3);

        DenseMatrix shapeV(ndof, dim);
        pfes->GetElementDofs(tet_id, vdofs);

        Vector up1_elem(4), up2_elem(4);
        up1_elem = 0.0;
        up2_elem = 0.0;
        if (para_handler.if_parallel_source == true) {
          for (int ii = 0; ii < 4; ii++) {

            up1_elem[ii] = up_all[source1_Id][vdofs[ii]];
            up2_elem[ii] = up_all[source2_Id][vdofs[ii]];
          }
        } else {
          up[source1_Id]->GetSubVector(vdofs, up1_elem);
          up[source2_Id]->GetSubVector(vdofs, up2_elem);
        }

        Point_double delta_Uis1, delta_Uis2;
        Point_double xi1, xi2, xp;
        for (int ii = 0; ii < 3; ii++) {
          xi1(ii) = para_handler.sources[source1_Id](ii);
          xi2(ii) = para_handler.sources[source2_Id](ii);
        }
        Matrix3D sigma0_1, sigma0_2;
        for (int gg = 0; gg < 3; gg++)
          for (int hh = 0; hh < 3; hh++) {
            {
              sigma0_1(gg, hh) = sigma0[source1_Id](gg, hh);
              sigma0_2(gg, hh) = sigma0[source2_Id](gg, hh);
            }
          }

        for (int q = 0; q < ir->GetNPoints(); ++q) {
          const IntegrationPoint &ip = ir->IntPoint(q);
          eltrans->SetIntPoint(&ip);
          Vector p(3);
          p = 0.0;
          eltrans->Transform(ip, p);

          for (int ii = 0; ii < 3; ii++) {
            xp(ii) = p[ii];
          }
          delta_Uis1 =
              gradient_Uis(sigma0_1, xp, xi1, is_source_on_surface[source1_Id]);
          delta_Uis2 =
              gradient_Uis(sigma0_2, xp, xi2, is_source_on_surface[source2_Id]);

          double WJ = eltrans->Weight() * ip.weight;
          fe->CalcPhysDShape(*eltrans, shapeV);
          for (int l = 0; l < ndof; l++) {
            rows[l] = vdofs[l];

            double l11 = (-1.0 * (shapeV(l, 0) * delta_Uis1(0) +
                                  shapeV(l, 1) * delta_Uis1(1) +
                                  shapeV(l, 2) * delta_Uis1(2)));
            double l12 = (-1.0 * (shapeV(l, 0) * delta_Uis2(0) +
                                  shapeV(l, 1) * delta_Uis2(1) +
                                  shapeV(l, 2) * delta_Uis2(2)));
            double j11 = 0.0;
            double j12 = 0.0;

            for (int j = 0; j < ndof; j++) {

              double temp =
                  (shapeV(l, 0) * shapeV(j, 0) + shapeV(l, 1) * shapeV(j, 1) +
                   shapeV(l, 2) * shapeV(j, 2));
              j11 += temp * up1_elem[j];
              j12 += temp * up2_elem[j];
            }

            per_cell(l, 0) += WJ * (l11 - j11 - l12 + j12);

          }
        }

        for (int ri = 0; ri < 4; ri++) {

          JTV_temp[cols[0]] +=
              per_cell(ri, 0) * X_(rows[ri]) * Trans_mat.get()[cols[0]];
        }

      }

    }
  }
}
