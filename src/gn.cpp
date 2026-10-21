/*
 * @Description: Gauss-Newton model update and PCG solution of the normal equations.
 * @Author: Lewen Qiu, Huang Chen
 * @Date: 2026-10-21
 */

// Copyright (c) 2026, Lewen Qiu and Huang Chen.
// This file is part of the DC3DPAINV program. DC3DPAINV is free
// software with source code available at
// https://github.com/luowen765/DC3DPAINV. You can redistribute it
// or modify it under the terms of the BSD-3 license. See the
// LICENSE file for details.

#include "gn.h"
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/PardisoSupport>

GN::GN(ParaHandler &inv_para_, int &world_rank_, int &world_size_,

       std::string inv_algorithm_)
    : InvBase(inv_para_, world_rank_, world_size_,

              inv_algorithm_) {
  this->init();
}

GN::~GN() { this->clear(); }

void GN::init() {}

void GN::clear() {}

void GN::update_model_direction(int N_iter) {

  update_dire.setZero();

  g_k = 2 * JT_V + 2 * lamda_k * WmTWm * (m_k - m_ref);

  RESM ME(N_m, N_m);
  ME.setIdentity();

  RESM M(N_m, N_m);

  M = 2 * lamda_k * WmTWm + ME;

  REV r(N_m), z(N_m), p(N_m);
  r.setZero(), z.setZero(), p.setZero();
  double alpha1, alpha2, alpha, beta1, beta2, beta;
  r = -g_k;

  double tol = inv_para.GN_PCG_TOL;

  Eigen::PardisoLLT<RESM> solver;

  solver.analyzePattern(M);
  solver.factorize(M);
  z = solver.solve(r);
  p = z;
  double e0 = r.norm();
  REV tempAP(N_m);
  tempAP.setZero();

  for (int i = 0; i < inv_para.GN_PCG_max_iter; i++) {
    alpha1 = (r.transpose() * z);

    tempAP = 2 * (JT * (inv_para.Wd.transpose() * inv_para.Wd) *
                      (JT.transpose() * p) +
                  lamda_k * WmTWm * p)

             + ME * p;
    alpha2 = p.transpose() * tempAP;
    alpha = alpha1 / alpha2;
    update_dire += alpha * p;

    r.noalias() = r - alpha * tempAP;
    double iter_err = r.norm() / e0;
    std::cout << "  GN_Iter_num: " << N_iter << ", PCG_Iter_num: " << i
              << ", PCG_Error: " << iter_err << std::endl;
    if (iter_err <= tol || i == inv_para.GN_PCG_max_iter - 1) {

      break;
    }

    z = solver.solve(r);

    beta1 = (z.transpose() * r);

    beta2 = alpha1;
    beta = beta1 / beta2;

    p.noalias() = z + beta * p;
  }
}

void GN::resetInvParas(int N_iter) {
  InvBase::resetInvParas(N_iter);
  if (world_rank == 0) {
    myassert(N_m > 0);
    myassert(N_d > 0);

    if (N_iter == 1) {
      JT.resize(N_m, N_d);
    } else {
      REM last_JT = JT;

      int n_comp = 1;
      if (inv_para.if_anisotropy == 1)
        n_comp = 2;
      else if (inv_para.if_anisotropy == 2)
        n_comp = 3;
      int last_N_inv_cell = last_JT.rows() / n_comp;
      JT.resize(N_m, N_d);

      for (int i = 0; i < N_inv_cell; i++) {
        int last_forward_id = mesh_inv->GetAttribute(inv_forward[i]) - 1;

        int last_inv_id = last_forward_inv[last_forward_id];
        JT.row(i) = last_JT.row(last_inv_id);

        if (inv_para.if_anisotropy == 1 || inv_para.if_anisotropy == 2) {
          JT.row(i + N_inv_cell) =
              last_JT.row(last_inv_id + last_N_inv_cell);
        }
        if (inv_para.if_anisotropy == 2) {
          JT.row(i + 2 * N_inv_cell) =
              last_JT.row(last_inv_id + 2 * last_N_inv_cell);
        }
      }
    }
  }

  for (int i = 0; i < mesh_inv->GetNE(); i++) {
    mesh_inv->SetAttribute(i, i + 1);
  }
  mesh_inv->SetAttributes();
}
