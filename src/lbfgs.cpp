/*
 * @Description: L-BFGS model update and correction-history handling.
 * @Author: Lewen Qiu, Huang Chen
 * @Date: 2026-10-21
 */

// Copyright (c) 2026, Lewen Qiu and Huang Chen.
// This file is part of the DC3DPAINV program. DC3DPAINV is free
// software with source code available at
// https://github.com/luowen7592/DC3DPAINV. You can redistribute it
// or modify it under the terms of the BSD-3 license. See the
// LICENSE file for details.

#include "lbfgs.h"

#include "mfemplus.h"

LBFGS::LBFGS(ParaHandler &inv_para_, int &world_rank_, int &world_size_,

             std::string inv_algorithm_)
    : InvBase(inv_para_, world_rank_, world_size_,

              inv_algorithm_) {
  this->init();
}

LBFGS::~LBFGS() { this->clear(); }

void LBFGS::init() {
  if (world_rank == 0) {
    y_ks.resize(inv_para.N_M), S_ks.resize(inv_para.N_M);
  }

  int group_msn = inv_para.n_group_ms;
  loc_d_obs.resize(group_msn);
  loc_Wd.resize(group_msn);

  for (int i = 0; i < group_msn; i++) {
    int gobal_s_id = inv_para.task_ms[i];

    int loc_n_data = inv_para.measure_sites_gather_locId[gobal_s_id].size();
    int global_ms_location = inv_para.ms_locations[gobal_s_id];

    loc_d_obs[i].resize(loc_n_data);
    loc_Wd[i].resize(loc_n_data, loc_n_data);
    for (int j = 0; j < loc_n_data; j++) {

      int col = global_ms_location + j;

      loc_d_obs[i](j) = inv_para.d_obs(col);
      loc_Wd[i].coeffRef(j, j) = inv_para.Wd.coeffRef(col, col);
    }
  }
}

void LBFGS::clear() {}

void LBFGS::update_model_direction(int N_iter) {

  g_k = 2 * JT_V + 2 * lamda_k * WmTWm * (m_k - m_ref);

  int N_M = inv_para.N_M;

  if (N_iter > 1) {
    if (N_iter <= N_M + 1) {
      S_ks[N_iter - 2] = m_k - m_kl;
      y_ks[N_iter - 2] =
          2 * (JT_V - JTV_kl) + 2 * lamda_k * WmTWm * S_ks[N_iter - 2];

    } else {
      for (int i = 0; i < N_M; i++) {
        if (i == N_M - 1) {
          S_ks[i] = m_k - m_kl;
          y_ks[i] = 2 * (JT_V - JTV_kl) + 2 * lamda_k * WmTWm * S_ks[i];
        } else {
          S_ks[i] = S_ks[i + 1];
          y_ks[i] = y_ks[i + 1];
        }
      }
    }
  }

  REV q = g_k;
  update_dire.setZero();
  if (N_iter <= N_M + 1) {
    if (N_iter == 1) {
      update_dire = (1.0 / g_k.norm()) * g_k;
    } else {

      REV alpha(N_iter - 1);
      alpha.setZero();
      for (int i = N_iter - 2; i >= 0; i--) {

        alpha(i) = (S_ks[i].dot(q)) / (y_ks[i].dot(S_ks[i]));

        q.noalias() = q - alpha(i) * y_ks[i];

      }

      double gamma_1 = y_ks[N_iter - 2].dot(S_ks[N_iter - 2]);
      double gamma_2 = y_ks[N_iter - 2].dot(y_ks[N_iter - 2]);
      double gamma = gamma_1 / gamma_2;
      update_dire = gamma * q;

      for (int i = 0; i <= N_iter - 2; i++) {
        double beta = (y_ks[i].dot(update_dire)) / (y_ks[i].dot(S_ks[i]));
        update_dire.noalias() = update_dire + (alpha(i) - beta) * S_ks[i];
      }
    }

  } else {
    REV alpha(N_M);
    alpha.setZero();
    for (int i = N_M - 1; i >= 0; i--) {
      alpha(i) = (S_ks[i].dot(q)) / (y_ks[i].dot(S_ks[i]));
      q.noalias() = q - alpha(i) * y_ks[i];
    }
    double gamma_1 = y_ks[N_M - 1].dot(S_ks[N_M - 1]);
    double gamma_2 = y_ks[N_M - 1].dot(y_ks[N_M - 1]);
    double gamma = gamma_1 / gamma_2;
    update_dire = gamma * q;
    for (int i = 0; i <= N_M - 1; i++) {
      double beta = (y_ks[i].dot(update_dire)) / (y_ks[i].dot(S_ks[i]));
      update_dire.noalias() = update_dire + (alpha(i) - beta) * S_ks[i];
    }
  }
  update_dire.noalias() = -update_dire;

  m_kl = m_k;
  JTV_kl = JT_V;
}

void LBFGS::resetInvParas(int N_iter) {
  InvBase::resetInvParas(N_iter);

  if (world_rank == 0) {
    myassert(N_m > 0);

    if (N_iter == 1) {
      JTV_kl.resize(N_m), m_kl.resize(N_m);
    } else if (N_iter > 1) {

      this->reset_invpara_REV(JTV_kl);
      this->reset_invpara_REV(m_kl);
      for (int i = 0; i < inv_para.N_M; i++) {
        if (S_ks[i].size() > 0 && y_ks[i].size() > 0) {
          this->reset_invpara_REV(S_ks[i]);
          this->reset_invpara_REV(y_ks[i]);
        }
      }

    }
  }

  for (int i = 0; i < mesh_inv->GetNE(); i++) {
    mesh_inv->SetAttribute(i, i + 1);
  }
  mesh_inv->SetAttributes();
}
