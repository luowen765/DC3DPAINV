/*
 * @Description: Declaration of the limited-memory BFGS optimization module.
 * @Author: Lewen Qiu, Huang Chen
 * @Date: 2026-10-21
 */

// Copyright (c) 2026, Lewen Qiu and Huang Chen.
// This file is part of the DC3DPAINV program. DC3DPAINV is free
// software with source code available at
// https://github.com/luowen7592/DC3DPAINV. You can redistribute it
// or modify it under the terms of the BSD-3 license. See the
// LICENSE file for details.

#ifndef _LBFGS_H
#define _LBFGS_H

#include "inv_base.h"

class LBFGS : public InvBase {
public:

  std::vector<REV> S_ks;
  std::vector<REV> y_ks;

  REV JTV_kl;
  REV m_kl;

public:
  LBFGS(ParaHandler &inv_para_, int &world_rank_, int &world_size,

        std::string inv_algorithm_ = "lbfgs");
  virtual ~LBFGS();

  void init();

  void update_model_direction(int N_iter);

  void clear();
  void resetInvParas(int N_iter);
};

#endif
