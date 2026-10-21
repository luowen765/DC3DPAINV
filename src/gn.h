/*
 * @Description: Declaration of the Gauss-Newton optimization module.
 * @Author: Lewen Qiu, Huang Chen
 * @Date: 2026-10-21
 */

// Copyright (c) 2026, Lewen Qiu and Huang Chen.
// This file is part of the DC3DPAINV program. DC3DPAINV is free
// software with source code available at
// https://github.com/luowen765/DC3DPAINV. You can redistribute it
// or modify it under the terms of the BSD-3 license. See the
// LICENSE file for details.

#ifndef _GN_H
#define _GN_H

#include "inv_base.h"

class GN : public InvBase {
public:
public:
  GN(ParaHandler &inv_para_, int &world_rank_, int &world_size,

     std::string inv_algorithm_ = "gn");
  virtual ~GN();

  void init();

  void update_model_direction(int N_iter);

  void resetInvParas(int N_iter);
  void clear();
};

#endif
