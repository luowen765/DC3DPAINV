/*
 * @Description: Declaration of post-processing utilities for potentials and apparent
 *   resistivities.
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

#ifndef _POST_H
#define _POST_H

#include <fstream>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "parahandler.h"

using namespace mfem;

class Post {
public:

  std::shared_ptr<ParMesh> pmesh;

  std::shared_ptr<ParFiniteElementSpace> pfes;

  std::vector<std::shared_ptr<ParGridFunction>> up;

  ParaHandler &para_handler;

  std::vector<Array<double>> local_Up;

  std::vector<Array<double>> Ex;
  std::vector<Array<double>> Ey;
  std::vector<Array<double>> Ez;
  std::vector<Array<double>> Jx;
  std::vector<Array<double>> Jy;
  std::vector<Array<double>> Jz;

  std::vector<Array<double>> Ut;

public:

  Post(std::shared_ptr<ParMesh> pmesh_,

       std::shared_ptr<ParFiniteElementSpace> pfes_,

       std::vector<std::shared_ptr<ParGridFunction>> up_,
       ParaHandler &para_handler_);

  ~Post();

public:

  void compute_sites_up(std::vector<DenseMatrix> &cond_att,
                        std::vector<DenseMatrix> &sigma0,
                        std ::vector<int> &is_source_on_surface);

  void save_as_one(std::vector<DenseMatrix> &sigma0,
                   std ::vector<int> &is_source_on_surface);

  void group_save_as_one();

  void compute_rho(std::string &survery_mode_file, REV &fw_data);
};

#endif
