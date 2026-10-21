/*
 * @Description: Implementation of post-processing and VTK/text output.
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

#include "post.h"

#include "mfemplus.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <mpi.h>

Post::Post(

    std::shared_ptr<ParMesh> pmesh_,

    std::shared_ptr<ParFiniteElementSpace> pfes_,

    std::vector<std::shared_ptr<ParGridFunction>> up_,
    ParaHandler &para_handler_

    )
    : pmesh(pmesh_), pfes(pfes_), up(up_), para_handler(para_handler_)

{}
Post::~Post() {}

void Post::compute_sites_up(std::vector<DenseMatrix> &cond_att,
                            std::vector<DenseMatrix> &sigma0,
                            std ::vector<int> &is_source_on_surface) {

  int group_ns = para_handler.n_group_s;

  local_Up.resize(group_ns);

  int nsite = para_handler.sites.size();
  myassert(nsite > 0);
  for (int k = 0; k < group_ns; k++) {

    local_Up[k].SetSize(nsite);
    local_Up[k] = 0.0;

    for (int i = 0; i < nsite; i++) {

      myassert(para_handler.local_sites_tets.Size() != 0);
      int tet_id = para_handler.local_sites_tets[i];
      if (tet_id != -2) {
        const FiniteElement *fe = pfes->GetFE(tet_id);
        int ndof = fe->GetDof();

        Vector pt(3);
        pt = 0.0;

        pt[0] = para_handler.sites[i](0);
        pt[1] = para_handler.sites[i](1);
        pt[2] = para_handler.sites[i](2);
        IntegrationPoint ip;
        ElementTransformation *tr = pfes->GetElementTransformation(tet_id);
        tr->TransformBack(pt, ip);
        tr->SetIntPoint(&ip);

        Vector shape(ndof);
        shape = 0.0;
        fe->CalcPhysShape(*tr, shape);

        Array<int> vdofs;

        pfes->GetElementDofs(tet_id, vdofs);

        Vector tempUp;
        tempUp = 0.0;
        up[k]->GetSubVector(vdofs, tempUp);
        double potential = 0.0;
        for (int j = 0; j < ndof; j++) {
          potential += shape[j] * tempUp[j];
        }

        local_Up[k][i] = potential;

      } else {
        myassert(local_Up[k][i] == 0.0);

      }
    }
  }
}

void Post::save_as_one(std::vector<DenseMatrix> &sigma0,
                       std ::vector<int> &is_source_on_surface) {

  MPI_Comm mycomm = pmesh->GetComm();
  MPI_Barrier(mycomm);

  MPI_Status status;
  int myid, n_procs;
  MPI_Comm_size(mycomm, &n_procs);
  MPI_Comm_rank(mycomm, &myid);

  int count = -1;
  double *msg_Up;

  int group_ns = para_handler.n_group_s;

  Ut.resize(group_ns);
  for (int k = 0; k < group_ns; k++) {
    if (myid == 0) {
      for (int p = 1; p < n_procs; p++) {
        MPI_Recv(&count, 1, MPI_INT, p, 101, mycomm, &status);
        myassert(count != -1);
        msg_Up = new double[count];

        MPI_Recv(msg_Up, count, MPI_DOUBLE, p, 105, mycomm, &status);

        for (int i = 0; i < count; i++) {
          local_Up[k][i] += msg_Up[i];
        }
        delete msg_Up;
      }

      int nsite = para_handler.sites.size();
      Ut[k].SetSize(nsite);

      Point_double pi;
      for (int kk = 0; kk < 3; kk++) {

        pi(kk) = para_handler.sources[para_handler.task_source[k]](kk);
      }
      Matrix3D sigma_0;
      for (int gg = 0; gg < 3; gg++)
        for (int hh = 0; hh < 3; hh++) {
          {

            sigma_0(gg, hh) = sigma0[para_handler.task_source[k]](gg, hh);
          }
        }

      for (int h = 0; h < nsite; h++) {

        Point_double pj;
        for (int gg = 0; gg < 3; gg++) {
          pj(gg) = para_handler.sites[h](gg);
        }

        if (pj(0) == pi(0) && pj(1) == pi(1) && pj(2) == pi(2)) {
          Ut[k][h] = 1E10;
        } else {

          double Us = Uis(sigma_0, pj, pi, is_source_on_surface[k]);
          Ut[k][h] = local_Up[k][h] + Us;
        }
      }
    } else {
      count = local_Up[k].Size();

      msg_Up = local_Up[k].GetData();

      MPI_Send(&count, 1, MPI_INT, 0, 101, mycomm);
      MPI_Send(msg_Up, count, MPI_DOUBLE, 0, 105, mycomm);
    }
  }
}

void Post::compute_rho(std::string &survery_mode_file, REV &fw_data) {

  std::ifstream mode_stream(survery_mode_file.c_str());
  myassert(mode_stream.good());

  int N_electrodes;
  std::vector<Point_double> electrodes;
  std::vector<int> is_electrode_on_surface;
  mode_stream >> N_electrodes;
  myassert(N_electrodes > 0);
  for (int i = 0; i < N_electrodes; i++) {
    Point_double electrode;
    int is_on_earth_surface;
    mode_stream >> electrode(0) >> electrode(1) >> electrode(2) >>
        is_on_earth_surface;
    myassert(is_on_earth_surface == 0 || is_on_earth_surface == 1);
    electrodes.push_back(electrode);
    is_electrode_on_surface.push_back(is_on_earth_surface);
  }
  myassert(electrodes.size() == N_electrodes);

  unsigned int n_data = 0;
  unsigned int mode = 0;
  mode_stream >> n_data
      >> mode;
  myassert(n_data > 0);
  myassert(mode == 11 || mode == 12 || mode == 21 || mode == 22);
  unsigned int n = 0;
  if (mode == 11)
    n = 2;
  else if (mode == 12)
    n = 3;
  else if (mode == 21)
    n = 3;
  else if (mode == 22)
    n = 4;
  myassert(n >= 2);

  myassert(fw_data.size() == n_data);
  myassert(Ut.size() == para_handler.sources.size() &&
           Ut[0].Size() == para_handler.sites.size());

  for (int i = 0; i < n_data; i++) {
    double obs_u_temp, noise_u_temp, obs_rho_temp, noise_rho_temp;
    double U = 0.0;
    if (mode == 11) {

      Point_double C_node, P_node;

      int s_id = -1, m_id = -1;
      mode_stream >> s_id >> m_id;
      for (int j = 0; j < 3; j++) {
        C_node(j) = electrodes[s_id - 1](j);
        P_node(j) = electrodes[m_id - 1](j);
      }

      unsigned int C_local_id = para_handler.c_map[C_node];
      unsigned int P_local_id = para_handler.p_map[P_node];

      U = this->Ut[C_local_id][P_local_id];

    } else if (mode == 12) {

      Point_double C1_node, P1_node, P2_node;

      int s1_id = -1, m1_id = -1, m2_id = -1;
      mode_stream >> s1_id >> m1_id >> m2_id;
      for (int j = 0; j < 3; j++) {
        C1_node(j) = electrodes[s1_id - 1](j);
        P1_node(j) = electrodes[m1_id - 1](j);
        P2_node(j) = electrodes[m2_id - 1](j);
      }

      unsigned int C1_local_id = para_handler.c_map[C1_node];
      unsigned int P1_local_id = para_handler.p_map[P1_node];
      unsigned int P2_local_id = para_handler.p_map[P2_node];
      U = this->Ut[C1_local_id][P1_local_id] -
          this->Ut[C1_local_id][P2_local_id];

    } else if (mode == 21) {

      Point_double C1_node, C2_node, P1_node;

      int s1_id = -1, s2_id = -1, m1_id = -1;
      mode_stream >> s1_id >> s2_id >> m1_id;
      for (int j = 0; j < 3; j++) {
        C1_node(j) = electrodes[s1_id - 1](j);
        C2_node(j) = electrodes[s2_id - 1](j);
        P1_node(j) = electrodes[m1_id - 1](j);
      }

      unsigned int C1_local_id = para_handler.c_map[C1_node];
      unsigned int C2_local_id = para_handler.c_map[C2_node];
      unsigned int P1_local_id = para_handler.p_map[P1_node];
      U = (this->Ut[C1_local_id][P1_local_id] -
           this->Ut[C2_local_id][P1_local_id]);

    } else if (mode == 22) {

      Point_double C1_node, C2_node, P1_node, P2_node;

      int s1_id = -1, s2_id = -1, m1_id = -1, m2_id = -1;
      mode_stream >> s1_id >> s2_id >> m1_id >> m2_id;
      for (int j = 0; j < 3; j++) {
        C1_node(j) = electrodes[s1_id - 1](j);
        C2_node(j) = electrodes[s2_id - 1](j);
        P1_node(j) = electrodes[m1_id - 1](j);
        P2_node(j) = electrodes[m2_id - 1](j);
      }

      unsigned int C1_local_id = para_handler.c_map[C1_node];
      unsigned int C2_local_id = para_handler.c_map[C2_node];
      unsigned int P1_local_id = para_handler.p_map[P1_node];
      unsigned int P2_local_id = para_handler.p_map[P2_node];
      U = (this->Ut[C1_local_id][P1_local_id] -
           this->Ut[C2_local_id][P1_local_id]) -
          (this->Ut[C1_local_id][P2_local_id] -
           this->Ut[C2_local_id][P2_local_id]);

    }

    if (para_handler.inv_data_type == 0) {
      fw_data(i) = U;
    } else {
      myassert(para_handler.inv_data_type == 1);
      double k = para_handler.K_coef(i);
      fw_data(i) = k * U;

      if (k == 0.0) {

        fw_data(i) = 0.0;
      } else {
        fw_data(i) = k * U;
        if (!std::isfinite(fw_data(i))) {
          std::cout << "fw_data(i) is infinite! \n";
          fw_data(i) = 0.0;
        }

        if (fw_data(i) < 0.0) {
          if (para_handler.noise[i] != 1e12) {
            std::cout << "para_handler.noise[i]: " << para_handler.noise[i]
                      << "\n";
            std::cout << "New rho < 0.0" << std::endl;
          }

        }

      }
    }

    mode_stream >> obs_u_temp >> noise_u_temp >> obs_rho_temp >> noise_rho_temp;
  }

  mode_stream.close();

  return;
}

void Post::group_save_as_one() {
  MPI_Barrier(MPI_COMM_WORLD);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Status status;

  int nsite = para_handler.sites.size();

  std::unique_ptr<double[]> msg_Ut(new double[nsite]);

  int group_ns = para_handler.n_group_s;
  int n_source_transfer;
  int local_task_s = -1;

  if (world_rank == 0) {
    std::vector<Array<double>> Ut_0 = Ut;
    int total_ns = para_handler.sources.size();
    Ut.resize(total_ns);
    for (int i = 0; i < total_ns; i++) {
      Ut[i].SetSize(nsite);

    }

    myassert(Ut_0.size() == group_ns && Ut_0[0].Size() == nsite);
    for (int i = 0; i < group_ns; i++) {
      Ut[para_handler.task_source[i]] = Ut_0[i];
    }

    for (int p = 1; p < para_handler.n_group; p++) {
      MPI_Recv(&n_source_transfer, 1, MPI_INT, p, 101, MPI_COMM_WORLD, &status);

      myassert(n_source_transfer > 0);

      for (int i = 0; i < n_source_transfer; i++) {
        MPI_Recv(&local_task_s, 1, MPI_INT, p, 102, MPI_COMM_WORLD, &status);
        MPI_Recv(msg_Ut.get(), nsite, MPI_DOUBLE, p, 110, MPI_COMM_WORLD,
                 &status);
        for (int j = 0; j < nsite; j++) {
          Ut[local_task_s][j] = msg_Ut.get()[j];
        }
      }
    }

  } else {
    myassert(world_rank > 0);
    if (para_handler.group_rank == 0) {
      n_source_transfer = group_ns;

      MPI_Send(&n_source_transfer, 1, MPI_INT, 0, 101, MPI_COMM_WORLD);
      for (int i = 0; i < n_source_transfer; i++) {
        local_task_s = para_handler.task_source[i];
        MPI_Send(&local_task_s, 1, MPI_INT, 0, 102, MPI_COMM_WORLD);
        for (int j = 0; j < nsite; j++) {
          msg_Ut.get()[j] = Ut[i][j];
        }
        MPI_Send(msg_Ut.get(), nsite, MPI_DOUBLE, 0, 110, MPI_COMM_WORLD);
      }
    }

  }
}
