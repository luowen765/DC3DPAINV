/*
 * @Description: Main program of DC3DPAINV: reads the inversion control file,
 *   initializes the parallel inversion, and dispatches the selected
 *   optimization algorithm.
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
#include "lbfgs.h"

int main(int argc, char *argv[]) {

  MPI_Init(&argc, &argv);
  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  extern std::vector<double> memoryUsed;
  extern double tempMemory;

  tempMemory = GetCurrentMemoryUsage();

  memoryUsed.push_back(tempMemory);

  double main_tic, main_toc, iter_tic, iter_toc;

  if (world_rank == 0) {
    main_tic = MPI_Wtime();
  }

  if (argc < 2) {
    if (world_rank == 0) {
      std::cerr << "Usage: " << argv[0] << " input_parameters_filename\n";
    }

    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }

  ParaHandler inv_para(argv[1]);

  inv_para.dual_parallel_resource_allocation(world_rank, world_size);

  std::unique_ptr<InvBase> dcinv;
  std::string inv_algorithm = inv_para.inv_algorithm;
  if (inv_algorithm == "gn") {

    dcinv.reset(new GN(inv_para, world_rank, world_size

                       ));
    if (world_rank == 0)
      std::cout << "  Execute Gauss Newton algorithm...\n";
  } else if (inv_algorithm == "lbfgs") {

    dcinv.reset(new LBFGS(inv_para, world_rank, world_size

                          ));

    if (world_rank == 0)
      std::cout << "  Execute LBFGS algorithm...\n";
  } else {
    if (world_rank == 0) {
      std::cerr << "  Unknown inversion algorithm: " << inv_algorithm << "\n";
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int end_loop = 0;

  std::string inv_reslut_folder = "Inversion_results";
  if (world_rank == 0) {
    int create_inversion_files = mkdir(inv_reslut_folder.c_str(), S_IRWXU);
  }

  for (int k = 1; k <= inv_para.max_iter; k++) {
    std::string iter_name = inv_reslut_folder + "/" + std::to_string(k);
    if (world_rank == 0) {
      iter_tic = MPI_Wtime();
      std::cout << "\n==========inversion iter: " << k << std::endl;
      int creat_file = mkdir(iter_name.c_str(), S_IRWXU);
    }

    if (k == 1) {

      dcinv->resetInvParas(k);

      dcinv->forward_modeling(k);

    } else {

      dcinv->StrongWolfeConditionsLineSearch(k);

    }

    if (world_rank == 0) {

      dcinv->save_predicted_data(iter_name, inv_para.obs_data_file);

      dcinv->save_cond_mesh(iter_name);
      dcinv->save_JTV_mesh(iter_name);

      dcinv->update_common_inv_para(k);

      if (dcinv->misfit[k - 1] < inv_para.stop_misfit) {
        std::cout
            << "  The data misfit reached the pre-set value, stop iteration."
            << std::endl;

        end_loop = 1;
      }

      if (k > 4) {
        double down_0 = (dcinv->misfit[k - 5] - dcinv->misfit[k - 4]) /
                        dcinv->misfit[k - 5];
        double down_1 = (dcinv->misfit[k - 4] - dcinv->misfit[k - 3]) /
                        dcinv->misfit[k - 4];
        double down_2 = (dcinv->misfit[k - 3] - dcinv->misfit[k - 2]) /
                        dcinv->misfit[k - 3];
        double down_3 = (dcinv->misfit[k - 2] - dcinv->misfit[k - 1]) /
                        dcinv->misfit[k - 2];
        if (down_0 < 0.0001 && down_1 < 0.0001 && down_2 < 0.0001 &&
            down_3 < 0.0001) {

          std::cout
              << "  The data misfit does not decrease four times continuously, "
                 "stop iteration."
              << std::endl;
          end_loop = 1;
        }
      }

      if (k == (inv_para.max_iter)) {
        std::cout
            << "  The pre-set maxmum iterations is reached, stop iteration."
            << std::endl;

        end_loop = 1;
      }

      for (int p = 1; p < world_size; p++) {
        MPI_Send(&end_loop, 1, MPI_INT, p, 666, MPI_COMM_WORLD);
        MPI_Send(&dcinv->if_amr_forward, 1, MPI_INT, p, 888, MPI_COMM_WORLD);
        MPI_Send(&dcinv->if_amr_inv, 1, MPI_INT, p, 999, MPI_COMM_WORLD);
      }
    } else {
      MPI_Recv(&end_loop, 1, MPI_INT, 0, 666, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Recv(&dcinv->if_amr_forward, 1, MPI_INT, 0, 888, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Recv(&dcinv->if_amr_inv, 1, MPI_INT, 0, 999, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }

    if (end_loop == 1) {

      myassert(memoryUsed.size() > 0);
      double max_element_error =
          *std::max_element(memoryUsed.begin(), memoryUsed.end());
      if (world_rank == 0) {
        std::cout << "myid " << world_rank
                  << " enter into initial memory: " << memoryUsed[0]
                  << std::endl;
        std::cout << "myid " << world_rank << " need memory in this code: "
                  << max_element_error - memoryUsed[0] << std::endl;
      }

      if (world_rank == 0) {

        dcinv->output_inversion_parameters(inv_reslut_folder);
        std::cout << "The number of cpus is: " << world_size << "\n";

        main_toc = MPI_Wtime();
        std::cout << "The total runtime for this inversion is: ";
        print_time(main_toc - main_tic);

        std::cout << "The total runtime for all the forward is: "
                  << dcinv->count_forward_time << std::endl;
      }
      MPI_Finalize();
      return 0;
    } else {

      if (dcinv->if_amr_inv == 1) {

        dcinv->refine_inv_mesh(inv_reslut_folder);
        dcinv->resetInvParas(k);
        dcinv->N_iter_adlbfgs = k - 1;

        dcinv->forward_modeling(k);
        if (dcinv->pmesh_amr_flag == true) {
          dcinv->pmesh.reset();
          dcinv->pmesh_amr_flag = false;
        }

      }

      if (world_rank == 0) {

        double tic = MPI_Wtime();

        if (dcinv->inv_algorithm == "lbfgs") {

          dcinv->update_model_direction(k - dcinv->N_iter_adlbfgs);
        } else {
          dcinv->update_model_direction(k);

        }

        double toc = MPI_Wtime();
        std::cout << "    Run update_model_direction time: " << toc - tic
                  << std::endl;
      }
    }

    if (world_rank == 0) {
      iter_toc = MPI_Wtime();
      std::cout << "  The runtime for this iteration is: ";
      print_time(iter_toc - iter_tic);
    }

  }

}
