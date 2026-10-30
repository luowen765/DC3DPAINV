# DC3DPAINV user guide

DC3DPAINV is a parallel adaptive finite-element inversion package for 3-D direct
current (DC) resistivity imaging with surface topography and electrical
anisotropy. It extends the secondary virtual potential formulation of
DC3DPAFEM to the inverse problem and provides a common framework for
Gauss--Newton (GN) and limited-memory BFGS (L-BFGS) inversion. Forward and
inversion meshes are decoupled: the forward mesh is refined to control
response accuracy, while the inversion mesh is refined where the data-misfit
gradient requires additional parameters.

The accompanying manuscript describes the method and the three numerical
examples. If you use this code, please cite the paper.

The code is hosted at <https://github.com/luowen765/DC3DPAINV>.

## Repository layout

```
DC3DPAINV/
├── examples/          input files, run scripts, and reference results
├── figures/           figures of the accompanying manuscript
├── src/               C++ sources
├── LICENSE            BSD-3-Clause license and third-party notices
└── README.md          this file
```

The third-party finite-element and solver libraries are not bundled. They must
be installed separately; see [Prerequisites](#prerequisites).

## Prerequisites

DC3DPAINV is developed in C++11 for Linux and uses MPI. The tested
configuration is:

| Component | Tested version | Purpose |
| --- | --- | --- |
| Intel oneAPI (MPI, MKL) | 2021.3 | MPI compiler `mpiicpc`, BLAS/LAPACK/PARDISO |
| MFEM | 4.6 | tetrahedral finite elements, MPI, AMG |
| hypre | 2.19.0 | BoomerAMG preconditioner |
| METIS | 5.1.0 | mesh partitioning |
| Eigen | 3.4.0 | dense/sparse linear algebra and PARDISO interface |
| MUMPS | 5.3.5 | direct solver (`linear_solver = mumps`) |
| GNU make | 4.x | build |

MFEM must provide MPI, METIS 5, hypre, and MUMPS support. The MUMPS interface
is compiled unconditionally, so `MFEM_USE_MUMPS=YES` is required even when
`linear_solver = amg`; GN additionally requires the MKL PARDISO interface used
by Eigen. The tested build enables all of these:

```text
MFEM_USE_MPI          = YES
MFEM_USE_METIS_5      = YES
MFEM_USE_MUMPS        = YES
MFEM_USE_MKL_PARDISO  = YES
```

MFEM, hypre, METIS, Eigen, and MUMPS are available at:

- <https://github.com/mfem/mfem>
- <https://github.com/hypre-space/hypre>
- <https://github.com/KarypisLab/METIS>
- <https://gitlab.com/libeigen/eigen>
- <http://mumps.enseeiht.fr/>

GSLIB is not required.

## Building

The example makefiles locate MFEM through the `MFEM_DIR` environment variable
and Eigen through `EIGEN_INCLUDE`. Set them once:

```bash
export MFEM_DIR=/path/to/mfem-4.6
export EIGEN_INCLUDE=-I/path/to/eigen-3.4.0
```

Then build from any example directory:

```bash
cd examples/ex3
make DC3DInv
```

The object files are written to `src/`; the executable `DC3DInv` is written to
the example directory. To remove the build products:

```bash
make clean
```

`MFEM_DIR` must point to an MFEM build directory that contains
`config/config.mk`. The makefiles stop with a clear error message otherwise.

## Running an example

Each case directory contains a control file, observation data, a conductivity
model, a mesh, a `run.sh`, and reference output. For example:

```bash
cd examples/ex3/iso
NP=4 bash run.sh
```

`run.sh` builds the executable in the parent directory and starts
`mpirun -np ${NP} ../DC3DInv inversion_parameters.txt`. It defaults to the
process count used for that case in the paper, as listed in
[Examples](#examples); set `NP` explicitly on a different machine:

```bash
NP=8 bash run.sh
```

The equivalent manual command is:

```bash
mpirun -np 4 ../DC3DInv inversion_parameters.txt
```

## Examples

| Directory | Paper section | Case | Processes | Reference result |
| --- | --- | --- | --- | --- |
| `examples/ex1/adinv` | 3.1, Figs. 3--5, Table 2 | undulating topography, adaptive-mesh L-BFGS | 16 | 296,486 → 627,684 elements, 27 iterations, 306.3 s |
| `examples/ex1/inv` | 3.1 | undulating topography, fixed-mesh L-BFGS with topography | 16 | 1,323,057 inversion elements, 38 iterations, 785.7 s |
| `examples/ex1/not_consider_topo` | 3.1 | flat-surface control, fixed-mesh L-BFGS | 16 | — |
| `examples/ex2/inv` | 3.2, Figs. 6--11, Table 3 | triaxial anisotropy, fixed-mesh L-BFGS | 32 | 248,629 elements / 745,887 unknowns, 61 iterations, 500.0 s |
| `examples/ex2/adinv` | 3.2 | triaxial anisotropy, adaptive-mesh L-BFGS | 32 | 22,469 → 62,891 elements / 188,673 unknowns, 43 iterations, 112.5 s |
| `examples/ex2/iso` | 3.2 | isotropic control for the triaxial data | 32 | — |
| `examples/ex3/iso` | 3.3, Figs. 12--15, Table 4 | mining-tunnel field data, isotropic GN | 4 | 10,246 → 21,528 elements, 6 updates, final RMS 1.0572, 140.2 s |
| `examples/ex3/vti` | 3.3 | mining-tunnel field data, VTI GN | 4 | — |
| `examples/ex3/pyGIMLi_refer` | 3.3 | pyGIMLi reference result for the tunnel data | 4 | — |

The "Reference result" column lists the outcome reported in the manuscript:
inversion elements, forward unknowns where quoted, iterations, and wall-clock
time. The timings were obtained on the platform used for the manuscript and
are indicative only. The first two examples require an HPC node, whereas the
field-data example runs on a laptop with four MPI processes.

Reference output files, where provided, are kept in each case directory
(`Inversion_results/` and `run.log`). The `examples/ex3/iso` directory retains
the original run log; a verification run with the released code, using the same
four-process configuration, reproduced the final weighted RMS of 1.0572.

All synthetic examples use a homogeneous 100 Ω·m initial and reference model.
The field-data examples use the median observed apparent resistivity,
approximately 257 Ω·m.

## Input files

### `inversion_parameters.txt`

The control file contains exactly **40 non-comment entries**, one value per
line. Blank lines and text after `#` are ignored. The entry order is fixed.

**Inversion control**

| # | Entry | Meaning |
| --- | --- | --- |
| 1 | `if_anisotropy` | 0 = isotropic, 1 = VTI, 2 = triaxial |
| 2 | `max_iter` | maximum number of inversion iterations |
| 3 | `N_M` | number of correction pairs stored by L-BFGS |
| 4 | `N_line_search` | maximum trial steps in the inexact line search |
| 5 | `obs_data_file` | observation data file |
| 6 | `inv_data_type` | 0 = potential difference (dU/I), 1 = apparent resistivity |
| 7 | `stop_misfit` | stopping threshold for the weighted RMS misfit |
| 8 | `noise_floor` | absolute noise floor added to the data errors |
| 9 | `init_lambda` | initial regularization parameter |
| 10 | `atte_lambda` | lambda attenuation factor |
| 11 | `lambda_atte_thre` | reduce lambda when the relative misfit improvement is below this value |
| 12 | `lambda_floor` | minimum lambda |
| 13 | `lambda_keep` | minimum number of iterations at each lambda |
| 14 | `min_cond` | lower conductivity bound (S/m) |
| 15 | `max_cond` | upper conductivity bound (S/m) |
| 16 | `cp` | exponent of the bounded model transformation |
| 17 | `GN_PCG_max_iter` | maximum PCG iterations for the GN system |
| 18 | `GN_PCG_TOL` | PCG tolerance for the GN system |
| 19 | `inv_algorithm` | `gn` or `lbfgs` |

**Inversion-mesh adaptivity**

| # | Entry | Meaning |
| --- | --- | --- |
| 20 | `relative_misfit_inv_up` | refine when the relative misfit drop exceeds this value |
| 21 | `relative_misfit_inv_down` | refine when the relative misfit drop is below this value |
| 22 | `N_refine_electrodes_local_inv` | local refinement levels around electrodes on the initial inversion mesh |
| 23 | `adInv_max_iter` | maximum number of inversion-mesh refinements; 0 keeps the mesh fixed |
| 24 | `invMesh_keep` | minimum number of iterations between refinements |
| 25 | `inv_amr_ratio` | refinement fraction for the data-misfit gradient indicator |

**Forward model, solver, and parallelism**

| # | Entry | Meaning |
| --- | --- | --- |
| 26 | `model_parameters_file` | initial and reference conductivity model |
| 27 | `mesh_file` | initial forward mesh |
| 28 | `relative_misfit_for_up` | refine the forward mesh when the relative misfit drop exceeds this value |
| 29 | `relative_misfit_for_down` | refine the forward mesh when the relative misfit drop is below this value |
| 30 | `N_refine_receivers_local_for` | local refinement levels around measurement electrodes on the forward mesh |
| 31 | `max_amr` | maximum number of forward-mesh AMR iterations; 0 disables AMR |
| 32 | `amr_type` | 0 = ratio, 1 = relative error |
| 33 | `amr_ratio` | ratio or threshold for the forward AMR criterion |
| 34 | `max_dofs` | maximum number of forward degrees of freedom |
| 35 | `linear_solver` | `amg` or `mumps` |
| 36 | `pcg_maxit` | maximum PCG iterations for forward and adjoint solves |
| 37 | `pcg_primal_tol` | PCG tolerance for primal solves |
| 38 | `pcg_dual_tol` | PCG tolerance for dual (adjoint) solves |
| 39 | `integration_order` | quadrature order: 1, 2, or 3 |
| 40 | `n_group` | number of source groups; 1 uses domain decomposition only |

A value of `n_group` greater than one splits the sources across process groups
and requires the `MPI_COMM_WORLD` size to be at least `n_group`; `n_group = 1`
assigns all processes to a single domain-decomposition group. The data error
column defines the inversion weights
`W_d(i,i) = 1 / (error_i + noise_floor)`, where `noise_floor` is in the same
units as the data.

### `sigma.file`

The conductivity model file must contain **no comments**. The format is:

```text
n_regions n_inversion_regions
marker sigma_xx sigma_yy sigma_zz angle_x angle_y angle_z
...
marker_1 marker_2 ... marker_n
```

Conductivities are in S/m. The principal axes are aligned with the survey
coordinates, so the three rotation angles must be zero; the inversion stops if
a nonzero angle is found.
All supplied examples use a single inversion region with marker `19210723`.

### `observation_data.txt`

The first line gives the number of electrodes. The next lines give electrode
coordinates and a surface flag:

```text
x y z is_on_surface
```

The following line gives the number of measurements and the survey mode
(`11`, `12`, `21`, or `22`). Each measurement line contains the electrode
indices followed by the observed values and errors. For a dipole--dipole
survey the columns are:

```text
A B M N dU/I error_U rho_a error_rho
```

For other survey modes the number of source and potential electrodes changes
accordingly. Invalid data can be marked with an error of `1.0e12`; these rows
are excluded from the inversion.

### Meshes

The meshes are MFEM unstructured tetrahedral `.msh` files. The inversion mesh
and the forward mesh share a nested tetrahedral hierarchy; the preparation of
the initial inversion mesh is described in the manuscript.

## Output files

For each iteration the program writes, under `Inversion_results/<iteration>/`:

- `cond_inv_mesh.vtk` — recovered conductivity model;
- `JTV_inv_mesh.vtk` — data-misfit gradient (JTV) indicator;
- `Modeling_data.txt` — predicted apparent resistivities.

`Inversion_results/Output_Inversion_parameter` summarizes the iteration,
regularization factor, weighted RMS, model roughness, objective function,
mesh-refinement flags, and the numbers of forward unknowns and inversion
elements. A copy of the final predicted data and the summary table is also
kept in each case directory where available. `run.log` records the complete
run.

## License

DC3DPAINV is released under the BSD-3-Clause license; see `LICENSE`, which also
contains the third-party and derived-code notices. The original authors are
additionally acknowledged in the source headers.

The reference outputs in `examples/ex3/pyGIMLi_refer` were generated with
pyGIMLi (Apache-2.0). The mining-tunnel field data are described by Günther
(2004) and are included for reproducibility with attribution.

## Authors

- Lewen Qiu, Research Institute of Exploration and Development, PetroChina
  Xinjiang Oilfield Company, Karamay, China. E-mail: luo_wen@csu.edu.cn
- Huang Chen, School of Resources and Safety Engineering, Chongqing University,
  Chongqing 400044, China; State Key Laboratory of Coal Mine Disaster Dynamics
  and Control, Chongqing University, Chongqing 400044, China.
  E-mail: chenhuang@cqu.edu.cn

The forward-modeling modules retain the contributions of Zhengguang Liu,
Hongbo Yao, and Jingtian Tang, as noted in the source headers.
