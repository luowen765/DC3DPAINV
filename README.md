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

## Repository layout

```
DC3DPAINV/
├── examples/          input files, run scripts, and reference results
├── figures/           figures of the accompanying manuscript
├── src/               C++ sources
├── LICENSE            BSD-3 license
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
| MUMPS | 5.3.5 | optional direct solver (`linear_solver = mumps`) |
| GNU make | 4.x | build |

MFEM must provide MPI, METIS 5, and hypre support. The tested build also
enables MUMPS and the MKL CPARDISO interface, which supply the libraries used
by the optional `mumps` solver and the PARDISO-based GN solver. A typical MFEM
configuration is:

```text
MFEM_USE_MPI          = YES
MFEM_USE_METIS_5      = YES
MFEM_USE_MUMPS        = YES
MFEM_USE_MKL_CPARDISO = YES
```

MFEM, hypre, METIS, Eigen, and MUMPS are available at:

- <https://github.com/mfem/mfem>
- <https://github.com/hypre-space/hypre>
- <https://github.com/KarypisLab/METIS>
- <https://gitlab.com/libeigen/eigen>
- <http://mumps.enseeiht.fr/>

GSLIB is not required. The input reader currently accepts only
`find_points_by = FindPoints`.

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
`mpirun -np ${NP} ../DC3DInv inversion_parameters.txt`. The default process
counts are those used in the paper (16 for ex1, 32 for ex2, and 4 for ex3); set
`NP` explicitly on a different machine:

```bash
NP=8 bash run.sh
```

The equivalent manual command is:

```bash
mpirun -np 4 ../DC3DInv inversion_parameters.txt
```

## Examples

| Directory | Paper section | Case |
| --- | --- | --- |
| `examples/ex1/adinv` | 3.1, Figs. 3--5, Table 1 | undulating topography, adaptive-mesh L-BFGS |
| `examples/ex1/inv` | 3.1 | undulating topography, fixed-mesh L-BFGS with topography |
| `examples/ex1/not_consider_topo` | 3.1 | flat-surface control, fixed-mesh L-BFGS |
| `examples/ex2/inv` | 3.2, Figs. 6--11, Table 2 | triaxial anisotropy, fixed-mesh L-BFGS |
| `examples/ex2/adinv` | 3.2 | triaxial anisotropy, adaptive-mesh L-BFGS |
| `examples/ex2/iso` | 3.2 | isotropic control for the triaxial data |
| `examples/ex3/iso` | 3.3, Figs. 12--15, Table 3 | mining-tunnel field data, isotropic GN |
| `examples/ex3/vti` | 3.3 | mining-tunnel field data, VTI GN |
| `examples/ex3/pyGIMLi_refer` | 3.3 | pyGIMLi reference result for the tunnel data |

All synthetic examples use a homogeneous 100 Ω·m initial and reference model.
The field-data examples use the median observed apparent resistivity,
approximately 257 Ω·m.

## Input files

### `inversion_parameters.txt`

The control file contains exactly **45 non-comment entries**, one value per
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
| 25 | `inv_amr_type` | 0 = model gradient, 1 = JTV, 2 = both |
| 26 | `inv_amr_ratio_model` | refinement fraction for indicator 0 |
| 27 | `inv_amr_ratio_data` | refinement fraction for indicator 1 |
| 28 | `h_to_surface` | reserved; currently unused, set to 0 |

**Forward model, solver, and parallelism**

| # | Entry | Meaning |
| --- | --- | --- |
| 29 | `model_parameters_file` | initial and reference conductivity model |
| 30 | `mesh_file` | initial forward mesh |
| 31 | `relative_misfit_for_up` | refine the forward mesh when the relative misfit drop exceeds this value |
| 32 | `relative_misfit_for_down` | refine the forward mesh when the relative misfit drop is below this value |
| 33 | `N_refine_receivers_local_for` | local refinement levels around measurement electrodes on the forward mesh |
| 34 | `max_amr` | maximum number of forward-mesh AMR iterations; 0 disables AMR |
| 35 | `amr_type` | 0 = ratio, 1 = relative error |
| 36 | `amr_ratio` | ratio or threshold for the forward AMR criterion |
| 37 | `max_dofs` | maximum number of forward degrees of freedom |
| 38 | `linear_solver` | `amg`, `ilu`, or `mumps` |
| 39 | `find_points_by` | point-location method; currently only `FindPoints` |
| 40 | `pcg_maxit` | maximum PCG iterations for forward and adjoint solves |
| 41 | `pcg_primal_tol` | PCG tolerance for primal solves |
| 42 | `pcg_dual_tol` | PCG tolerance for dual (adjoint) solves |
| 43 | `error_esti_way` | `jn` or `zz_patch` |
| 44 | `integration_order` | quadrature order: 1, 2, or 3 |
| 45 | `n_group` | number of source groups; 1 uses domain decomposition only |

### `sigma.file`

The conductivity model file must contain **no comments**. The format is:

```text
n_regions n_inversion_regions
marker sigma_xx sigma_yy sigma_zz angle_x angle_y angle_z
...
marker_1 marker_2 ... marker_n
```

Conductivities are in S/m. The rotation angles are accepted by the reader but
must currently be zero; the inversion code stops if a nonzero angle is found.
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

## Reproducing the paper

The paper reports the following reference runs:

| Example | MPI processes | Key result |
| --- | --- | --- |
| ex1 fixed mesh | 16 | 1,323,057 inversion elements, 38 iterations, 785.7 s |
| ex1 adaptive mesh | 16 | 296,486 → 627,684 elements, 27 iterations, 306.3 s |
| ex2 fixed mesh | 32 | 248,629 elements / 745,887 unknowns, 61 iterations, 500.0 s |
| ex2 adaptive mesh | 32 | 22,469 → 62,891 elements / 188,673 unknowns, 43 iterations, 112.5 s |
| ex3 isotropic GN | 4 | 10,246 → 21,528 elements, 6 updates, final RMS 1.0572, 140.2 s |

The first two examples require an HPC node. The field-data example runs on a
laptop with four MPI processes. The `examples/ex3/iso` directory retains the
original reference run log; a verification run with the released code, using
the same four-process configuration, reproduced the final weighted RMS of
1.0572.

## Configuration notes

- `n_group = 1` assigns all MPI processes to one domain-decomposition group.
  Values greater than one split the sources across process groups and require
  `MPI_COMM_WORLD` size to be at least `n_group`.
- `h_to_surface` is retained for input compatibility and is currently unused.
- Only `find_points_by = FindPoints` is implemented.
- Only `error_esti_way = jn` and `zz_patch` are implemented.
- The `mumps` solver requires MFEM built with `MFEM_USE_MUMPS=YES`; the `amg`
  solver requires hypre. GN additionally uses Eigen's PARDISO interface and
  therefore requires MKL.
- The data error column defines the weights
  `W_d(i,i) = 1 / (error_i + noise_floor)`; `noise_floor` is in the same units
  as the data.

## License

DC3DPAINV is released under the BSD-3 license; see `LICENSE`. Portions of the
code are derived from DC3DPAFEM and from the MFEM library, both distributed
under the BSD-3 license. The original authors are acknowledged in the source
headers.

## Authors

- Lewen Qiu, Research Institute of Exploration and Development, PetroChina
  Xinjiang Oilfield Company, Karamay, China. E-mail: luo_wen@csu.edu.cn
- Huang Chen, School of Geosciences and Info-Physics, Central South University,
  Changsha, China. E-mail: Huang-chen@outlook.com

The forward-modeling modules retain the contributions of Zhengguang Liu,
Hongbo Yao, and Jingtian Tang, as noted in the source headers.
