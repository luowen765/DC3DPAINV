/*
 * @Description: Implementations of the MFEM extensions declared in mfemplus.h.
 * @Author: Lewen Qiu (original DC3DPAFEM module); portions adapted from MFEM
 *   (BSD-3); adapted for DC3DPAINV by Lewen Qiu and Huang Chen
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

#include <iomanip>
#include <iostream>

#include "mfemplus.h"

using namespace std;

double B(Point_double &x, Point_double &x_c, Matrix3D &tet_sigma) {
  Point_double rc = x - x_c;
  RowVector3D gg = RowVector3D::Zero(1, 3);
  Vector3D hh = Vector3D::Zero(3, 1);
  for (int i = 0; i < 3; i++) {
    gg[i] = rc(i);
    hh[i] = rc(i);
  }
  Matrix3D rho = tet_sigma.inverse();
  double B = 0;

  B = (gg * rho * hh);
  return B;
}

double beta(double B, Point_double &x, Point_double &x_c, Point_double &n) {
  Point_double rc = x - x_c;
  double temp = (rc * n / B);
  return temp;
}

double Uis(Matrix3D &sigma, Point_double &x, Point_double &x_s,
           int if_s_on_surface) {
  Matrix3D a = sigma.inverse();
  double b = a.determinant();
  double B0 = B(x, x_s, sigma);
  double temp = 0.0;

  if (if_s_on_surface == 1) {
    temp = 1.0 * sqrt(b) * (1.0 / sqrt(B0)) / (2.0 * EM::PI);
  } else {
    temp = 1.0 * sqrt(b) * (1.0 / sqrt(B0)) / (4.0 * EM::PI);
  }

  return temp;
}

Point_double gradient_Uis(Matrix3D &sigma, Point_double &x, Point_double &x_s,
                          int if_s_on_surface) {
  Matrix3D a = sigma.inverse();
  double b = a.determinant();
  double C = 0.0;
  if (if_s_on_surface == 1) {
    C = -1.0 * sqrt(b) / (2.0 * EM::PI);
  } else {
    C = -1.0 * sqrt(b) / (4.0 * EM::PI);
  }

  double B0 = B(x, x_s, sigma);
  Vector3D d1 = Vector3D::Zero(3, 1);
  for (int i = 0; i < 3; i++) {
    d1[i] = (x - x_s)(i);
  }
  Vector3D dd = C * a * (d1 / pow(B0, 1.5));
  Point_double temp;
  for (int j = 0; j < 3; j++)
    temp(j) = dd[j];

  return temp;
}

SolversPlus::SolversPlus() : CGSolver() {}
SolversPlus::SolversPlus(MPI_Comm comm_) : CGSolver(comm_) {}
void SolversPlus::Mult(const Vector &b, Vector &x) {
  int i;
  double r0, den, nom, nom0, betanom, alpha, beta;
  double initial_r;

  x.UseDevice(true);
  if (iterative_mode) {
    oper->Mult(x, r);
    subtract(b, r, r);
  } else {
    r = b;
    x = 0.0;
  }

  if (prec) {
    prec->Mult(r, z);
    d = z;
  } else {
    d = r;
  }
  nom0 = nom = Dot(d, r);

  MFEM_ASSERT(IsFinite(nom), "nom = " << nom);
  if (print_level == 1 || print_level == 3) {

    initial_r = sqrt(nom);

    mfem::out << "   Iteration : " << setw(3) << 0 << "  ||r|| = \t"
              << sqrt(nom) / initial_r << (print_level == 3 ? " ...\n" : "\n");
  }
  Monitor(0, nom, r, x);

  if (nom < 0.0) {
    if (print_level >= 0) {
      mfem::out
          << "PCG: The preconditioner is not positive definite. (Br, r) = "
          << nom << '\n';
    }
    converged = 0;
    final_iter = 0;
    final_norm = nom;
    return;
  }
  r0 = std::max(nom * rel_tol * rel_tol, abs_tol * abs_tol);

  if (nom <= r0) {
    converged = 1;
    final_iter = 0;
    final_norm = sqrt(nom);
    return;
  }

  oper->Mult(d, z);
  den = Dot(z, d);
  MFEM_ASSERT(IsFinite(den), "den = " << den);
  if (den <= 0.0) {
    if (Dot(d, d) > 0.0 && print_level >= 0) {
      mfem::out << "PCG: The operator is not positive definite. (Ad, d) = "
                << den << '\n';
    }
    if (den == 0.0) {
      converged = 0;
      final_iter = 0;
      final_norm = sqrt(nom);
      return;
    }
  }

  converged = 0;
  final_iter = max_iter;
  for (i = 1; true;) {
    alpha = nom / den;
    add(x, alpha, d, x);
    add(r, -alpha, z, r);

    if (prec) {
      prec->Mult(r, z);
      betanom = Dot(r, z);
    } else {
      betanom = Dot(r, r);
    }
    MFEM_ASSERT(IsFinite(betanom), "betanom = " << betanom);
    if (betanom < 0.0) {
      if (print_level >= 0) {
        mfem::out
            << "PCG: The preconditioner is not positive definite. (Br, r) = "
            << betanom << '\n';
      }
      converged = 0;
      final_iter = i;
      break;
    }

    if (print_level == 1) {

      mfem::out << "   Iteration : " << setw(3) << i << "  ||r|| = \t"
                << sqrt(betanom) / initial_r << '\n';
    }

    Monitor(i, betanom, r, x);

    if (betanom <= r0) {
      if (print_level == 2) {
        mfem::out << "Number of PCG iterations: " << i << '\n';
      } else if (print_level == 3) {
        mfem::out << "   Iteration : " << setw(3) << i
                  << "  (B r, r) = " << betanom << '\n';
      }
      converged = 1;
      final_iter = i;
      break;
    }

    if (++i > max_iter) {
      break;
    }

    beta = betanom / nom;
    if (prec) {
      add(z, beta, d, d);
    } else {
      add(r, beta, d, d);
    }
    oper->Mult(d, z);
    den = Dot(d, z);
    MFEM_ASSERT(IsFinite(den), "den = " << den);
    if (den <= 0.0) {
      if (Dot(d, d) > 0.0 && print_level >= 0) {
        mfem::out << "PCG: The operator is not positive definite. (Ad, d) = "
                  << den << '\n';
      }
      if (den == 0.0) {
        final_iter = i;
        break;
      }
    }
    nom = betanom;
  }
  if (print_level >= 0 && !converged) {
    if (print_level != 1) {
      if (print_level != 3) {
        mfem::out << "   Iteration : " << setw(3) << 0
                  << "  (B r, r) = " << nom0 << " ...\n";
      }

      mfem::out << "   Iteration : " << setw(3) << final_iter
                << "  (B r, r) = " << betanom << '\n';
    }
    mfem::out << "PCG: No convergence!" << '\n';
  }
  if (print_level >= 1 || (print_level >= 0 && !converged)) {
    mfem::out << "Average reduction factor = "
              << pow(betanom / nom0, 0.5 / final_iter) << '\n';
  }
  final_norm = sqrt(betanom);

  Monitor(final_iter, final_norm, r, x, true);
}

Meshplus::Meshplus() : Mesh() {}
Meshplus::Meshplus(ParMesh *pmesh_) : Mesh(*pmesh_) {}
Meshplus::Meshplus(Mesh *mesh_) : Mesh(*mesh_) {}

void Meshplus::PrintVTK_eta(std::ostream &out, Vector &other_attri,
                            int n_value) {
  out << "# vtk DataFile Version 3.0\n"
         "Generated by MFEM\n"
         "ASCII\n"
         "DATASET UNSTRUCTURED_GRID\n";

  if (Nodes == NULL) {
    out << "POINTS " << NumOfVertices << " double\n";
    for (int i = 0; i < NumOfVertices; i++) {
      out << vertices[i](0);
      int j;
      for (j = 1; j < spaceDim; j++) {
        out << ' ' << vertices[i](j);
      }
      for (; j < 3; j++) {
        out << ' ' << 0.0;
      }
      out << '\n';
    }
  } else {
    Array<int> vdofs(3);
    out << "POINTS " << Nodes->FESpace()->GetNDofs() << " double\n";
    for (int i = 0; i < Nodes->FESpace()->GetNDofs(); i++) {
      vdofs.SetSize(1);
      vdofs[0] = i;
      Nodes->FESpace()->DofsToVDofs(vdofs);
      out << (*Nodes)(vdofs[0]);
      int j;
      for (j = 1; j < spaceDim; j++) {
        out << ' ' << (*Nodes)(vdofs[j]);
      }
      for (; j < 3; j++) {
        out << ' ' << 0.0;
      }
      out << '\n';
    }
  }

  int order = -1;
  if (Nodes == NULL) {
    int size = 0;
    for (int i = 0; i < NumOfElements; i++) {
      size += elements[i]->GetNVertices() + 1;
    }
    out << "CELLS " << NumOfElements << ' ' << size << '\n';
    for (int i = 0; i < NumOfElements; i++) {
      const int *v = elements[i]->GetVertices();
      const int nv = elements[i]->GetNVertices();
      out << nv;
      for (int j = 0; j < nv; j++) {
        out << ' ' << v[j];
      }
      out << '\n';
    }
    order = 1;
  } else {
    Array<int> dofs;
    int size = 0;
    for (int i = 0; i < NumOfElements; i++) {
      Nodes->FESpace()->GetElementDofs(i, dofs);
      MFEM_ASSERT(Dim != 0 || dofs.Size() == 1,
                  "Point meshes should have a single dof per element");
      size += dofs.Size() + 1;
    }
    out << "CELLS " << NumOfElements << ' ' << size << '\n';
    const char *fec_name = Nodes->FESpace()->FEColl()->Name();

    if (!strcmp(fec_name, "Linear") || !strcmp(fec_name, "H1_0D_P1") ||
        !strcmp(fec_name, "H1_1D_P1") || !strcmp(fec_name, "H1_2D_P1") ||
        !strcmp(fec_name, "H1_3D_P1")) {
      order = 1;
    } else if (!strcmp(fec_name, "Quadratic") ||
               !strcmp(fec_name, "H1_1D_P2") || !strcmp(fec_name, "H1_2D_P2") ||
               !strcmp(fec_name, "H1_3D_P2")) {
      order = 2;
    }
    if (order == -1) {
      mfem::err << "Mesh::PrintVTK : can not save '" << fec_name
                << "' elements!" << std::endl;
      mfem_error();
    }
    for (int i = 0; i < NumOfElements; i++) {
      Nodes->FESpace()->GetElementDofs(i, dofs);
      out << dofs.Size();
      if (order == 1) {
        for (int j = 0; j < dofs.Size(); j++) {
          out << ' ' << dofs[j];
        }
      } else if (order == 2) {
        const int *vtk_mfem;
        switch (elements[i]->GetGeometryType()) {
        case Geometry::SEGMENT:
        case Geometry::TRIANGLE:
        case Geometry::SQUARE:
          vtk_mfem = vtk_quadratic_hex;
          break;
        case Geometry::TETRAHEDRON:
          vtk_mfem = vtk_quadratic_tet;
          break;
        case Geometry::PRISM:
          vtk_mfem = vtk_quadratic_wedge;
          break;
        case Geometry::CUBE:
        default:
          vtk_mfem = vtk_quadratic_hex;
          break;
        }
        for (int j = 0; j < dofs.Size(); j++) {
          out << ' ' << dofs[vtk_mfem[j]];
        }
      }
      out << '\n';
    }
  }

  out << "CELL_TYPES " << NumOfElements << '\n';
  for (int i = 0; i < NumOfElements; i++) {
    int vtk_cell_type = 5;
    Geometry::Type geom_type = GetElement(i)->GetGeometryType();
    if (order == 1) {
      switch (geom_type) {
      case Geometry::POINT:
        vtk_cell_type = 1;
        break;
      case Geometry::SEGMENT:
        vtk_cell_type = 3;
        break;
      case Geometry::TRIANGLE:
        vtk_cell_type = 5;
        break;
      case Geometry::SQUARE:
        vtk_cell_type = 9;
        break;
      case Geometry::TETRAHEDRON:
        vtk_cell_type = 10;
        break;
      case Geometry::CUBE:
        vtk_cell_type = 12;
        break;
      case Geometry::PRISM:
        vtk_cell_type = 13;
        break;
      default:
        break;
      }
    } else if (order == 2) {
      switch (geom_type) {
      case Geometry::SEGMENT:
        vtk_cell_type = 21;
        break;
      case Geometry::TRIANGLE:
        vtk_cell_type = 22;
        break;
      case Geometry::SQUARE:
        vtk_cell_type = 28;
        break;
      case Geometry::TETRAHEDRON:
        vtk_cell_type = 24;
        break;
      case Geometry::CUBE:
        vtk_cell_type = 29;
        break;
      case Geometry::PRISM:
        vtk_cell_type = 32;
        break;
      default:
        break;
      }
    }

    out << vtk_cell_type << '\n';
  }

  out << "CELL_DATA " << NumOfElements << '\n'
      << "SCALARS conductivity double\n"
      << "LOOKUP_TABLE default\n";
  for (int i = 0; i < NumOfElements; i++) {
    out << other_attri[i] << '\n';
  }

  if (n_value == 2) {
    out << "SCALARS resistivity double\n"
        << "LOOKUP_TABLE default\n";
    for (int i = 0; i < NumOfElements; i++) {
      out << 1.0 / other_attri[i] << '\n';
    }
  }

  out.flush();
}

void Meshplus::write_out_node_info_vtk(
    std::ostream &out, Vector &node_data

) {
  out << "# vtk DataFile Version 3.0\n"
         "Generated by MFEM\n"
         "ASCII\n"
         "DATASET UNSTRUCTURED_GRID\n";

  if (Nodes == NULL) {
    out << "POINTS " << NumOfVertices << " double\n";
    for (int i = 0; i < NumOfVertices; i++) {
      out << vertices[i](0);
      int j;
      for (j = 1; j < spaceDim; j++) {
        out << ' ' << vertices[i](j);
      }
      for (; j < 3; j++) {
        out << ' ' << 0.0;
      }
      out << '\n';
    }
  } else {
    Array<int> vdofs(3);
    out << "POINTS " << Nodes->FESpace()->GetNDofs() << " double\n";
    for (int i = 0; i < Nodes->FESpace()->GetNDofs(); i++) {
      vdofs.SetSize(1);
      vdofs[0] = i;
      Nodes->FESpace()->DofsToVDofs(vdofs);
      out << (*Nodes)(vdofs[0]);
      int j;
      for (j = 1; j < spaceDim; j++) {
        out << ' ' << (*Nodes)(vdofs[j]);
      }
      for (; j < 3; j++) {
        out << ' ' << 0.0;
      }
      out << '\n';
    }
  }

  int order = -1;
  if (Nodes == NULL) {
    int size = 0;
    for (int i = 0; i < NumOfElements; i++) {
      size += elements[i]->GetNVertices() + 1;
    }
    out << "CELLS " << NumOfElements << ' ' << size << '\n';
    for (int i = 0; i < NumOfElements; i++) {
      const int *v = elements[i]->GetVertices();
      const int nv = elements[i]->GetNVertices();
      out << nv;
      for (int j = 0; j < nv; j++) {
        out << ' ' << v[j];
      }
      out << '\n';
    }
    order = 1;
  } else {
    Array<int> dofs;
    int size = 0;
    for (int i = 0; i < NumOfElements; i++) {
      Nodes->FESpace()->GetElementDofs(i, dofs);
      MFEM_ASSERT(Dim != 0 || dofs.Size() == 1,
                  "Point meshes should have a single dof per element");
      size += dofs.Size() + 1;
    }
    out << "CELLS " << NumOfElements << ' ' << size << '\n';
    const char *fec_name = Nodes->FESpace()->FEColl()->Name();

    if (!strcmp(fec_name, "Linear") || !strcmp(fec_name, "H1_0D_P1") ||
        !strcmp(fec_name, "H1_1D_P1") || !strcmp(fec_name, "H1_2D_P1") ||
        !strcmp(fec_name, "H1_3D_P1")) {
      order = 1;
    } else if (!strcmp(fec_name, "Quadratic") ||
               !strcmp(fec_name, "H1_1D_P2") || !strcmp(fec_name, "H1_2D_P2") ||
               !strcmp(fec_name, "H1_3D_P2")) {
      order = 2;
    }
    if (order == -1) {
      mfem::err << "Mesh::PrintVTK : can not save '" << fec_name
                << "' elements!" << std::endl;
      mfem_error();
    }
    for (int i = 0; i < NumOfElements; i++) {
      Nodes->FESpace()->GetElementDofs(i, dofs);
      out << dofs.Size();
      if (order == 1) {
        for (int j = 0; j < dofs.Size(); j++) {
          out << ' ' << dofs[j];
        }
      } else if (order == 2) {
        const int *vtk_mfem;
        switch (elements[i]->GetGeometryType()) {
        case Geometry::SEGMENT:
        case Geometry::TRIANGLE:
        case Geometry::SQUARE:
          vtk_mfem = vtk_quadratic_hex;
          break;
        case Geometry::TETRAHEDRON:
          vtk_mfem = vtk_quadratic_tet;
          break;
        case Geometry::PRISM:
          vtk_mfem = vtk_quadratic_wedge;
          break;
        case Geometry::CUBE:
        default:
          vtk_mfem = vtk_quadratic_hex;
          break;
        }
        for (int j = 0; j < dofs.Size(); j++) {
          out << ' ' << dofs[vtk_mfem[j]];
        }
      }
      out << '\n';
    }
  }

  out << "CELL_TYPES " << NumOfElements << '\n';
  for (int i = 0; i < NumOfElements; i++) {
    int vtk_cell_type = 5;
    Geometry::Type geom_type = GetElement(i)->GetGeometryType();
    if (order == 1) {
      switch (geom_type) {
      case Geometry::POINT:
        vtk_cell_type = 1;
        break;
      case Geometry::SEGMENT:
        vtk_cell_type = 3;
        break;
      case Geometry::TRIANGLE:
        vtk_cell_type = 5;
        break;
      case Geometry::SQUARE:
        vtk_cell_type = 9;
        break;
      case Geometry::TETRAHEDRON:
        vtk_cell_type = 10;
        break;
      case Geometry::CUBE:
        vtk_cell_type = 12;
        break;
      case Geometry::PRISM:
        vtk_cell_type = 13;
        break;
      default:
        break;
      }
    } else if (order == 2) {
      switch (geom_type) {
      case Geometry::SEGMENT:
        vtk_cell_type = 21;
        break;
      case Geometry::TRIANGLE:
        vtk_cell_type = 22;
        break;
      case Geometry::SQUARE:
        vtk_cell_type = 28;
        break;
      case Geometry::TETRAHEDRON:
        vtk_cell_type = 24;
        break;
      case Geometry::CUBE:
        vtk_cell_type = 29;
        break;
      case Geometry::PRISM:
        vtk_cell_type = 32;
        break;
      default:
        break;
      }
    }

    out << vtk_cell_type << '\n';
  }

  out << "\nPOINT_DATA\t " << NumOfVertices << '\n'
      << "SCALARS potential double 1\n"
      << "LOOKUP_TABLE default\n";
  for (int i = 0; i < NumOfVertices; i++) {
    out << node_data(i) << '\n';
  }
  out.flush();
}

void LVCoefficient::Eval(DenseMatrix &M, ElementTransformation &T,
                         const IntegrationPoint &ip) {

  M = mat[T.Attribute - 1];

}

double LGamma1Coefficient::Eval(ElementTransformation &T,
                                const IntegrationPoint &ip) {

  Vector normal(3);
  normal = 0.0;

  compute_normal(T, normal);

  Vector x(3);
  x = 0.0;
  T.SetIntPoint(&ip);

  T.Transform(ip, x);

  int tet_id = -1;

  tet_id = get_tetId_bdr(pmesh, T.ElementNo);
  DenseMatrix sigma = mat[tet_id];

  Matrix3D sigma_;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      sigma_(i, j) = sigma(i, j);
    }
  }
  Point_double xp, xi, normal_d;
  for (int j = 0; j < 3; j++) {
    xp(j) = x(j);
    xi(j) = x_c(j);
    normal_d(j) = normal(j);
  }
  double B_value = B(xp, xi, sigma_);
  double mass_coefficient = beta(B_value, xp, xi, normal_d);

  return mass_coefficient;
}

void RVCoefficient::Eval(Vector &V, ElementTransformation &T,
                         const IntegrationPoint &ip) {
  T.SetIntPoint(&ip);
  Vector x(3);
  x = 0.0;

  T.Transform(ip, x);
  V.SetSize(3);
  V = 0.0;

  Matrix3D sigma_0;
  for (int j = 0; j < 3; j++)
    for (int k = 0; k < 3; k++) {
      {
        sigma_0(j, k) = sigma0(j, k);
      }
    }
  Point_double xp, xi;
  for (int j = 0; j < 3; j++) {
    xp(j) = x(j);
    xi(j) = x_i(j);
  }
  Point_double grad_U_i_s = gradient_Uis(sigma_0, xp, xi, if_s_on_surface);

  DenseMatrix temp = sigma0;
  temp -= mat[T.Attribute - 1];

  for (int j = 0; j < 3; j++) {
    for (int k = 0; k < 3; k++) {

      V(k) += temp(j, k) * grad_U_i_s(k);
    }
  }
}

double RGamma0Coefficient::Eval(ElementTransformation &T,
                                const IntegrationPoint &ip) {

  Vector normal(3);
  normal = 0.0;

  compute_normal(T, normal);

  Vector x(3);
  x = 0.0;
  T.SetIntPoint(&ip);

  T.Transform(ip, x);

  Matrix3D sigma_0;
  for (int j = 0; j < 3; j++)
    for (int k = 0; k < 3; k++) {
      {
        sigma_0(j, k) = sigma0(j, k);
      }
    }
  Point_double xp, xi, normal_d;
  for (int j = 0; j < 3; j++) {
    xp(j) = x(j);
    xi(j) = x_i(j);
    normal_d(j) = normal(j);
  }
  double Us = Uis(sigma_0, xp, xi, if_s_on_surface);
  double B0 = B(xp, xi, sigma_0);
  double beta_value = beta(B0, xp, xi, normal_d);
  double mass_coef = beta_value * Us;

  myassert(T.Attribute == 1);

  return mass_coef;
}

double RGamma1Coefficient::Eval(ElementTransformation &T,
                                const IntegrationPoint &ip) {

  myassert(T.Attribute == 2);

  Vector normal(3);
  normal = 0.0;

  compute_normal(T, normal);

  Vector x(3);
  x = 0.0;
  T.SetIntPoint(&ip);

  T.Transform(ip, x);

  int tet_id = -1;

  tet_id = get_tetId_bdr(pmesh, T.ElementNo);
  DenseMatrix sigma = mat[tet_id];

  Matrix3D sigma_;
  for (int j = 0; j < 3; j++)
    for (int k = 0; k < 3; k++) {
      sigma_(j, k) = sigma(j, k);
    }
  Matrix3D sigma_0;
  for (int j = 0; j < 3; j++)
    for (int k = 0; k < 3; k++) {
      {
        sigma_0(j, k) = sigma0(j, k);
      }
    }
  Point_double xp, xi, normal_d;
  for (int j = 0; j < 3; j++) {
    xp(j) = x(j);
    xi(j) = x_i(j);
    normal_d(j) = normal(j);
  }
  double Us = Uis(sigma_0, xp, xi, if_s_on_surface);
  double B_value = B(xp, xi, sigma_);
  double B0 = B(xp, xi, sigma_0);
  double coef1 = beta(B_value, xp, xi, normal_d) * Us;
  double coef2 = beta(B0, xp, xi, normal_d) * Us;

  double mass_coef = coef2 - coef1;

  return mass_coef;
}
