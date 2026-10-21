/*
 * @Description: Implementations of the utility functions declared in em.h, including
 *   sorting, statistics, file output, and memory reporting.
 * @Author: Lewen Qiu (original DC3DPAFEM module); adapted for DC3DPAINV by
 *   Lewen Qiu and Huang Chen
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

#include "em.h"

void display(REV x) {
  for (int i = 0; i < x.size(); i++) {
    std::cout << x[i] << '\t';
  }
  std::cout << std::endl;
}

void display(Vertex x) {
  for (int i = 0; i < 3; i++) {
    std::cout << x(i) << '\t';
  }
  std::cout << std::endl;
}

void display(std::vector<Vertex> x) {
  for (auto ele : x) {
    display(ele);
  }
}

std::vector<int> sort(const std::vector<int> &a) {

  std::vector<int> tmp(a.size(), 0.0);
  for (size_t i = 0; i < a.size(); i++)
    tmp[i] = a[i];
  std::sort(tmp.begin(), tmp.end());
  return tmp;

}

std::vector<int> unique(const std::vector<int> &a) {

  std::vector<int> tmp(a.size()), u;
  for (size_t i = 0; i < a.size(); i++)
    tmp[i] = a[i];
  std::unique_copy(tmp.begin(), tmp.end(), back_inserter(u));

  return tmp;

}

void print_time(double time_in_seconds) {
  int seconds = (int)time_in_seconds;
  int hr = seconds / 3600;
  seconds %= 3600;
  int min = seconds / 60;
  double sec = time_in_seconds - hr * 3600 - min * 60;
  std::cout << hr << " hr " << min << " min " << sec << " sec\n";
}

void print_value(double value, int dim1, int dim2, std::string fn) {
  std::ostringstream ss;

  ss << fn;
  std::ofstream os;

  os.precision(8);

  os.open(ss.str().c_str(), std::ios::app);

  if (dim1 == -1) {
    std::cout << "check the dim of data! ";
  }

  if (dim2 == -1) {
    os << dim1 << "\t" << value << "\n";
  } else {
    os << dim1 << "\t" << dim2 << "\t" << value << "\n";
  }
}

double median(const REV &a) {
  std::vector<int> temp;
  temp.resize(a.size());
  for (int i = 0; i < a.size(); i++) {
    temp[i] = a[i];
  }
  int dim = a.size();
  if (dim == 1)
    return temp[0];
  if (dim > 1) {
    std::vector<int> tmp = sort(temp);
    if (std::fabs(dim / 2.0 - rint(dim / 2.0)) < 1e-12) {
      return (tmp[dim / 2] + tmp[dim / 2 - 1]) / 2.0;
    } else {
      return tmp[(dim - 1) / 2];
    }
  }
  return 0.0;
}

REV logspace(double a, double b, int n) {
  REV y(n);
  double q = pow(10.0, (b - a) / (n - 1));
  for (int i = 0; i < n; i++) {
    y[i] = pow(10, a) * pow(q, i);
  }
  return y;
}

REV normalRandn(int n) {
  std::default_random_engine gen(1337);
  std::normal_distribution<double> dis(0, 1);
  REV y(n);
  for (int i = 0; i < n; ++i)
    y[i] = dis(gen);
  return y;
}

bool save(const REV &a, const std::string &filename, IOFormat format) {

  if (filename.rfind(VECTORASCSUFFIX) != std::string::npos)
    format = Ascii;
  else if (filename.rfind(VECTORBINSUFFIX) != std::string::npos)
    format = Binary;
  std::string fname(filename);

  if (format == Ascii) {
    if (fname.rfind(".") == std::string::npos)
      fname += VECTORASCSUFFIX;

    std::ofstream file;
    file.open(fname.c_str());
    if (!file) {
      throwError(filename + ": " + strerror(errno));
    }

    file.setf(std::ios::scientific, std::ios::floatfield);
    file.precision(14);

    for (size_t i = 0, imax = a.size(); i < imax; i++)
      file << a(i) << std::endl;
    file.close();
  } else {
    if (fname.rfind(".") == std::string::npos)
      fname += VECTORBINSUFFIX;

    FILE *file;
    file = fopen(fname.c_str(), "w+b");
    if (!file) {
      throwError(filename + ": " + strerror(errno));
    }

    int64 count = (int64)(a.size());
    size_t ret = 0;
    ret = fwrite((char *)&count, sizeof(int64), 1, file);
    if (ret == 0) {
      fclose(file);
      return false;
    }
    for (size_t i = 0; i < a.size(); i++)
      ret = fwrite((char *)&a(i), sizeof(double), 1, file);
    fclose(file);
  }
  return true;
}

bool saveVec(const REV &a, const std::string &filename, IOFormat format) {
  return save(a, filename, format);
}

double length_two_point(double *coord1, double *coord2) {
  double length = (coord2[0] - coord1[0]) * (coord2[0] - coord1[0]) +
                  (coord2[1] - coord1[1]) * (coord2[1] - coord1[1]) +
                  (coord2[2] - coord1[2]) * (coord2[2] - coord1[2]);
  length = std::sqrt(length);
  return length;
}

double length_two_point(Vector &coord1, Vector &coord2) {
  myassert(coord1.Size() == 3);
  myassert(coord2.Size() == 3);
  double c1[3];
  double c2[3];
  for (int i = 0; i < 3; i++) {
    c1[i] = coord1[i];
    c2[i] = coord2[i];
  }
  double length = length_two_point(c1, c2);
  return length;
}

double length_two_point(Point_double &coord1, Point_double &coord2) {
  double c1[3];
  double c2[3];
  for (int i = 0; i < 3; i++) {
    c1[i] = coord1(i);
    c2[i] = coord2(i);
  }
  double length = length_two_point(c1, c2);
  return length;
}

bool cmp(std::vector<double> a, std::vector<double> b) {
  if (a[0] != b[0])
    return a[0] < b[0];
  else if (a[1] != b[1])
    return a[1] < b[1];
  else
    return a[2] < b[2];
}

bool cmp2(std::vector<double> a, std::vector<double> b) {
  if (a[0] != b[0])
    return a[0] < b[0];
  else if (a[1] != b[1])
    return a[1] < b[1];
  else if (a[2] != b[2])
    return a[2] < b[2];
  else if (a[3] != b[3])
    return a[3] < b[3];
  else if (a[4] != b[4])
    return a[4] < b[4];
  else
    return a[5] < b[5];
}

bool cmpMN15(std::vector<double> a, std::vector<double> b) {

  int m = 1;
  int n = 5;
  int pos_1 = (m - 1) * 3;
  int pos_5 = (n - 1) * 3;
  if (a[pos_1] != b[pos_1])
    return a[pos_1] < b[pos_1];
  else if (a[pos_1 + 1] != b[pos_1 + 1])
    return a[pos_1 + 1] < b[pos_1 + 1];
  else if (a[pos_1 + 2] != b[pos_1 + 2])
    return a[pos_1 + 2] < b[pos_1 + 2];

  else if (a[pos_5] != b[pos_5])
    return a[pos_5] < b[pos_5];
  else if (a[pos_5 + 1] != b[pos_5 + 1])
    return a[pos_5 + 1] < b[pos_5 + 1];
  else
    return a[pos_5 + 2] < b[pos_5 + 2];
}

bool cmpN(std::vector<double> a, std::vector<double> b, int N) {

  int pos = (N - 1) * 3;
  if (a[N] != b[N])
    return a[N] < b[N];
  else if (a[N + 1] != b[N + 1])
    return a[N + 1] < b[N + 1];
  else
    return a[N + 2] < b[N + 2];
}

bool cmp3(std::vector<double> a, std::vector<double> b) {
  int pos_1 = (1 - 1) * 3;
  int pos_2 = (2 - 1) * 3;
  int pos_3 = (3 - 1) * 3;

  if (a[pos_1] != b[pos_1])
    return a[pos_1] < b[pos_1];
  else if (a[pos_1 + 1] != b[pos_1 + 1])
    return a[pos_1 + 1] < b[pos_1 + 1];
  else if (a[pos_1 + 2] != b[pos_1 + 2])
    return a[pos_1 + 2] < b[pos_1 + 2];

  if (a[pos_2] != b[pos_2])
    return a[pos_2] < b[pos_2];
  else if (a[pos_2 + 1] != b[pos_2 + 1])
    return a[pos_2 + 1] < b[pos_2 + 1];
  else if (a[pos_2 + 2] != b[pos_2 + 2])
    return a[pos_2 + 2] < b[pos_2 + 2];

  if (a[pos_3] != b[pos_3])
    return a[pos_3] < b[pos_3];
  else if (a[pos_3 + 1] != b[pos_3 + 1])
    return a[pos_3 + 1] < b[pos_3 + 1];
  else
    return a[pos_3 + 2] < b[pos_3 + 2];
}

bool cmp4(std::vector<double> a, std::vector<double> b) {
  int pos_1 = (1 - 1) * 3;
  int pos_2 = (2 - 1) * 3;
  int pos_3 = (3 - 1) * 3;
  int pos_4 = (4 - 1) * 3;

  if (a[pos_1] != b[pos_1])
    return a[pos_1] < b[pos_1];
  else if (a[pos_1 + 1] != b[pos_1 + 1])
    return a[pos_1 + 1] < b[pos_1 + 1];
  else if (a[pos_1 + 2] != b[pos_1 + 2])
    return a[pos_1 + 2] < b[pos_1 + 2];

  if (a[pos_2] != b[pos_2])
    return a[pos_2] < b[pos_2];
  else if (a[pos_2 + 1] != b[pos_2 + 1])
    return a[pos_2 + 1] < b[pos_2 + 1];
  else if (a[pos_2 + 2] != b[pos_2 + 2])
    return a[pos_2 + 2] < b[pos_2 + 2];

  if (a[pos_3] != b[pos_3])
    return a[pos_3] < b[pos_3];
  else if (a[pos_3 + 1] != b[pos_3 + 1])
    return a[pos_3 + 1] < b[pos_3 + 1];
  else if (a[pos_3 + 2] != b[pos_3 + 2])
    return a[pos_3 + 2] < b[pos_3 + 2];

  if (a[pos_4] != b[pos_4])
    return a[pos_4] < b[pos_4];
  else if (a[pos_4 + 1] != b[pos_4 + 1])
    return a[pos_4 + 1] < b[pos_4 + 1];
  else
    return a[pos_4 + 2] < b[pos_4 + 2];
}

void remove_duplicate_vertex(std::vector<mfem::Vertex> &all_point) {

  struct Point {
    double x;
    double y;
    double z;
    bool operator==(const Point &other) const {
      return (x == other.x) && (y == other.y) && (z == other.z);
    }
    bool operator<(const Point &other) const {
      if (x != other.x) {
        return x < other.x;
      }
      if (y != other.y) {
        return y < other.y;
      }
      return z < other.z;
    }
  };

  std::vector<Point> points;
  points.resize(all_point.size());
  for (int i = 0; i < all_point.size(); i++) {
    points[i].x = all_point[i](0);
    points[i].y = all_point[i](1);
    points[i].z = all_point[i](2);
  }

  std::sort(points.begin(), points.end());

  auto unique_end = std::unique(points.begin(), points.end());
  points.erase(unique_end, points.end());

  all_point.resize(points.size());
  for (int i = 0; i < points.size(); i++) {
    all_point[i](0) = points[i].x;
    all_point[i](1) = points[i].y;
    all_point[i](2) = points[i].z;
  }

}

double GetCurrentMemoryUsage() {
  rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  return double(usage.ru_maxrss / 1024.0 /
                1024.0);
}

bool equalDM(DenseMatrix a, DenseMatrix b) {
  myassert(a.Size() == b.Size());
  for (int i = 0; i < a.Size(); i++) {
    for (int j = 0; j < a.Size(); j++) {
      if (a(i, j) != b(i, j))
        return false;
    }
  }
  return true;
}

bool equalVector(Vector a, Vector b) {
  myassert(a.Size() == b.Size());
  for (int i = 0; i < a.Size(); i++) {
    if (a(i) != b(i))
      return false;
  }
  return true;
}
