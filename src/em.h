/*
 * @Description: Basic types, geometry utilities, and assertion helpers shared by
 *   the DC3DPAINV modules.
 * @Author: Lewen Qiu (original DC3DPAFEM module); adapted for DC3DPAINV by
 *   Lewen Qiu and Huang Chen
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

#ifndef _EM_H
#define _EM_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <fstream>
#include <iostream>

#include <string>
#include <unistd.h>

#include <sys/resource.h>

#include "mfem.hpp"
using namespace mfem;

namespace EM {
static const double PI = 3.1415926535897932384626433832795;

typedef Eigen::SparseMatrix<double, Eigen::RowMajor> RESM;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> REM;
typedef Eigen::VectorXd REV;
typedef Eigen::VectorXd DenseVector;
typedef Eigen::Matrix<double, 3, 3> Matrix3D;
typedef Eigen::Matrix<double, 1, 3> RowVector3D;
typedef Eigen::Vector3d Vector3D;
}
using namespace EM;

#define MY_ENABLE_ASSERT
void static assert_fail(const char *file, int line, const char *func,
                        const char *expr, const std::string &msg) {
  (void)file;
  (void)line;
  (void)func;
  (void)expr;
  (void)msg;
  std::string err_msg = std::string("Assertion fails:") + expr +
                        ", message:" + msg + ", at " + file + ":" +
                        std::to_string(line) + ":" + func;
  std::cout << err_msg << std::endl;
  std::terminate();
}

#ifdef MY_ENABLE_ASSERT
#define myassert(expr)                                                         \
  do {                                                                         \
    if (!(expr)) {                                                             \
      assert_fail(__FILE__, __LINE__, __func__, #expr, "");                    \
    }                                                                          \
  } while (0)
#else
#define myassert(expr) (void)0
#endif

#ifndef _Point_double_H
#define _Point_double_H

#include <cassert>
#include <cmath>
#include <iostream>

class Point_double {
public:
  Point_double(const double x = 0., const double y = 0., const double z = 0.);
  Point_double(const Point_double &p);
  virtual ~Point_double();

  friend std::ostream &operator<<(std::ostream &os,
                                  const Point_double &Point_double);

  Point_double &operator=(const Point_double &p);
  double operator()(const unsigned int i) const;
  double &operator()(const unsigned int i);
  Point_double operator+(const Point_double &v) const;
  Point_double operator-(const Point_double &v) const;
  Point_double operator*(const double a) const;
  double operator*(const Point_double &v) const;
  bool operator==(const Point_double &v) const;
  bool operator<(const Point_double &v) const;
  void operator=(const double a);

  Point_double cross(const Point_double &v) const;
  Point_double unit() const;
  double size() const;
  void zero();
  double *get_xyz() { return _coords; }

protected:
  double _coords[3];
};

inline Point_double::Point_double(const double x, const double y,
                                  const double z) {
  _coords[0] = x;
  _coords[1] = y;
  _coords[2] = z;
}

inline Point_double::Point_double(const Point_double &p) {
  for (unsigned int i = 0; i < 3; i++)
    _coords[i] = p._coords[i];
}

inline Point_double &Point_double::operator=(const Point_double &p) {
  _coords[0] = p._coords[0];
  _coords[1] = p._coords[1];
  _coords[2] = p._coords[2];

  return (*this);
}

inline Point_double::~Point_double() {

}

inline double Point_double::operator()(const unsigned int i) const {
  assert(i < 3);
  return _coords[i];
}

inline double &Point_double::operator()(const unsigned int i) {
  assert(i < 3);
  return _coords[i];
}

inline Point_double Point_double::operator+(const Point_double &p) const {
  return Point_double(_coords[0] + p._coords[0], _coords[1] + p._coords[1],
                      _coords[2] + p._coords[2]);
}

inline Point_double Point_double::operator-(const Point_double &p) const {
  return Point_double(_coords[0] - p._coords[0], _coords[1] - p._coords[1],
                      _coords[2] - p._coords[2]);
}

inline Point_double Point_double::operator*(const double factor) const {
  return Point_double(_coords[0] * factor, _coords[1] * factor,
                      _coords[2] * factor);
}

inline double Point_double::operator*(const Point_double &p) const {
  return (_coords[0] * p(0) + _coords[1] * p(1) + _coords[2] * p(2));
}

inline bool Point_double::operator==(const Point_double &rhs) const {
  return ((fabs(_coords[0] - rhs._coords[0]) +
           fabs(_coords[1] - rhs._coords[1]) +
           fabs(_coords[2] - rhs._coords[2])) < 3 * 1e-6);
}

inline bool Point_double::operator<(const Point_double &rhs) const {

  if (*this == rhs)
    return false;
  if ((*this)(0) < rhs(0))
    return true;
  else if ((*this)(0) > rhs(0))
    return false;
  else if (std::abs((*this)(0) - rhs(0)) < 1e-6) {
    if ((*this)(1) < rhs(1))
      return true;
    else if ((*this)(1) > rhs(1))
      return false;
    else if (std::abs((*this)(1) - rhs(1)) < 1e-6) {
      if ((*this)(2) < rhs(2))
        return true;
      else if ((*this)(2) > rhs(2))
        return false;
    }
  }
  return false;
}

inline void Point_double::operator=(const double a) {
  _coords[0] = a;
  _coords[1] = a;
  _coords[2] = a;
}

inline Point_double Point_double::cross(const Point_double &p) const {
  return Point_double(_coords[1] * p._coords[2] - _coords[2] * p._coords[1],
                      -_coords[0] * p._coords[2] + _coords[2] * p._coords[0],
                      _coords[0] * p._coords[1] - _coords[1] * p._coords[0]);
}

inline Point_double Point_double::unit() const {
  const double length = size();
  return Point_double(_coords[0] / length, _coords[1] / length,
                      _coords[2] / length);
}

inline double Point_double::size() const {
  double value = std::pow(_coords[0], 2) + std::pow(_coords[1], 2) +
                 std::pow(_coords[2], 2);

  return std::sqrt(value);
}

inline void Point_double::zero() {
  _coords[0] = 0.;
  _coords[1] = 0.;
  _coords[2] = 0.;
}

inline std::ostream &operator<<(std::ostream &os,
                                const Point_double &Point_double) {
  os << Point_double(0) << "\t" << Point_double(1) << "\t" << Point_double(2);
  return os;
}

#endif

struct PointPair {

  Point_double p1, p2;

  bool operator<(const PointPair &other) const {
    if (p1 < other.p1)
      return true;
    if (other.p1 < p1)
      return false;
    return p2 < other.p2;
  }
};

inline std::ostream &operator<<(std::ostream &output, REV &v) {
  for (int i = 0; i < v.size(); i++) {
    std::cout << v(i) << " ";
  }
  return output;
}

inline std::ostream &operator<<(std::ostream &output, const REV &v) {
  for (int i = 0; i < v.size(); i++) {
    std::cout << v(i) << " ";
  }
  return output;
}

void print_time(double time_in_seconds);

double length_two_point(double *coord1, double *coord2);
double length_two_point(Point_double &coord1, Point_double &coord2);

double GetCurrentMemoryUsage();

#endif
