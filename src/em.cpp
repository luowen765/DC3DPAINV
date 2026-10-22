/*
 * @Description: Implementation of the utility functions declared in em.h:
 *   time and memory reporting, and point-to-point distances.
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

#include "em.h"

void print_time(double time_in_seconds) {
  int seconds = (int)time_in_seconds;
  int hr = seconds / 3600;
  seconds %= 3600;
  int min = seconds / 60;
  double sec = time_in_seconds - hr * 3600 - min * 60;
  std::cout << hr << " hr " << min << " min " << sec << " sec\n";
}

double length_two_point(double *coord1, double *coord2) {
  double length = (coord2[0] - coord1[0]) * (coord2[0] - coord1[0]) +
                  (coord2[1] - coord1[1]) * (coord2[1] - coord1[1]) +
                  (coord2[2] - coord1[2]) * (coord2[2] - coord1[2]);
  length = std::sqrt(length);
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

double GetCurrentMemoryUsage() {
  rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  return double(usage.ru_maxrss / 1024.0 /
                1024.0);
}
