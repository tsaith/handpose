#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>

using namespace std;

class Point_c {
public:
  Point_c(int x, int y);
  ~Point_c();

  int _x, _y;

};

class Rect_c {
public:
  Rect_c(int x, int y, int width, int height);
  ~Rect_c();

  int _x, _y, _width, _height;

};

