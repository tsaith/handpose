#include "../include/types.h"

Point_c::Point_c(int x, int y)
{
  _x = x;
  _y = y;
}

Point_c::~Point_c()
{
}

Rect_c::Rect_c(int x, int y, int width, int height)
{
  _x = x;
  _y = y;
  _width  = width;
  _height = height;
}

Rect_c::~Rect_c()
{
}
