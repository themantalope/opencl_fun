#include <math.h>

float dot(float * a, float * b, int size){
  float out = 0.0

  for (i = 0; i<size; i++){
    out += a[i] * b[i];
  }

  return out;

}


float ols_cost(float * x, float * theta, float y int xsize){
  // cost = \sum (x * theta - y)**2

  float estimate = dot(x, theta);
  float cost = powf(estimate - y, 2.0);
}




__kernel void ols_cost(__global float * X, __global float * theta, __global float * y, const int nrows, const int ncols){
  // here we are taking X to be a 1 dimensional array, tranlated from a row-majored indexed array
}
