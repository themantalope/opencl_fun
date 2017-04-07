// #include <math.h>

float dotproduct(__global float * a, __global float * b, int size)
{
  float out = 0.0;

  for (int i = 0; i<size; i++){
    out += a[i] * b[i];
  }

  return out;

}


float ols_cost_loc(__global float * x, __global float * theta, float y, int xsize)
{
  // cost = \sum (x * theta - y)**2

  float estimate = dotproduct(x, theta, xsize);
  float cost = powf(estimate - y, 2.0);

  return cost;
}

float compute_avg(__global float * vector, int size)
{
  float sum = 0.0;
  for(int i = 0; i < size; i++){
    sum += vector[i];
  }

  float avg = sum;
  return avg;
}

float compute_sum(__global float * array, int size){
  float sum = 0.0;
  for(int i = 0; i < size; i++){
    sum += array[i];
  }
  return sum;
}



__kernel void ols_cost(__global float * X, __global float * theta, __global float * y, __global float * cost, const int nrows, const int ncols)
{
  // here we are taking X to be a 2 dimensional array, tranlated from a row-majored indexed array
  // theta is a 1 dimensional vector and we are doing matrix multiplication, X * theta
  //

  int row_id = get_global_id(0);
  __global float * scratch;

  // float scratch_cost[nrows];

  cost[row_id] = ols_cost_loc(&X[row_id * ncols], theta, y[row_id], ncols);

  



}
