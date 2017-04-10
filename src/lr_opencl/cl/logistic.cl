float sigmoid(__global float * X, __global float * theta, const int size)
{
  float linear_sum = dotproduct(X, theta, size);
  float exponential = powf(M_E_F, -1.0 * linear_sum);
  float sig = powf(1.0 + exponential, -1.0);
  return sig;
}

float dotproduct(__global float * a, __global * float b, const int size)
{
  float out = 0.0;
  for(int i = 0; i < size; i++){
    out += a[i] * b[i];
  }
  return out;
}

// void rowaverage(__global float * in, float * out, const int nrows, const int ncols)
// {
//   float nrowsf = (float) nrows;
//   int i = get_global_id(0);
//   for(int j = 0; i < ncols; j++){
//     out[j] += in[i*ncols + j] / nrowsf;
//   }
//
// }





__kernel void logistic_cost_ols(__global float * X, __global float * theta, __global float * y, __global float * cost, const int nrows, const int ncols)
{
  int row_id = get_global_id(0);
  cost[row_id] = powf((sigmoid(&X[row_id * ncols], theta, ncols) - y[row_id]), 2.0);
}


__kernel void logistic_gradient_ols(__global float * X, __global float * theta, __global float * y, __global float * gradient, __global float * sig, const int nrows, const int ncols)
{
  int row_id = get_global_id(0);
  sig[row_id] = sigmoid(&X[row_id * ncols], theta, ncols);

  for(int i = 0; i < ncols; i ++){
    gradient[row_id + i] = (y[row_id] - sig[row_id]) * theta[i];
  }

}

__kernel void logistic_update_ols(__global float * theta, __global float * gradient, const float learning_rate, const int nrows, const int ncols)
{
  // get the average of what we computed before
  float grad_avg[2] = {0.0f};
  rowaverage(gradient, grad_avg, nrows, ncols);

  // update theta
  for (int j = 0; j < ncols; j++){
    theta[j] -= learning_rate * grad_avg[j];
  }

}
