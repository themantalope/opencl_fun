

inline float dotproduct(__global float * a, __global float * b, const int size)
{
  float out = 0.0f;
  for(int i = 0; i < size; i++){
    out += a[i] * b[i];
  }
  return out;
}

inline float sigmoid(__global float * X, __global float * theta, int row_id, int nrows, int ncols)
{
  float linear_sum = 0.0f;
  for(int j = 0; j < ncols; j++){
    linear_sum += (X[row_id + j*nrows] * theta[j]);
  }

  float exponential = pow(M_E_F, -linear_sum);
  exponential += 1.0f;
  float sig = 1.0f / exponential;
  return sig;
}

__kernel void matrix_row_mean(__global float * in, __global float * out, __local float * scratch, const int nrows, const int ncols)
{
  int gid = get_global_id(0);
  int lid = get_local_id(0);
  int global_size = get_global_size(0);

  // first step is a sequential loop
  for(int j = 0; j < ncols; j++){
    float accum = 0.0f;
    while(gid < nrows){
      accum += in[gid*ncols + j];
      gid += global_size;
    }
    scratch[lid*ncols + j] = accum;
  }


  // now do a parallel reduction

  barrier(CLK_LOCAL_MEM_FENCE);
  for(int i = get_local_size(0)/2; i > 0; i >>=1 ){
    if(lid < i){
      for(int j = 0; j < ncols; j++){
        scratch[lid*ncols + j] = scratch[lid*ncols + i*ncols + j];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if(lid == 0){
    for(int j = 0; j < ncols; j++){
      out[get_group_id(0)*ncols + j] = scratch[0*ncols + j];
    }

  }
}

/*
* Functions here are specific to logisitic regression
*/

__kernel void sig(__global float * X, __global float * theta, __global float * out, const int nrows, const int ncols)
{
  int gid = get_global_id(0);
  out[gid] = sigmoid(X, theta, gid, nrows, ncols);
}

__kernel void logistic_cost_ols(__global float * X, __global float * theta, __global float * y, __global float * cost, const int nrows, const int ncols)
{
  int gid = get_global_id(0);
  float diff = sigmoid(X, theta,gid,nrows, ncols) - y[gid];
  cost[gid] = powf(diff, 2.0f);
  cost[gid] /= 2.0;
}


__kernel void logistic_gradient_ols(__global float * X, __global float * theta, __global float * y, __global float * gradient, const int nrows, const int ncols)
{
  int gid = get_global_id(0);

  for(int j = 0; j < ncols; j++){
    gradient[gid*ncols + j] = (y[gid] - sigmoid(X, theta, gid, nrows, ncols)) * X[gid + j*nrows];
  }

}

__kernel void logisitc_prediction(__global float * X, __global float * theta, __global float * out, const int nrows, const int ncols)
{
  int gid = get_global_id(0);
  out[gid] = sigmoid(X, theta,gid,nrows, ncols);
}

__kernel void logistic_update_ols(__global float * theta, __global float * gradient, const float learning_rate, const int nrows, const int ncols)
{
  for (int j = 0; j < ncols; j++){
    theta[j] -= learning_rate * gradient[j];
  }

}
