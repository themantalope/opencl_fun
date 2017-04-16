__kernel void addem(__global float * a, __global float * b, __global float * c)
{

  int i = get_global_id(0);
  c[i] = a[i] + b[i];

}


__kernel void multiplyem(__global float * a, __global float * b, __global float * c)
{
  int i = get_global_id(0);
  c[i] = a[i] * b[i];
}

__kernel void testdot(__global float * a, __global float * b, __global float * c){
  int gid = get_global_id(0);
  c[gid] = dot(a[gid], b[gid]);
}

__kernel void test_rowaverage(__global float * in, __global float * out, const int nrows, const int ncols)
{
  float nrowsf = (float) nrows;
  for(int i = 0; i < nrows; i++){
    for (int j = 0; j < ncols; j++){
      out[j] += in[i * ncols + j];
      out[j] /= nrowsf;
    }
  }

}

__kernel void array_copy(__global float * in, __global float * out, const int nrows, const int ncols)
{
  int gid = get_global_id(0);
  for(int j = 0; j < ncols-1; j++){
    out[gid*ncols + j] = (float) (gid*ncols + j);
  }
}


__kernel void two_stage_reduce(__global float * in, __local float * scratch, __global float * out, __const int size)
{
  int gid = get_global_id(0);
  float accum = 0.0;
  // loop sequentially over the input
  while(gid < size){
    float element = in[gid];
    accum += element;
    gid += get_global_size(0);
  }

  // now do the parallel reduction
  int lid = get_local_id(0);
  scratch[lid] = accum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int i = get_local_size(0) / 2; i > 0; i >>= 1){
    if(lid < i){
      scratch[lid] += scratch[lid + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if(lid == 0){
    out[get_group_id(0)] = scratch[0];
  }

}

inline float mydot(__global float * a, __global float * b, const int size)
{
  float out = 0.0f;
  for(int i = 0; i < size; i++){
    out += a[i] * b[i];
  }
  return out;
}

inline float dotproduct(__global float * a, __global float * b, const int size)
{
  float out = 0.0f;
  for(int i = 0; i < size; i++){
    out += a[i] * b[i];
  }
  return out;
}

inline float sigmoid(__global float * X, __global float * theta, const int size)
{
  float linear_sum = dotproduct(X, theta, size);
  float exponential = pow(M_E_F, -linear_sum);
  float sig = pow(1.0f + exponential, -1.0f);
  return sig;
}

__kernel void sig(__global float * X, __global float * theta, __global float * out, const int nrows, const int ncols)
{
  int gid = get_global_id(0);
  out[gid] = sigmoid(&X[gid*ncols], theta, ncols);
}

__kernel void test_dot2(__global float * a, __global float * b, __global float * out, const int size)
{
  *out = mydot(a, b, size);
}

__kernel void powtest(__global float * a, __global float * b, __global float * out)
{
  int gid = get_global_id(0);
  out[gid] = pow(a[gid], b[gid]);
}

__kernel void test_reduction_avg_global(__global float * in, __global float * out, __global float * partial_sums, const int nrows)
{
  int gid = get_global_id(0);
  int global_size = get_global_size(0);
  float nrowsf = (float) nrows;

  partial_sums[gid] = in[gid];
  barrier(CLK_GLOBAL_MEM_FENCE);

  for(int i = global_size/2; i > 0; i >>= 1){
    if(gid < i){
      partial_sums[gid] += partial_sums[gid + i];
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
  }

  // if(gid == 0){
  //   out[0] = partial_sums[0];
  //   out[1] = (float) group_size;
  // }
  //
  out[gid] = partial_sums[gid];


}


__kernel void test_reduction_avg(__global float * in, __global float * out, __local float * partial_sums, const int nrows)
{
  int lid = get_local_id(0);
  int gid = get_global_id(0);
  int group_size = get_local_size(0);
  float nrowsf = (float) nrows;

  partial_sums[lid] = in[gid];
  barrier(CLK_LOCAL_MEM_FENCE);

  for(int i = group_size/2; i > 0; i >>= 1){
    if(lid < i){
      partial_sums[lid] += partial_sums[lid + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if(lid == 0){
    out[get_group_id(0)] = partial_sums[0];
  }
}

inline void matrix_row_avg(__global float * mat, __global float * out, __local float * partial_sums, const int nrows, const int ncols)
{
  int lid = get_local_id(0);
  int gid = get_global_id(0);
  float nrowsf = (float) nrows;

  for(int j = 0; j < ncols; j++){
    partial_sums[lid*ncols + j] = mat[gid*ncols + j];
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  for(int i = nrows/2; i > 0; i >>= 1){
    if(lid < i){
      for(int j = 0; j < ncols; j++){
        partial_sums[lid*ncols + j] += partial_sums[lid*ncols + i*ncols + j];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if(lid == 0){
    for(int j = 0 ; j < ncols; j ++){
      out[get_group_id(0) + j] = partial_sums[0 + j]/nrowsf;
    }
  }
}

__kernel void test_reduction_avg_matrix(__global float * in_mat, __global float * out_vec, __local float * partial_sums, const int nrows, const int ncols)
{
  matrix_row_avg(in_mat, out_vec, partial_sums, nrows, ncols);
}
