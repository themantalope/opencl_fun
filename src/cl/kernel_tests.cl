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
    out[get_group_id(0)] = partial_sums[0]/nrowsf;
  }


}

__kernel void test_reduction_avg_matrix(__global float * in_mat, __global float * out_vec, __local float * partial_sums, const int nrows, const int ncols)
{
  int lid = get_local_id(0);
  int gid = get_global_id(0);
  int group_size = get_local_size(0);
  float nrowsf = (float) nrows;

  for(int j = 0; j < ncols; j++){
    partial_sums[lid*ncols + j] = in_mat[gid*ncols + j];
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  for(int i = group_size/2; i > 0; i >>= 1){
    if(lid < i){
      for(int j = 0; j < ncols; j++){
        partial_sums[lid*ncols + j] += partial_sums[lid*ncols + i*ncols + j];
      }
    }
  }

  if(lid == 0){
    for(int j = 0 ; j < ncols; j ++){
      out_vec[get_group_id(0) + j] = partial_sums[0 + j] / nrowsf;
    }
  }
}
