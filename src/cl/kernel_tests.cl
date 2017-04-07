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
