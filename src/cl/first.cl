__kernel void first(__global int* num1, __global int* num2,__global int* out)
{
    int i = get_global_id(0);
    out[i] = num1[i]*num1[i]+ num2[i]*num2[i];
}
