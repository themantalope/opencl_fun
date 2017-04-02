__kernel void first(__global int* num1, __global int* num2, __global int* out)
{
    int i = get_global_id(0);
    out[i] = num1[i]*num1[i]+ num2[i]*num2[i];
}

__kernel void access2darray(__global float * array, __global float * out, const int row_to_access, const int nrows, const int ncols)
{

  __global float * row = &array[row_to_access * ncols]; //gets the starting position of the row
  for(int i=0; i < ncols; i++){
    out[i] = row[i];
  }



}
