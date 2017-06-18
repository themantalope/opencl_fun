// #include "settings.h"

// TODO: Figure out how to get this into the settings.h file, and get the code
// to compile

// transpose sizes
#define TRANSPOSEX 16
#define TRANSPOSEY 16

// padding sizes
#define PADDINGX 16
#define PADDINGY 16

// constants for SPGMM
#define TSM 128 // tile size in M (rows of A)
#define TSN 128 // tile size in N (columns of B)
#define TSK 16 // tile size in K (columns of A and rows of B)
#define WPTM 8 // work per thread in M
#define WPTN 8 // work per thread in N
#define RTSM (TSM/WPTM) // total number of threads in M
#define RTSN (TSN/WPTN) // total number of threads in N
#define LPTA ((TSK*TSM)/(RTSM*RTSN)) // number of loads per thread required for A
#define LPTB ((TSK*TSN)/(RTSM*RTSN)) // number of loads per thread required for B

// preprocessor macros
#define MIN(a,b) ((a) > (b) ? (b) : (a))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CEIL_DIV(x,y) (((x) + (y) - 1) / (y))
#define MOD2(x,y) ((x) % (y))
#define DIV2(x,y) ((x) / (y))
// kernels and code
__kernel void zeroPad(const int P,
                      const int Q,
                      const __global float * in,
                      const int P_XL,
                      const int Q_XL,
                      __global float * out){

  // thread ids
  const int tx = get_group_id(0)*PADDINGX + get_local_id(0);
  const int ty = get_group_id(1)*PADDINGY + get_local_id(1);

  // check if we are within the bounds [0,P_XL] and [0,Q_XL]
  if(tx < P_XL && ty < Q_XL){
    float value;
    if(tx < P && ty < Q){
      value = in[tx*Q + ty];
    } else {
      value = 0.0f;
    }

    out[tx*Q_XL + ty] = value;
  }
}

__kernel void zeroTrim(const int P,
                       const int Q,
                       const __global float * in,
                       const int P_XS,
                       const int Q_XS,
                       __global float * out){
  // thread ids
  const int tx = get_group_id(0)*PADDINGX + get_local_id(0);
  const int ty = get_group_id(1)*PADDINGY + get_local_id(1);
  // check if we are in the bounds [0,P] and [0,Q]
  if(tx < P_XS && ty < Q_XS){
    out[tx*Q_XS + ty] = in[tx*Q + ty];
  }
}

__kernel void transpose(const int P,
                        const int Q,
                        const __global float * in,
                        __global float * out){
  const int tx = get_local_id(0);
  const int ty = get_local_id(1);
  const int ID0 = get_group_id(0)*TRANSPOSEX + tx; // 0..P
  const int ID1 = get_group_id(1)*TRANSPOSEY + ty; // 0..Q

  // local memory for doing the shuffling
  __local float tempbuff[TRANSPOSEX][TRANSPOSEY];

  if (ID0 < P && ID1 < Q){
    tempbuff[tx][ty] = in[ID0*Q + ID1];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  const int nID0 = get_group_id(1)*TRANSPOSEY + tx;
  const int nID1 = get_group_id(0)*TRANSPOSEX + ty;

  if(nID0 < Q && nID1 < P){
    out[nID0*P + nID1] = tempbuff[ty][tx];
  }
}

// TODO: do the matrix math stuff!!!

// single precision, general matrix multiplication with 2D register blocking.
// multiplies A*B, where A is a MxK matrix, and B is a KxN matrix
// output (C) is a MxN matrix
// assumes that matrix B has been transposed (NxK)
__kernel void SPGMM(const int M,
                    const int N,
                    const int K,
                    const __global float * A,
                    const __global float * B,
                    __global float * C){
  // thread identifiers
  const int tidm = get_local_id(0);
  const int tidn = get_local_id(1);
  const int offsetM = get_group_id(0) * TSM;
  const int offsetN = get_group_id(1) * TSN;

  // local mem to fit a tile of A and B
  __local float Asub[TSM][TSK];
  __local float Bsub[TSN][TSK+2];

  // allocate reg space
  float Areg;
  float Breg[WPTN];
  float acc[WPTM][WPTN];

  // initalize the accumlation registers
  #pragma unroll
  for(int wm = 0; wm < WPTM; wm++){
    #pragma unroll
    for(int wn = 0; wn < WPTN; wn++){
      acc[wm][wn] = 0.0f;
    }
  }

  // loop over all tiles
  const int nTiles = K/TSK;
  for(int t = 0; t < nTiles; t++){
    // load a tile of A and B into local mem
    #pragma unroll
    for(int wm = 0; wm < WPTM; wm++){
      int r = tidm + RTSM*wm;

      #pragma unroll
      for(int c = 0; c < TSK; c++){
        Asub[r][c] = A[(offsetM + r)*K + TSK*t + c];
        Bsub[r][c] = B[(offsetN + r)*K + TSK*t + c];
      }

    }

    barrier(CLK_LOCAL_MEM_FENCE);
    // loop over values in a single tile
    #pragma unroll
    for(int k=0; k<TSK; k++){
      // cache values of Bsub in the register
      #pragma unroll
      for(int wn = 0; wn<WPTN; wn++){
        int col = tidn + wn*RTSN;
        Breg[wn] = Bsub[col][k];
      }

      // perform the computation
      #pragma unroll
      for(int wm = 0; wm<WPTM; wm++){
        int row = tidm + wm*RTSM;
        Areg = Asub[row][k];
        #pragma unroll
        for(int wn=0; wn<WPTN; wn++){
          acc[wm][wn] += Areg * Breg[wn];
        }
      }
    }
    // sync local memory before loading the next tile
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // we be done. store the results in C
  #pragma unroll
  for(int wm = 0; wm<WPTM; wm++){
    int globalRow = offsetM + tidm + wm*RTSM;
    #pragma unroll
    for(int wn = 0; wn<WPTN; wn++){
      int globalCol = offsetN + tidn + wn*RTSN;
      C[globalRow*N + globalCol] = acc[wm][wn];
    }
  }

}

#define BLOCK_SIZE 16
#define ACCUMULATION_ELEMENTS (BLOCK_SIZE*BLOCK_SIZE)
#define ELEMENTS_SHARED 16

__kernel void SPGMM2(const int M,
                     const int N,
                     const int K,
                     const __global float * A,
                     const __global float * B,
                     __global float * C)
{
  const int local_m = get_local_id(0);
  const int local_n = get_local_id(1);
  const int global_m = get_global_id(0);
  const int global_n = get_global_id(1);

  __local float Asub[BLOCK_SIZE][ELEMENTS_SHARED];
  __local float Bsub[BLOCK_SIZE][ELEMENTS_SHARED];

  float Areg[ELEMENTS_SHARED];
  float Breg;
  float accum[BLOCK_SIZE][BLOCK_SIZE];

  // initalize the accumlation array
  #pragma unroll
  for(int row = 0; row < BLOCK_SIZE; row++){
    #pragma unroll
    for(int col = 0; col < BLOCK_SIZE; col++){
      accum[row][col] = 0.0f;
    }
  }

  const int n_tiles = K / ELEMENTS_SHARED;
  for(int t = 0; t < n_tiles; t++){
    // load the submatricies
    #pragma unroll
    for(int row = 0; row < BLOCK_SIZE; row++){
      #pragma unroll
      for(int col = 0; col < ELEMENTS_SHARED; col++){
        Asub[row][col] = A[(global_m + local_m + row)*K + t*ELEMENTS_SHARED + col];
        Bsub[row][col] = B[(global_n + local_n + row)*K + t*ELEMENTS_SHARED + col];
      }
    }


    barrier(CLK_LOCAL_MEM_FENCE);

    // do the computation

    // load a row of A intro the register
    // #pragma unroll
    // for(int col = 0; col < ELEMENTS_SHARED; col++){
    //   Areg[col] = Asub[local_m][col];
    // }

    // get the accumulation
    // #pragma unroll
    // for(int col = 0; col < ELEMENTS_SHARED; col++){
    //   Breg = Bsub[local_n][col];
    //   accum[local_m][local_n] += Areg[col] * Breg;
    // }

    for(int a_row = 0; a_row < BLOCK_SIZE; a_row++){
      // load a row of A into the register
      #pragma unroll
      for(int col = 0; col < ELEMENTS_SHARED; col++){
        Areg[col] = Asub[a_row][col];
      }

      #pragma unroll
      for(int b_row = 0; b_row < BLOCK_SIZE; b_row++){
        #pragma unroll
        for(int col = 0; col < ELEMENTS_SHARED; col++){
          Breg = Bsub[b_row][col];
          accum[a_row][b_row] += Breg * Areg[col];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // write the output to C
  for(int row = 0; row < BLOCK_SIZE; row++){
    for(int col = 0; col < BLOCK_SIZE; col++){
      C[(global_m + local_m + row)*N + global_n + local_n + col] = accum[row][col];
    }
  }


}


/* Matrix multiplication: C = A * B.
 * Device code.
 * from: http://gpgpu-computing4.blogspot.com/2009/10/matrix-multiplication-3-opencl.html
 */

// Thread block size
#define BLOCK_SIZE 16

//////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! wA is A's width and wB is B's width
//////////////////////////////////////////////////////
__kernel void
matrixMul(__global float* C,
          __global float* A,
          __global float* B,
          int wA,
          int wB)
{
    // Block index
    int bx = get_group_id(0);
    int by = get_group_id(1);

    // Thread index
    int tx = get_local_id(0);
    int ty = get_local_id(1);

    // Index of the first sub-matrix of A processed
    // by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed
    // by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the
    // sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed
    // by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the
    // sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    float Csub = 0.0f;
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep)
    {

        // Declaration of the local memory array As
        // used to store the sub-matrix of A
        __local float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the local memory array Bs
        // used to store the sub-matrix of B
        __local float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from global memory
        // to local memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices
        // are loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k){
          Csub += As[ty][k] * Bs[k][tx];
        }


        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        barrier(CLK_LOCAL_MEM_FENCE);

    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;

}


__kernel void SPMVM(){}
