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
  const int gid_0 = get_global_id(0);
  const int gid_1 = get_global_id(1);
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
    for(int c = 0; c < TSK; c++){
      #pragma unroll
      for(int wm = 0; wm < WPTM; wm++){
        int r = tidm + RTSM*wm;
        Asub[r][c] = A[(offsetM + r)*K + TSK*t + c];
        Bsub[r][c] = B[(offsetN + r)*K + TSK*t + c];
      }
    }


    // #pragma unroll
    // for(int r = 0; r < TSM; r++){
    //   #pragma unroll
    //   for(int c = 0; c < TSK; c++){
    //     Asub[r][c] = A[(offsetM + r)*K + TSK*t + c];
    //     Bsub[r][c] = B[(offsetN + r)*K + TSK*t + c];
    //   }
    // }

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

__kernel void SPMVM(){}
