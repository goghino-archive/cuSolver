/* * How to compile (assume cuda is installed at /usr/local/cuda/)
* nvcc -c -I/usr/local/cuda/include getrf_example.cpp
* g++ -fopenmp -o a.out getrf_example.o -L/usr/local/cuda/lib64 -lcusolver -lcudart 
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define cpuErrchk(ans) { cpuAssert((ans), __FILE__, __LINE__); }
inline void cpuAssert(void *ptr, const char *file, int line, bool abort=true)
{
   if (ptr == NULL) 
   {
      fprintf(stderr,"CPUassert: %s %d\n", file, line);
      if (abort) exit(1);
   }
}

void printMatrix(int m, int n, const double*A, int lda, const char* name)
{ 
    for(int row = 0 ; row < m ; row++)
    { 
        for(int col = 0 ; col < n ; col++)
        { 
            double Areg = A[row + col*lda]; 
            printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg); 
        }
    }
}

void loadMatrix(int nrows, int ncols, double *A, int lda, const char *filename)
{
    FILE *f = NULL;
    f = fopen(filename, "r");

    if (f == NULL)
    {
        fprintf(stderr, "Failed to open file: %s for loading\n", filename);
        exit(1);
    }

    for (size_t i = 0; i < nrows; i++)
    {
        for (size_t j = 0; j < ncols; j++)
        {
            fscanf(f, "%lf", &A[i + j*lda]);
        }
    }

    fclose(f);
}

double residual(int N, double *X, double *A, double* B)
{
    // compute residual
    double *res = (double *) malloc(N * sizeof(double)); cpuErrchk(res);
    for (int i = 0; i < N; i++) {
        res[i] = 0;
    }
    
    // res = A*x - b
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            res[j] += A[i*N + j] * X[i];
        }
        res[i] -= B[i];
    }
    
    // norms
    double normr = 0.;
    double normb = 0.;
    for (int i = 0; i < N; i++) {
        normr += res[i]*res[i];
        normb += B[i]*B[i];
    }
    
    normr = sqrt(normr);
    normb = sqrt(normb);

    return normr/normb;
}

void generateMatrix(int nrows, int ncols, double *A, int lda)
{
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            A[i*lda + j] = rand()/(double)RAND_MAX;
        }
    }
}


/*    | 1 2 3 | 
* A = | 4 5 6 | 
*     | 7 8 10 | 
* 
* with pivoting: P*A = L*U 
*     | 0 0 1 | 
* P = | 1 0 0 | 
*     | 0 1 0 | 
* 
*     | 1 0 0 |             | 7 8 10 | 
* L = | 0.1429 1 0 |,   U = | 0 0.8571 1.5714 | 
*     | 0.5714 0.5 1 |      | 0 0 -0.5 | 
*/

int main(int argc, char*argv[]) 
{
    if (argc != 4 && argc != 2)
     {
         fprintf(stderr, "Usage: srun %s N [A.mat RHS.mat]\n", argv[0]);
         return -1;
     }

    int x = atoi(argv[1]);

    const int m = x; 
    const int lda = m; 
    const int ldb = m;

    double *A = (double *) malloc(lda*m * sizeof(double)); cpuErrchk(A);
    double *B = (double *) malloc(m * sizeof(double)); cpuErrchk(B);
    double *X = (double *) malloc(m * sizeof(double)); cpuErrchk(X); /* X = A\B */ 
    double *LU = (double *) malloc(lda*m * sizeof(double)); cpuErrchk(LU); /* L and U */ 
    int *Ipiv = (int *) malloc(m * sizeof(int)); cpuErrchk(Ipiv); /* host copy of pivoting sequence */ 
    int info = 0; /* host copy of error info */ 
    double *d_A = NULL; /* device copy of A */ 
    double *d_B = NULL; /* device copy of B */ 

    if (argc == 4)
    {
        loadMatrix(m, m, A, lda, argv[2]);
        loadMatrix(m, 1, B, lda, argv[3]);
    } else
    {
        generateMatrix(m, m, A, lda);
        generateMatrix(m, 1, B, 1);
    }

    printf("example of getrf \n"); 

    if (m < 10)
    {
        printf("A = (matlab base-1)\n"); 
        printMatrix(m, m, A, lda, "A"); 
        printf("=====\n"); 
        printf("B = (matlab base-1)\n"); 
        printMatrix(m, 1, B, ldb, "B"); 
        printf("=====\n");
    }

    cudaStream_t stream = NULL;
    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess; 
    cudaError_t cudaStat2 = cudaSuccess; 
    cudaError_t cudaStat3 = cudaSuccess; 
    cudaError_t cudaStat4 = cudaSuccess; 

    /* ------------------------------------------------------------- */
    /* step 1: create cusolver handle, alternatively bind a stream 
       cuSolverDN library was designed to solve dense linear systems */
    /* ------------------------------------------------------------- */
    cusolverDnHandle_t cusolverH = NULL;
    status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    // If the application performs several small independent computations,
    // or if it makes data transfers in parallel with the computation,
    // CUDA streams can be used to overlap these tasks.
    // cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking); 
    // assert(cudaSuccess == cudaStat1); 
    // status = cusolverDnSetStream(cusolverH, stream);
    // assert(CUSOLVER_STATUS_SUCCESS == status);

    // Timing using CUDA Events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time_solve = 0.;
    float time_memory1 = 0.;
    float time_memory2 = 0.;

    /* --------------------------------------------------- */
    /* step 2: allocate device memory and copy A to device */
    /* --------------------------------------------------- */ 

    cudaStat1 = cudaMalloc ((void**)&d_A, sizeof(double) * lda * m); 
    gpuErrchk(cudaStat1); 
    cudaStat2 = cudaMalloc ((void**)&d_B, sizeof(double) * m); 
    gpuErrchk(cudaStat2);
    
    cudaEventRecord(start);

    cudaStat1 = cudaMemcpy(d_A, A, sizeof(double)*lda*m, cudaMemcpyHostToDevice); 
    gpuErrchk(cudaStat1); 
    cudaStat2 = cudaMemcpy(d_B, B, sizeof(double)*m, cudaMemcpyHostToDevice); 
    gpuErrchk(cudaStat2);

    //get the timig
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_memory1, start, stop);

    /* ------------------------------------------------------------- */
    /* step 3: query working space of getrf 
       helper functions calculate the size of work buffers needed
       D = double precision */
    /* ------------------------------------------------------------- */
    int lwork = 0; /* size of workspace */ 
    status = cusolverDnDgetrf_bufferSize( cusolverH, m, m, d_A, lda, &lwork); 
    assert(CUSOLVER_STATUS_SUCCESS == status); 
    double *d_work = NULL; /* device workspace for getrf */ 
    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork); 
    assert(cudaSuccess == cudaStat1);

    /* ------------------------------------------------------------- */
    /* step 4: LU factorization */
    /* ------------------------------------------------------------- */
    cudaEventRecord(start);

    int *d_Ipiv = NULL; /* pivoting sequence */ 
    cudaStat3 = cudaMalloc ((void**)&d_Ipiv, sizeof(int) * m); 
    gpuErrchk(cudaStat3);
    int *d_info = NULL; /* error info */ 
    cudaStat4 = cudaMalloc ((void**)&d_info, sizeof(int)); 
    gpuErrchk(cudaStat4); 

    // perform LU with pivoting
    status = cusolverDnDgetrf( cusolverH, m, m, d_A, lda, d_work, d_Ipiv, d_info); 
    assert(CUSOLVER_STATUS_SUCCESS == status);
    cudaStat1 = cudaDeviceSynchronize();
    gpuErrchk(cudaStat1);

    cudaStat3 = cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost); 
    gpuErrchk(cudaStat3); 
    if ( 0 > info ){ 
        printf("%d-th parameter is wrong \n", -info); 
        exit(1); 
    } 
    
    if (m < 10)
    {
        // print pivots
        cudaStat1 = cudaMemcpy(Ipiv , d_Ipiv, sizeof(int)*m, cudaMemcpyDeviceToHost);
        gpuErrchk(cudaStat1);

        printf("pivoting sequence, matlab base-1\n"); 
        for(int j = 0 ; j < m ; j++){ 
            printf("Ipiv(%d) = %d\n", j+1, Ipiv[j]); 
        }

        printf("L and U = (matlab base-1)\n"); 
        cudaStat2 = cudaMemcpy(LU , d_A , sizeof(double)*lda*m, cudaMemcpyDeviceToHost);
        gpuErrchk(cudaStat2);
        printMatrix(m, m, LU, lda, "LU");
        printf("=====\n");
    }


    /* ------------------------------------------------------------- */
    /* * step 5: solve A*X = B 
     *     | 1 |      | -0.3333 | 
     * B = | 2 |, X = | 0.6667 | 
     *     | 3 |      | 0 | 
     */
     /* ------------------------------------------------------------- */

    int nrhs = 1;
    cublasOperation_t trans = CUBLAS_OP_N; //consider normal A, do not transpose
    status = cusolverDnDgetrs( cusolverH, trans, m, nrhs, d_A, lda, d_Ipiv, d_B, ldb, d_info); 

    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == status); 
    assert(cudaSuccess == cudaStat1); 

    //get the timig 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_solve, start, stop);

    cudaEventRecord(start);
    
    // copy back the result
    cudaStat1 = cudaMemcpy(X , d_B, sizeof(double)*m, cudaMemcpyDeviceToHost); 
    assert(cudaSuccess == cudaStat1); 

    //get the timig
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_memory2, start, stop);

    if (m < 10)
    {
        printf("X = (matlab base-1)\n"); 
        printMatrix(m, 1, X, ldb, "X");
        printf("=====\n");
    }

    printf("[cuSolve]: ||Ax - b|| / ||b|| = %e\n", residual(m, X, A, B));
    printf("[cuSolve]: Solve took %e ms\n", time_solve);
    printf("[cuSolve]: Memory transfers took %e ms\n", time_memory1 + time_memory2);

     /* free resources */ 
    free(A);
    free(B);
    free(X);
    free(LU);
    free(Ipiv);
    if (d_A ) cudaFree(d_A); 
    if (d_B ) cudaFree(d_B); 
    if (d_Ipiv ) cudaFree(d_Ipiv); 
    if (d_info ) cudaFree(d_info);
    if (d_work ) cudaFree(d_work); 
    if (cusolverH ) cusolverDnDestroy(cusolverH); 
    if (stream ) cudaStreamDestroy(stream); 
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaDeviceReset(); 

    return 0; 
}
