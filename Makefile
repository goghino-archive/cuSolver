all: target deviceinfo

CUFLAGS = -O3

target: main.cu
	nvcc $(CUFLAGS) main.cu -lcusolver -o main

deviceinfo: deviceInfo.cu
	nvcc $(CUFLAGS) deviceInfo.cu -o deviceInfo

clean:
	rm main

run: target
	srun ./main 3 matrices/testA.mat matrices/testRHS.mat
	srun ./main 518 matrices/Schur1354.mat matrices/RHS1354.mat
	srun ./main 2888 matrices/Schur9241.mat matrices/RHS9241.mat
	srun ./main 8182 matrices/Schur13659.mat matrices/RHS13659.mat
