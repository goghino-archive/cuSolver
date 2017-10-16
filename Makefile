all: target

target: main.cu
	nvcc --gpu-architecture=compute_60 --gpu-code=sm_60 main.cu -lcusolver -o main

deviceinfo: deviceInfo.cu
	nvcc deviceInfo.cu -o deviceInfo

clean:
	rm main

run: target
	srun ./main 3 matrices/testA.mat matrices/testRHS.mat
	srun ./main 518 matrices/Schur1354.mat matrices/RHS1354.mat
	srun ./main 2888 matrices/Schur9241.mat matrices/RHS9241.mat
	srun ./main 8182 matrices/Schur13659.mat matrices/RHS13659.mat
