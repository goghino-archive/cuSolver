all: target deviceinfo

target: main.cu
	nvcc main.cu -lcusolver -o main

deviceinfo: deviceInfo.cu
	nvcc deviceInfo.cu -o deviceInfo

clean:
	rm main