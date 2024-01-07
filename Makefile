train: train.o
	nvcc -o train -lm -lcuda -lrt train.o src/network.o src/mnist.o src/layer/*.o src/loss/*.o src/optimizer/*.o -I./ -L/usr/local/cuda/lib64 -lcudart

train.o: train.cc
	nvcc --compile train.cc -I./ -L/usr/local/cuda/lib64 -lcudart

train_model: train
	./train

# test: test.o dnnNetwork.o network.o mnist.o layer loss optimizer
# 	nvcc -o test -lm -lcuda -lrt test.o dnnNetwork.o src/network.o src/mnist.o src/layer/*.o src/loss/*.o src/optimizer/*.o -I./ -L/usr/local/cuda/lib64 -lcudart

test: test.o network.o mnist.o layer loss optimizer
	nvcc -o test -lm -lcuda -lrt test.o src/network.o src/mnist.o src/layer/*.o src/loss/*.o src/optimizer/*.o -I./ -L/usr/local/cuda/lib64 -lcudart

test.o: test.cc
	nvcc --compile test.cc -o test.o -I./ -L/usr/local/cuda/lib64 -lcudart

# dnnNetwork.o: dnnNetwork.cc
# 	nvcc --compile dnnNetwork.cc -o dnnNetwork.o -I./ -L/usr/local/cuda/lib64 -lcudart

network.o: src/network.cc
	nvcc --compile src/network.cc -o src/network.o -I./ -L/usr/local/cuda/lib64 -lcudart

mnist.o: src/mnist.cc
	nvcc --compile src/mnist.cc -o src/mnist.o  -I./ -L/usr/local/cuda/lib64 -lcudart

layer: src/layer/conv.cc src/layer/ave_pooling.cc src/layer/fully_connected.cc src/layer/max_pooling.cc src/layer/relu.cc src/layer/sigmoid.cc src/layer/softmax.cc 
	nvcc --compile src/layer/ave_pooling.cc -o src/layer/ave_pooling.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc --compile src/layer/conv.cc -o src/layer/conv.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc --compile src/layer/conv_gpu.cc -o src/layer/conv_gpu.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc --compile src/layer/fully_connected.cc -o src/layer/fully_connected.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc --compile src/layer/max_pooling.cc -o src/layer/max_pooling.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc --compile src/layer/relu.cc -o src/layer/relu.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc --compile src/layer/sigmoid.cc -o src/layer/sigmoid.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc --compile src/layer/softmax.cc -o src/layer/softmax.o -I./ -L/usr/local/cuda/lib64 -lcudart

cpu:
	nvcc --compile src/layer/gpu/utils.cu -o src/layer/utils.o -I./ -L/usr/local/cuda/lib64 -lcudart 
	nvcc --compile src/layer/gpu/conv_cpu.cc -o src/layer/conv_cpu.o -I./ -L/usr/local/cuda/lib64 -lcudart 

gpu_basic:
#	rm -f src/layer/conv_gpu.o
#	nvcc --compile src/layer/conv_gpu.cc -o src/layer/conv_gpu.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc --compile src/layer/gpu/utils.cu -o src/layer/utils.o -I./ -L/usr/local/cuda/lib64 -lcudart 
	nvcc --compile src/layer/gpu/conv_kernel.cu -o src/layer/conv_kernel.o -I./ -L/usr/local/cuda/lib64 -lcudart 

gpu_v1:
# 	rm -f src/layer/conv_gpu.o
# 	nvcc --compile src/layer/conv_gpu.cc -o src/layer/conv_gpu.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc --compile src/layer/gpu/utils.cu -o src/layer/utils.o -I./ -L/usr/local/cuda/lib64 -lcudart 
	nvcc --compile src/layer/gpu/conv_kernel1.cu -o src/layer/conv_kernel1.o -I./ -L/usr/local/cuda/lib64 -lcudart  

gpu_v2:
# 	rm -f src/layer/conv_gpu.o
# 	nvcc --compile src/layer/conv_gpu.cc -o src/layer/conv_gpu.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc --compile src/layer/gpu/utils.cu -o src/layer/utils.o -I./ -L/usr/local/cuda/lib64 -lcudart 
	nvcc --compile src/layer/gpu/conv_kernel2.cu -o src/layer/conv_kernel2.o -I./ -L/usr/local/cuda/lib64 -lcudart 

gpu_v3:
# 	rm -f src/layer/conv_gpu.o
# 	nvcc --compile src/layer/conv_gpu.cc -o src/layer/conv_gpu.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc --compile src/layer/gpu/utils.cu -o src/layer/utils.o -I./ -L/usr/local/cuda/lib64 -lcudart 
	nvcc -arch=sm_75 --compile src/layer/gpu/conv_kernel3.cu -o src/layer/conv_kernel3.o -I./ -L/usr/local/cuda/lib64 -lcudart 

	

loss: src/loss/cross_entropy_loss.cc src/loss/mse_loss.cc
	nvcc --compile src/loss/cross_entropy_loss.cc -o src/loss/cross_entropy_loss.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc --compile src/loss/mse_loss.cc -o src/loss/mse_loss.o -I./ -L/usr/local/cuda/lib64 -lcudart

optimizer: src/optimizer/sgd.cc
	nvcc --compile src/optimizer/sgd.cc -o src/optimizer/sgd.o -I./ -L/usr/local/cuda/lib64 -lcudart

setup: 
	make network.o
	make mnist.o
	make layer
	make loss
	make optimizer

clean:
##	rm -f infoGPU infoGPU.o test test.o
	rm -f src/layer/*.o
	rm test.o
	rm test

run: test
	./test