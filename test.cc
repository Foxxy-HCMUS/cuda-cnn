/*
 * CNN demo for MNIST dataset
 * Author: Kai Han (kaihana@163.com)
 * Details in https://github.com/iamhankai/mini-dnn-cpp
 * Copyright 2018 Kai Han
 */
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
 
#include "src/layer.h"
#include "src/layer/conv.h"
#include "src/layer/fully_connected.h"
#include "src/layer/ave_pooling.h"
#include "src/layer/max_pooling.h"
#include "src/layer/relu.h"
#include "src/layer/sigmoid.h"
#include "src/layer/softmax.h"
#include "src/loss.h"
#include "src/loss/mse_loss.h"
#include "src/loss/cross_entropy_loss.h"
#include "src/mnist.h"
#include "src/network.h"
#include "src/optimizer.h"
#include "src/optimizer/sgd.h"

#include "src/layer/conv_gpu.h"
// #include "src/layer/gpu/utils.h"

Network LeNet5() 
{
    Network dnn;
    Layer *conv1_gpu = new ConvGPU(1, 28, 28, 6, 5, 5); 
    Layer *pool1 = new MaxPooling(6, 24, 24, 2, 2, 2);
    Layer *conv2_gpu = new ConvGPU(6, 12, 12, 16, 5, 5);
    Layer *pool2 = new MaxPooling(16, 8, 8, 2, 2, 2);

    Layer* fc3 = new FullyConnected(pool2->output_dim(), 120);
    Layer* fc4 = new FullyConnected(120, 84);
    Layer* fc5 = new FullyConnected(84, 10);

    Layer* relu1 = new ReLU;
    Layer* relu2 = new ReLU;
    Layer* relu3 = new ReLU;
    Layer* relu4 = new ReLU;
    Layer* relu5 = new ReLU;
    Layer* softmax = new Softmax;

    dnn.add_layer(conv1_gpu);
    dnn.add_layer(relu1);
    dnn.add_layer(pool1);
    dnn.add_layer(conv2_gpu);
    dnn.add_layer(relu2);
    dnn.add_layer(pool2);
    dnn.add_layer(fc3);
    dnn.add_layer(relu3);
    dnn.add_layer(fc4);
    dnn.add_layer(relu4);
    dnn.add_layer(fc5);
    // dnn.add_layer(relu5);
    dnn.add_layer(softmax);

    return dnn;
}

 
int main() {
    // 0. Check GPU
    GPUUtils gpu;
    gpu.printDeviceInfo();  

    // data
    MNIST dataset("./data/fashion-mnist/");
    dataset.read();
    int n_train = dataset.train_data.cols();
    int dim_in = dataset.train_data.rows();
    std::cout << "mnist train number: " << n_train << std::endl;
    std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;

    // dnn gpu
    Network dnn_gpu = LeNet5();

    // Load parameters
    dnn_gpu.load_parameters("./checkpoint/weights-1.bin");
    dnn_gpu.forward(dataset.test_data);
    float acc = compute_accuracy(dnn_gpu.output(), dataset.test_labels);
    std::cout << std::endl << "Test accuraccy on GPU: " << acc << std::endl;
    return 0;
}

