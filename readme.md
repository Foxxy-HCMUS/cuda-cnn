# mini-dnn-cpp
**mini-dnn-cpp** is a C++ demo of deep neural networks. It is implemented purely in C++, whose only dependency, Eigen, is header-only. 

## Usage
Download and unzip [FASHION-MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist) dataset in `mini-dnn-cpp/data/mnist/`.

### Clean object files and executables
```shell
make clean
```

### Train
```shell
make setup
make train
make train_model
```

### Test
```shell
make setup
make gpu_basic
make test
make run
```
