# Sample CMake project using mlpack


## Install the dependencies

You need to *install* [mlpack](https://www.mlpack.org) on your machine. We provide a script that install mlpack *on the virtual machine* that we provided. To install mlpack run the following commands in the terminal:

```
$> wget https://raw.githubusercontent.com/smover/mlpack_sample_project/master/install_mlpack.bash
$> bash install_mlpack.bash
```

The installation will take *some time* (the compilation of mlpack takes time). CMake shows you a percentage of completion --- it should not take more than ~30 minutes.

The script will ask you for the root password (e.g., to install the packages and libraries globally on your system).

We are compiling also the Python binding: you can use them to be fast in learning how the library works, but the final project should be in C++ (this also includes the learning part of the model).

The same script *should* work on your Ubuntu machine, but this is not guaranteed (different version, different dependencies). You can use the same script as a starting point to install mlpack for other Linux distributions.

## CMake project
This repository sets up a minimal project using [CMake](https://cmake.org) and including mlpack.

The project and the sample source code was extracted from https://github.com/mlpack/models


In a nutshell, CMake manages for you the process of building the project.

To build a CMake project you have to perform two steps:

1. Create and configure the build files (e.g., a `Makefile`) specific for your environment. The build file will take care of compiling and linking your code. When creating the build file CMake will perform tasks like determining your compiler, *finding the libraries and headers your project depends on*, ...

From the project main directory you can create the configuration files as follows:
```
$> mkdir build
$> cd build
$> cmake ../
```

Note that the directory `build` can be an arbitrary directory on your system and does not need to be nested in your project directory. The argument to the `cmake` command is the root directory of the project containing the configuration file (called `CMakeLists.txt`)

2. Compile your project

To compile your project just enter the `build` folder and type make.
```
$> cd build
$> make
```

Now the project is configured to create an executable called `nnsample` in the `build` directory.
To run the sample code you also need the data to train and test the neural network.
```
$> cd build
$> wget https://github.com/mlpack/models/raw/master/Kaggle/kaggle_train_test_dataset.zip
$> tar xvzf kaggle_train_test_dataset.zip
$> ./nnsample
```



## Your project

You can copy this template and use it as starting point for your project.

You'll probably need to extend the configuration file `CMakeLists.txt` to organize your code or to include additinal libraries.

The [CMake tutorial](https://cmake.org/cmake-tutorial/) is a good place to start using CMake.




