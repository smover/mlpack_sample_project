# Sample CMake project using mlpack


## Install the dependencies

You need to *install* [mlpack](https://www.mlpack.org) on your machine. We provide a script that install mlpack *on the virtual machine* that we provided. To install mlpack run the following commands in the terminal:

```
$> wget
$> bash install_mlpack.bash
```

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




