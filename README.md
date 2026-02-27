# [Paper Title: From-fields-to-random-trees]

Implementation of our paper [From-fields-to-random-trees]([link_to_paper]). 

This repository contains the C++ implementation of the **SPT** (Spanning Tree) algorithm for solving Markov Random Fields (MRFs) / Energy Minimization problems.

## Installation

### Prerequisites
Our C++ implementation requires **C++17** and depends on several external libraries:
- [CMake](https://cmake.org/) (>= 3.16)
- [Eigen3](https://eigen.tuxfamily.org/)
- [OpenMP](https://www.openmp.org/)
- [libDAI](https://github.com/dbat/libDAI) (0.3.1)
- GMP / GMPXX

### Environment Setup (Linux)

1. Install system dependencies (Eigen3, GMP, OpenMP, CMake):
```bash
sudo apt-get update
sudo apt-get install cmake g++ libeigen-dev libgmp-dev libgmpxx4ldbl libomp-dev
```
2. Install and compile libDAI:
Please follow the official libDAI installation guide to compile the library.

3. Configure CMakeLists.txt:
Before building the project, you must update the paths in CMakeLists.txt to point to your local libDAI installation. Replace the placeholders with your actual paths:
set(libDAI_DIR /path/to/your/libDAI)
# ...
target_link_libraries(SPT_cpp PRIVATE ... /path/to/your/libdai.a)

## Compilation
Build the project using CMake:
```bash
mkdir build
cd build
cmake ..
make -j4
```
This will generate the SPT_cpp executable in the build directory.

## Data Preparation
Place your MRF model files (e.g., UAI format) in the data/ directory.
(Note: Add specific instructions here if you have a script to download datasets or if users need to format their graphs in a specific way).

## Usage
### Running SPT
Run the compiled SPT_cpp executable to evaluate the algorithm on your models.
```bash
./build/SPT_cpp --model_path data/sample_model.uai --num_trees 10
```
(Note: Please update the command-line arguments above to match how main.cpp actually parses inputs in your code).

Key Parameters:
--model_path: Path to the input graphical model.
--num_trees: Number of random spanning trees to sample.
(Add any other relevant flags your main.cpp accepts, like thread count or max iterations).

## Acknowledgements
This work was supported by the National Key R&D Program of China under grant 2022YFA1003900.

