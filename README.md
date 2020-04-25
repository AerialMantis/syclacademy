# SYCL Academy

![SYCL Academy](sycl_academy.png "SYCL Academy")

This repository provides materials that can be used for teaching SYCL 1.2.1. The
materials are provided using the "Creative Commons Attribution Share Alike 4.0
International" license.

## What is SYCL?

If you're not familiar with SYCL or would like some further resources for
learning about SYCL below are a list of useful resources:

*  Read a description of SYCL on the [Khronos website SYCL page](https://www.khronos.org/sycl/).
*  Go to the Khronos website to find [a list of SYCL resources](https://www.khronos.org/sycl/resources).
* Check out the [SYCL 1.2.1 reference guide](https://www.khronos.org/files/sycl/sycl-12-reference-card.pdf).
*  Browse SYCL news, blog posts, videos, projects and more on the [sycl.tech community website](https://sycl.tech/)
*  Get a list of the [available SYCL implementations](https://sycl.tech/#get-sycl)

## Contributing to SYCL Academy

Contributions to the materials are very gratefully received and this can be done
by submitting a Pull Request with any changes. Please limit the scope of each
Pull Request so that they can be reviewed and merged in a timely manner.

### List of Contributors

Codeplay Software Ltd., Heidelberg University.

## Supporting Organizations
Abertay University, Heriot Watt University, University of Edinburgh, TU Dresden,
University of the West of Scotland, Universitat Innsbruck,
Universidad de Concepcion and University of Salerno.

## Lesson Curriculum

The SYCL Academy curriculum is divided up into a number of short lessons
consisting of slides for presenting the material and a more detailed write-up,
each accompanied by a tutorial for getting hands on experience with the subject
matter.

Each of the lessons are designed to be self contained modules in order to
support both academic and training style teaching environments.

| Lesson | Title | Slides | Write-up | Exercise | Source | Solution |
|--------|-------|--------|----------|----------|--------|----------|
| 1 | Introduction to SYCL | [slides][lesson-1-slides] | TODO | [exercise][lesson-1-tutorial] | NA | NA |
| 2 | Topology Discover & Queue Creation | TODO | TODO | [exercise][lesson-2-tutorial] | [source][lesson-2-source] | [solution][lesson-2-solution] |
| 3 | SYCL Kernel Functions | [slides][lesson-3-slides] | TODO | [exercise][lesson-3-tutorial] | [source][lesson-3-source] | [solution][lesson-3-solution] |
| 4 | Managing Data in SYCL | [slides][lesson-4-slides] | TODO | [exercise][lesson-4-tutorial] | [source][lesson-4-source] | [solution][lesson-4-solution] |
| 5 | Image Grayscale (Optional) | TODO | TODO | TODO | TODO | TODO |
| 6 | Matrix Transpose (Optional) | TODO | TODO | TODO | TODO | TODO |
| 7 | Unified Shared Memory Extension (Optional) | TODO | TODO | TODO | TODO | TODO |

### How to use the Materials

To use these materials simply close this repository.

The lectures are written in reveal.js, and can be found in "Lesson_Materials",
in the sub-directory for each topic. To open them simply open the "index.html"
file in your browser. Your browser will have a "Full Screen" mode that can be
used to run the presentation, use the right and left cursors to move forward and
backward in the presentation.

The exercises can be found in "Code_Exercises" in the sub-directory for each
topic. Each exercise has a markdown document instructing what to do in the
exercise, a source file to start with and a solution file to provide an
example implementation to compare against.

## Building the Exercises

The exercises can be built for ComputeCpp CE, DPC++ and hipSYCL.

### Supported Platforms

Below is the supported platforms and devices for each SYCL implementations, see
this before deciding which SYCL implementation to use.

Make sure to also install the specified version to ensure that you can build
all of the exercises.

| Implementation | Supported Platforms | Supported Devices | Required Version |
|----------------|---------------------|-------------------|------------------|
| ComputeCpp | Windows 10 Visual Studio 2019 (64bit) <br> Ubtuntu 18.04 (64bit) | Intel CPU (OpenCL) <br> Intel GPU (OpenCL) | CE 2.0.0 |
| DPC++ | Intel DevCloud <br> Windows 10 Visual Studio 2019 (64bit) <br> Ubtuntu 18.04 (64bit) | Intel CPU (OpenCL) <br> Intel GPU (OpenCL) <br> Intel FPGA (OpenCL) <br> Nvidia GPU (CUDA) | 2021.1-beta05	|
| hipSYCL | Any Linux | CPU (OpenMP) <br> AMD GPU (ROCm)* <br> Nvidia GPU (CUDA) | Latest build |

\* Only the Vega 10 and Vega 20 devices are officially supported by AMD.

### Install SYCL implementations

First you'll need to install your chosen SYCL implementation and any
dependencies they require.

#### Installing ComputeCpp

To set up ComputeCpp download the [ComputeCpp CE package][computecpp-download]
and follow the [getting stared instructions][computecpp-getting-started].

#### Installing DPC++

To set up DPC++ follow the
[getting started instructions][dpcpp-getting-started].

If you are using the Intel DevCloud then the latest version of DPC++ will
already be installed and available in the path.

#### Installing hipSYCL

To set up hipSYCL download the [repository packages][hip-sycl-download] and
install them.

### Pre-requisites

Before building the exercises you'll need:

* One of the platforms in the support matrix above, depending on which SYCL
implementation you are wishing to build for.
* A C++17 or above tool-chain.
* An appropriate build system for the platform you are targeting (CMake, Ninja,
Make, Visual Studio).

### Configuring using CMake

Clone this repository, there are some additional dependencies configured as git
sub-modules so make sure to clone those as well. Then simply invoke CMake as
follows:

```
mkdir build

cd build

cmake ../ -G<cmake_generator> -A<cmake_arch> -D<sycl_implementation>=ON
```

For `<cmake_generator>` / `<cmake_arch>` we recommend:

* Visual Studio 16 2019 / x64 (Windows)
* Ninja / NA (Windows or Linux)
* Make / NA (Linux)

For `sycl_implementation` this can be one of:

* `SYCL_ACADEMY_USE_COMPUTECPP`
* `SYCL_ACADEMY_USE_DPCPP`
* `SYCL_ACADEMY_USE_HIPSYCL`

You can also specify the additional optional options:

-DSYCL_ACADEMY_INSTALL_ROOT=<path_to_sycl_impl_install_root>

For `<path_to_sycl_impl_install_root>` we recommend you specify the path to the
root directory of your SYCL implementation installation, though this may not
always be required.

-DSYCL_ACADEMY_ENABLE_SOLUTIONS=ON

This will enable building the solutions for each exercise as well as the source
files. This is disabled by default.

### Building with Make (DPC++ only)

TODO

## Online Interactive Tutorial

Hosted by tech.io, this [SYCL Introduction](https://tech.io/playgrounds/48226/introduction-to-sycl/introduction-to-sycl-2) tutorial introduces the concepts of SYCL. The website also provides the ability to compile and execute SYCL code from your web browser.

## Setting up Computers for SYCL

#### Machine Setup Instructions

ComputeCpp, a SYCL v1.2.1 conformant implementation by Codeplay Software provides setup instructions on [developer.codeplay.com](https://developer.codeplay.com). There is more detailed information about what hardware is supported by ComputeCpp on the [Platform Support](https://developer.codeplay.com/products/computecpp/ce/guides/platform-support) page.

Other SYCL implementations can be found on the SYCL community website [sycl.tech](https://sycl.tech).

#### Setting up a Docker Container for Lab Machines

In order to more easily deploy a SYCL implementation onto a bank of machines in a university lab for example, a Docker container can be used to deploy on these machines. This ensures all the dependencies that are needed are installed on each machine.

An example of how to set up a Docker container:

For Intel CPU or GPU
* Download the OpenCL Drivers for Intel from the [Intel website](https://software.intel.com/en-us/articles/opencl-drivers)
* Download the latest ComputeCpp release package from [developer.codeplay.com](https://developer.codeplay.com)

Now create a DockerFile that uses these packages, an example of how this might be done is below. Please note this file is not tested or maintained regularly but shows the elements that need to be installed.

```
FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y ninja-build
RUN apt-get install -y g++
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN apt-get install -y software-properties-common

RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test
RUN apt-get update

RUN python3 -m pip install cmake

# install Intel OpenCL drivers from downloaded package
RUN tar -xvf l_opencl_p_18.1.0.015.tgz
RUN cd l_opencl_p_18.1.0.015
RUN chmod +x install.sh
RUN ./install.sh

# Download the Khronos OpenCL headers
RUN git clone https://github.com/KhronosGroup/OpenCL-Headers.git
RUN mv OpenCL-Headers/CL/ /opt/khronos/opencl/include

# Set up the ICD Loader
RUN mkdir -p /etc/OpenCL/vendors/ \
    && echo "$OCL_LIB/libintelocl.so" > /etc/OpenCL/vendors/intel.icd

# Create a directory for ComputeCpp
RUN mkdir /usr/local/computecpp
RUN cd /usr/local/computecpp

# Copy the ComputeCpp release package and extract it to /usr/local/computecpp
RUN cp Ubuntu-16.04-64bit.tar.gz . 
RUN tar -xvf Ubuntu-16.04-64bit.tar.gz

# Add the ComputeCpp location to the path on the machine
ENV PATH=OCL_INC:OCL_LIB:/usr/local/computecpp/bin:/usr/local/computecpp/include:/usr/local/computecpp/lib:${PATH}

ENV CC=gcc-8
ENV CXX=g++-8
```

SYCL and the SYCL logo are trademarks of the Khronos Group Inc.

[computecpp-download]: https://www.codeplay.com/products/computesuite/computecpp
[computecpp-getting-started]: https://developer.codeplay.com/products/computecpp/ce/guides/getting-started?
[dpcpp-getting-started]: https://software.intel.com/en-us/articles/how-to-install-oneapi-products-and-run-data-parallel-cpp-code-samples
[hipsycl-download]: https://github.com/illuhad/hipSYCL/blob/master/install/scripts/README.md#installing-from-repositories
[hipsycl-getting-started]: https://github.com/illuhad/hipSYCL#building-and-installing-hipsycl

[lesson-1-slides]: ./Lesson_Materials/Lesson-1-Introduction-to-SYCL/index.html
[lesson-1-tutorial]: ./Code_Exercises/cppcon/docs/sycl_00_setting_up_computecpp.md

[lesson-2-slides]: ./Lesson_Materials/Lesson-2-Creating-a-SYCL-Queue/index.html
[lesson-2-tutorial]: ./Code_Exercises/cppcon/docs/sycl_01_configuring_a_queue.md
[lesson-2-source]: ./Code_Exercises/cppcon/source/sycl_01_configuring_a_queue.cpp
[lesson-2-solution]: ./Code_Exercises/cppcon/solutions/sycl_01_configuring_a_queue.cpp

[lesson-3-slides]: ./Lesson_Materials/Lesson-2-Launching-SYCL-Kernels/index.html
[lesson-3-tutorial]: ./Code_Exercises/cppcon/docs/sycl_02_hello_world.md
[lesson-3-source]: ./Code_Exercises/cppcon/source/sycl_02_hello_world.cpp
[lesson-3-solution]: ./Code_Exercises/cppcon/solutions/sycl_02_hello_world.cpp

[lesson-4-slides]: ./Lesson_Materials/Lesson-4-Managing-Data-in-SYCL-Applications/index.html
[lesson-4-tutorial]: ./Code_Exercises/cppcon/docs/sycl_03_vector_add.md
[lesson-4-source]: ./Code_Exercises/cppcon/source/sycl_03_vector_add.cpp
[lesson-4-solution]: ./Code_Exercises/cppcon/solutions/sycl_03_vector_add.cpp

[lesson-5-slides]: ./Lesson_Materials/Lesson-3-Handling-SYCL-Errors/index.html

