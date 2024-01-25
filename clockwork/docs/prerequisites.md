# Installation Pre-Requisites

This document describes how to install the packages Clockwork requires.


## 1. NVIDIA Driver and CUDA 


Make sure NVIDIA driver and CUDA are installed and CUDA is on your PATH. MPI cluster machines have CUDA 9 installed by default; however Clockwork may also work with other CUDA versions. You can check if CUDA is installed and the version by running `nvcc --version`.

**Google Cloud VMs** We have tested the below instructions using Operating System `Deep Learning on Linux`, Version `Deep Learning Image: Base m55 (with CUDA 10.0)`.  This image comes with CUDA pre-installed.  If you use this image, remove `conda` from your `PATH` variable, else you will see protobuf version mismatch errors.

## 2. Required Packages

The following apt packages are pre-requisites:

```
apt-get install libtbb-dev libasio-dev libconfig++-dev libboost-all-dev g++-8 \
make cmake automake autoconf libtool curl unzip clang llvm
```
## 3. Installing Protobuf

```
git clone --recursive -b v3.12.0 https://github.com/protocolbuffers/protobuf.git
cd protobuf
./autogen.sh && ./configure 
make -j $(nproc) 
make install
/sbin/ldconfig
cd ..
```

