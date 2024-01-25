# Building Clockwork

Make sure you have completed the [Installation Pre-Requisites](prerequisites.md)

## Modified TVM

Clone our modified TVM and check out our modified branch (`clockwork-v0.6`):
```
git clone --recursive -b clockwork-v0.6 https://gitlab.mpi-sws.org/cld/ml/tvm
```

Build TVM
```
cd tvm/build
cmake ..
make -j $(nproc)
cd ..
```

Set `TVM_HOME` environment variable and add `$TVM_HOME/build` to your `LD_LIBRARY_PATH` and `DYLD_LIBRARY_PATH` environment variables
```
echo "export TVM_HOME=`pwd`" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TVM_HOME/build" >> ~/.bashrc
echo "export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$TVM_HOME/build" >> ~/.bashrc
source ~/.bashrc
```

## Clockwork

Check out Clockwork

```
git clone --recursive https://gitlab.mpi-sws.org/cld/ml/clockwork.git
```

Build Clockwork

```
cd clockwork
mkdir build
cd build
cmake ..
make -j $(nproc)
```

## Recommended: Models

Pre-compiled models can be downloaded from the [clockwork-modelzoo-volta](https://gitlab.mpi-sws.org/cld/ml/clockwork-modelzoo-volta) repository.  You will need these models if you are running the experiments described in [clockwork-results](https://gitlab.mpi-sws.org/cld/ml/clockwork-results).

Models are only needed on worker machines.

```
git clone https://gitlab.mpi-sws.org/cld/ml/clockwork-modelzoo-volta.git
```

Set `CLOCKWORK_MODEL_DIR` to point to your checkout

## Recommended: Azure Traces

Some of the experiments described in [clockwork-results](https://gitlab.mpi-sws.org/cld/ml/clockwork-results) use workload traces from Microsoft Azure.  You will need these if you are running those experiments.

Traces are only needed on client machines

```
git clone https://gitlab.mpi-sws.org/cld/trace-datasets/azure-functions
```

Set `AZURE_TRACE_DIR` to point to your checkout

**Note for artifact evaluators:** The Clockwork submission used a pre-release dataset of azure-functions that differs substantially from the above dataset.  The pre-release dataset is deprecated and not public.  We will make the pre-release dataset available privately for artifact evaluation only.


## Troubleshooting


### Protobuf compiler version doesn't match library version

Installing protocol buffers is annoying.  The compiler version must match the library version.  Check where the `protoc` command leads to (`which protoc`).  Applications like `conda` sometimes install their own, different version of the protocol buffers compiler.  If you are on a Google Cloud VM, modify your `PATH` variable to remove conda.

### G++ version

Compilation can fail with versions of g++ less than 8.