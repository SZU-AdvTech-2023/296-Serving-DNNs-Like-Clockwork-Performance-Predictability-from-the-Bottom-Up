# Troubleshooting

Please e-mail clockwork-users@googlegroups.com if you encounter any issues.

## Cannot find nvidia-ml

Currently, the CMakeLists assumes CUDA lives in either `/usr/local/cuda/lib64` (the default location in Ubuntu 14.x) or `/usr/lib/x86_64-linux-gnu/nvidia/current` (the default location for MPI cluster machines).  If you get build errors saying cannot find CUDA or cannot find nvidia-ml, then you'll need to update the `include_directories` and `link_directories` directives in the CMakeLists.txt with the CUDA location on your machine.

## Undefinied reference to tvm::runtime::ManagedCuda...

Undefined reference to tvm::runtime::ManagedCuda... -- this probably means you didn't build TVM properly.  Make sure you haven't modified or deleted the file `build/config.cmake` in the TVM repository.  `make clean` and `make` TVM again.

## Unable to set number of open files with ulimit

Unable to set number of open files with ulimit: default values are picked up from conf files, e.g. /etc/security/limits.conf, but they may be overwritten by files in a subdirectory, e.g. /etc/security/limits.d/mpi.conf

Make sure, upon restarting, that the correct ulimit values have been set, by running `./profile [check]`

## Cannot apply memory protection

If you are loading lots of models, you might see the following:
*  `src/clockwork/model/so.cpp:20: Check failed: lib_handle_ != nullptr: Failed to load SO /proc/26344/fd/14656/proc/26344/fd/14656: cannot apply additional memory protection after relocation: Cannot allocate memory`
*  `src/clockwork/model/so.cpp:20: Check failed: lib_handle_ != nullptr: Failed to load SO /proc/12386/fd/11804/proc/12386/fd/11804: failed to map segment from shared object`

Make sure your `mmap` limits have been correctly set as described above.  You can check by running `./profile [check]`

## CUDA: out of memory

Upon starting a worker process, you might see:
```
  what():  [20:58:09] /home/jcmace/clockwork/src/clockwork/cache.cpp:237: Check failed: e == cudaSuccess || e == cudaErrorCudartUnloading: CUDA: out of memory
```

If this happens, make sure the `weights_cache_size` is set to an appropriate value in `config/default.cfg`.  By default, these values are configured assuming a 32GB v100 GPU.  If you are using a 16GB v100 GPU, you need to reduce these values.  For a 16GB GPU, `10737418240L` is an appropriate value for `weights_cache_size`.

## cuModuleLoadData Error: CUDA_ERROR_OUT_OF_MEMORY

Clockwork cannot load infinitely many models.  Each model requires up to 1MB for its kernels on the GPU.  With thousands of models this can add up!

While loading models, the client may exit with:
```
  what():  [21:03:51] /home/jcmace/clockwork/src/clockwork/model/cuda.cpp:80: cuModuleLoadData Error: CUDA_ERROR_OUT_OF_MEMORY
```

If this happens, the GPU ran out of memory.  To fix it, you can:
* Use fewer models for your experiment
* Reduce the amount of GPU memory used for `weights_cache_size`.  You can modify this in `config/default.cfg` on workers.  Reducing weights cache will leave more memory available for kernels.

## Protobuf compiler version doesn't match library version

Installing protocol buffers is annoying.  The compiler version must match the library version.  Check where the `protoc` command leads to (`which protoc`).  Applications like `conda` sometimes install their own, different version of the protocol buffers compiler.  If you are on a Google Cloud VM, modify your `PATH` variable to remove conda.

## All cores exhausted

Clockwork workers require a minimum of 12 CPU cores (3 cores + 9 cores per GPU).

Clockwork controller requires 32 cores.

## cuModuleLoadData Error: CUDA_ERROR_NO_BINARY_FOR_GPU

This may happen when using models from [clockwork-modelzoo-volta](https://gitlab.mpi-sws.org/cld/ml/clockwork-modelzoo-volta).  All models are compiled specifically for `sm_70` GPUs (Volta architecture).  While it is possible to compile models for different architectures, this process is not automated.  If possible, the easiest solution is to use `Tesla v100` GPUs.