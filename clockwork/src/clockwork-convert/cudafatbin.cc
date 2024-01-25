#include "clockwork/model/cuda.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "clockwork/tvm/pack_args.h"
#include "dmlc/memory_io.h"

#include "clockwork/util.h"
#include <iostream>
#include "tbb/task_scheduler_init.h"
#include "clockwork/common.h"
#include <sstream>
#include <atomic>
#include <thread>
#include <fstream>
#include <istream>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <pods/pods.h>
#include <pods/binary.h>
#include <pods/buffers.h>
#include <pods/streams.h>
#include <cuda_runtime.h>
#include <chrono>
#include "clockwork/cuda_common.h"
#include "clockwork/cache.h"
#include "clockwork/util.h"
#include "clockwork/model/so.h"
#include <dmlc/logging.h>
#include <stdio.h>
#include <string.h>

void convert_ptx_to_fatbin(std::string model) {
    clockwork::so::SharedObject so(model+".so");

    // Extract the CUDA module blob
    char* cuda_blob = reinterpret_cast<char*>(so.GetSymbol(tvm::runtime::symbol::tvm_dev_mblob));
    CHECK(cuda_blob != nullptr) << "Could not find " << tvm::runtime::symbol::tvm_dev_mblob;

      uint64_t nbytes = 0;
      for (size_t i = 0; i < sizeof(nbytes); ++i) {
        uint64_t c = cuda_blob[i];
        nbytes |=  (c & 0xffUL) << (i * 8);
      }

      dmlc::MemoryFixedSizeStream fs(
          const_cast<char*>(cuda_blob + sizeof(nbytes)), static_cast<size_t>(nbytes));
      dmlc::Stream* stream = &fs;
      uint64_t size;
      CHECK(stream->Read(&size));
      CHECK(size == 1) << "Only expected one dev_mblob, found " << size;

      std::string tkey;
      CHECK(stream->Read(&tkey));
      std::string fkey = "module.loadbinary_" + tkey;
      CHECK(tkey == "cuda") << "Expected dev_mblob of type cuda, found " << tkey;
      std::string fmt;
      stream->Read(&fmt);

      std::unordered_map<std::string, tvm::runtime::FunctionInfo> fmap;
      stream->Read(&fmap);


      std::string data;
      stream->Read(&data);

    std::ofstream outs(model+".ptx");

    outs << data;
    outs.close();

    std::cout << "count is " << data.size() << std::endl;

    // Next, to generate fatbin invoke
    // nvcc --fatbin tvm-model.ptx --gpu-architecture=compute_52 --gpu-code=sm_52
}

void loadmodules(std::string model) {

    clockwork::so::SharedObject so(model+".so");

    // Extract the CUDA module blob
    char* cuda_blob = reinterpret_cast<char*>(so.GetSymbol(tvm::runtime::symbol::tvm_dev_mblob));
    CHECK(cuda_blob != nullptr) << "Could not find " << tvm::runtime::symbol::tvm_dev_mblob;

      uint64_t nbytes = 0;
      for (size_t i = 0; i < sizeof(nbytes); ++i) {
        uint64_t c = cuda_blob[i];
        nbytes |=  (c & 0xffUL) << (i * 8);
      }

      dmlc::MemoryFixedSizeStream fs(
          const_cast<char*>(cuda_blob + sizeof(nbytes)), static_cast<size_t>(nbytes));
      dmlc::Stream* stream = &fs;
      uint64_t size;
      CHECK(stream->Read(&size));
      CHECK(size == 1) << "Only expected one dev_mblob, found " << size;

      std::string tkey;
      CHECK(stream->Read(&tkey));
      std::string fkey = "module.loadbinary_" + tkey;
      CHECK(tkey == "cuda") << "Expected dev_mblob of type cuda, found " << tkey;
      std::string fmt;
      stream->Read(&fmt);

      std::unordered_map<std::string, tvm::runtime::FunctionInfo> fmap;
      stream->Read(&fmap);


      std::string data;
      stream->Read(&data);

      std::vector<CUmodule> modules;

      CUcontext pctx;


      cudaSetDevice(0);


      cuInit(0);
      while (true) {
          CUmodule module;

          uint64_t pre = clockwork::util::now();
          CUresult result = cuModuleLoadData(&module, data.c_str());
          uint64_t post = clockwork::util::now();
          std::cout << "cuModuleLoadData size=" << data.size() << " took " << (post-pre) << std::endl;
          if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) {
            const char *msg;
            cuGetErrorName(result, &msg);
            std::ostringstream os;
            os << "cuModuleLoadData Error: " << msg << "\n";
            LOG(FATAL) << os.str();    
          }

      }
}

int main(int argc, char *argv[]) {
    std::cout << "begin" << std::endl;

    std::string model = "/home/jcmace/modelzoo/resnet50/tesla-m40_batchsize1/tvm-model";

    loadmodules(model);

    std::cout << "end" << std::endl;
}
