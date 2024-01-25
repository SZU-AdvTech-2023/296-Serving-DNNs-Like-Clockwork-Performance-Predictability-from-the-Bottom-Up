#include <catch2/catch.hpp>
#include <cuda_runtime.h>
#include "clockwork/cuda_common.h"
#include "clockwork/util.h"
#include "clockwork/test/util.h"
#include "clockwork/model/so.h"
#include "clockwork/model/cuda.h"
#include <atomic>
#include <thread>
#include <unistd.h>
#include "clockwork/thread.h"

using namespace clockwork;

void fill_memory(size_t &total_malloced, size_t &peak_usage) {
    size_t cudaMallocSize = 16 * 1024 * 1024;
    total_malloced = 0;

    cudaError_t status;
    
    size_t free, total;
    status = cudaMemGetInfo(&free, &total);
    REQUIRE((status == cudaSuccess));
    size_t initialUsed = total-free;

    std::vector<void*> ptrs;
    for (unsigned i = 0; true; i++) {
        void* ptr;
        status = cudaMalloc(&ptr, cudaMallocSize);
        REQUIRE((status == cudaSuccess || status == cudaErrorMemoryAllocation));

        if (status == cudaErrorMemoryAllocation) {
            status = cudaMemGetInfo(&free, &total);
            REQUIRE((status == cudaSuccess));
            peak_usage = total - free;

            break;
        } else {
            ptrs.push_back(ptr);
            total_malloced += cudaMallocSize;
        }
    }
    for (void* &ptr : ptrs) {
        status = cudaFree(ptr);
        REQUIRE(status == cudaSuccess);
    }

    status = cudaMemGetInfo(&free, &total);
    REQUIRE((status == cudaSuccess));
    REQUIRE( (total-free) == initialUsed );
}

class TransferThread {
public:
    bool success = true;
    cudaError_t status;
    std::string error;

    std::thread thread;

    std::atomic_int &countdown_begin;
    std::atomic_int &countdown_end;
    std::atomic_bool &alive;

    TransferThread(unsigned gpu_id, std::vector<unsigned> cores, std::atomic_int &countdown_begin, std::atomic_int &countdown_end, std::atomic_bool &alive, void* host, size_t host_size) : 
        thread(&TransferThread::run, this, gpu_id, host, host_size, cores), countdown_begin(countdown_begin), countdown_end(countdown_end), alive(alive) {
    }

    bool successful() {
        if (status != cudaSuccess) {
            this->status = status;
            this->error = std::string(cudaGetErrorString(status));
            this->success = false;
            std::cout << this->error << std::endl;
            return false;
        }
        return true;
    }

    void run(unsigned gpu_id, void* host, size_t host_size, std::vector<unsigned> cores) {
        status = cudaSetDevice(gpu_id);
        if (!successful()) { countdown_begin--; countdown_end--; return; };

        cudaStream_t stream;
        status =cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, 0);
        if (!successful()) { countdown_begin--; countdown_end--; return; };
        // TODO: set core affinity for GPU

        size_t transfer_size = 16*1024*1024; // 16MB transfers
        size_t device_size = host_size; // 10GB on device

        void* device;

        status = cudaMalloc(&device, device_size);
        if (!successful()) { countdown_begin--; countdown_end--; return; };

        uint64_t started_at = 0;
        uint64_t ended_at = 0;
        unsigned count = 0;

        size_t host_offset = gpu_id * 1000000000UL;
        size_t device_offset = gpu_id * 1000000000UL;
        size_t increment = 1027392767UL * gpu_id;

        countdown_begin--;
        while (countdown_end > 0) {
            if (started_at == 0 && countdown_begin == 0) {
                started_at = util::now();
                std::cout << "Started " << gpu_id << std::endl;
            }

            host_offset += increment;
            if (host_offset > host_size) host_offset -= host_size;
            if (host_offset + transfer_size > host_size) {
                host_offset += increment;
                host_offset -= host_size;
            }
            void* hostptr = host + host_offset;

            device_offset += increment;
            if (device_offset > device_size) device_offset -= device_size;
            if (device_offset + transfer_size > device_size) {
                device_offset += increment;
                device_offset -= device_size;
            }
            void* deviceptr = device + device_offset;

            status = cudaMemcpyAsync(hostptr, deviceptr, transfer_size, cudaMemcpyHostToDevice, stream);
            if (!successful()) { countdown_end--; return; };

            status = cudaStreamSynchronize(stream);
            if (!successful()) { countdown_end--; return; };

            if (alive) {
                count++;
            } else if (ended_at == 0) {
                count++;
                ended_at = util::now();
                std::cout << "Ended " << gpu_id << std::endl;
                countdown_end--;
            }
        }

        size_t throughput = 1000UL * (count * transfer_size) / ((ended_at - started_at) / 1000000UL);

        std::cout << "Exited " << gpu_id << " with throughput " << throughput << std::endl;     
    }
};

TEST_CASE("Profile concurrent transfers on single GPU", "[singlegpu]") {
    cudaError_t status = cudaSetDevice(0);
    REQUIRE(status == cudaSuccess);

    size_t host_size = 10000000000UL; // 10GB host side
    void* host;

    status = cudaMallocHost(&host, host_size);
    REQUIRE(status == cudaSuccess);

    std::atomic_int countdown_begin(1);
    std::atomic_int countdown_end(1);
    std::atomic_bool alive(true);

    auto cores0 = threading::getGPUCoreAffinity(0);

    TransferThread t0(0, cores0, countdown_begin, countdown_end, alive, host, host_size);

    usleep(50000000UL);
    alive = false;

    t0.thread.join();

    std::cout << "Done" << std::endl;

    REQUIRE(t0.success);

    // util::initializeCudaStream(0);
    // size_t total_malloced = 0;
    // size_t peak_usage = 0;
    // fill_memory(total_malloced, peak_usage);
    // std::cout << "cudaMalloc total=" << total_malloced << " plus " << (peak_usage - total_malloced) << " additional" << std::endl;

    // void* ptr;
    // cudaError_t status;
    // status = cudaMalloc(&ptr, total_malloced);
    // REQUIRE(status == cudaSuccess);
    // status = cudaFree(ptr);
    // REQUIRE(status == cudaSuccess);
}

TEST_CASE("Profile concurrent transfers on same GPU", "[samegpu]") {
    cudaError_t status = cudaSetDevice(0);
    REQUIRE(status == cudaSuccess);

    size_t host_size = 10000000000UL; // 10GB host side
    void* host;

    status = cudaMallocHost(&host, host_size);
    REQUIRE(status == cudaSuccess);

    std::atomic_int countdown_begin(2);
    std::atomic_int countdown_end(2);
    std::atomic_bool alive(true);

    auto cores0 = threading::getGPUCoreAffinity(0);

    TransferThread t0(0, cores0, countdown_begin, countdown_end, alive, host, host_size);
    TransferThread t1(0, cores0, countdown_begin, countdown_end, alive, host, host_size);

    usleep(10000000UL);
    alive = false;

    t0.thread.join();
    t1.thread.join();

    std::cout << "Done" << std::endl;

    REQUIRE(t0.success);

    // util::initializeCudaStream(0);
    // size_t total_malloced = 0;
    // size_t peak_usage = 0;
    // fill_memory(total_malloced, peak_usage);
    // std::cout << "cudaMalloc total=" << total_malloced << " plus " << (peak_usage - total_malloced) << " additional" << std::endl;

    // void* ptr;
    // cudaError_t status;
    // status = cudaMalloc(&ptr, total_malloced);
    // REQUIRE(status == cudaSuccess);
    // status = cudaFree(ptr);
    // REQUIRE(status == cudaSuccess);
}

TEST_CASE("Profile concurrent transfers on multiple GPUs", "[multigpu]") {
    cudaError_t status = cudaSetDevice(0);
    REQUIRE(status == cudaSuccess);

    size_t host_size = 10000000000UL; // 10GB host side
    void* host;

    status = cudaMallocHost(&host, host_size);
    REQUIRE(status == cudaSuccess);

    std::atomic_int countdown_begin(2);
    std::atomic_int countdown_end(2);
    std::atomic_bool alive(true);

    auto cores0 = threading::getGPUCoreAffinity(0);
    auto cores1 = threading::getGPUCoreAffinity(1);

    TransferThread t0(0, cores0, countdown_begin, countdown_end, alive, host, host_size);
    TransferThread t1(1, cores1, countdown_begin, countdown_end, alive, host, host_size);

    usleep(50000000UL);
    alive = false;

    t0.thread.join();
    t1.thread.join();

    std::cout << "Done" << std::endl;

    REQUIRE(t0.success);
    REQUIRE(t1.success);

    // util::initializeCudaStream(0);
    // size_t total_malloced = 0;
    // size_t peak_usage = 0;
    // fill_memory(total_malloced, peak_usage);
    // std::cout << "cudaMalloc total=" << total_malloced << " plus " << (peak_usage - total_malloced) << " additional" << std::endl;

    // void* ptr;
    // cudaError_t status;
    // status = cudaMalloc(&ptr, total_malloced);
    // REQUIRE(status == cudaSuccess);
    // status = cudaFree(ptr);
    // REQUIRE(status == cudaSuccess);
}

TEST_CASE("Host-side multi-gpu malloc", "[hostmalloc]") {
    cudaError_t status;

    status = cudaSetDevice(0);
    REQUIRE(status == cudaSuccess);

    size_t size = 100 * 1024 * 1024;

    void* host1;
    status = cudaMallocHost(&host1, size);
    REQUIRE(status == cudaSuccess);

    void* device1;
    status = cudaMalloc(&device1, size);
    REQUIRE(status == cudaSuccess);

    cudaStream_t stream1;  
    status =cudaStreamCreateWithPriority(&stream1, cudaStreamNonBlocking, 0);
    REQUIRE(status == cudaSuccess);

    status = cudaSetDevice(1);
    REQUIRE(status == cudaSuccess);

    void* host2;
    status = cudaMallocHost(&host2, size);
    REQUIRE(status == cudaSuccess);

    void* device2;
    status = cudaMalloc(&device2, size);
    REQUIRE(status == cudaSuccess);

    cudaStream_t stream2;  
    status =cudaStreamCreateWithPriority(&stream2, cudaStreamNonBlocking, 0);
    REQUIRE(status == cudaSuccess);


    status = cudaSetDevice(0);
    REQUIRE(status == cudaSuccess);
    
    status = cudaMemcpyAsync(host1, device1, size, cudaMemcpyHostToDevice, stream1);
    REQUIRE(status == cudaSuccess);

    status = cudaStreamSynchronize(stream1);
    REQUIRE(status == cudaSuccess);    
    
    status = cudaSetDevice(1);
    REQUIRE(status == cudaSuccess);

    status = cudaMemcpyAsync(host2, device2, size, cudaMemcpyHostToDevice, stream2);
    REQUIRE(status == cudaSuccess);

    status = cudaStreamSynchronize(stream2);
    REQUIRE(status == cudaSuccess);



    status = cudaSetDevice(0);
    REQUIRE(status == cudaSuccess);
    
    status = cudaMemcpyAsync(host2, device1, size, cudaMemcpyHostToDevice, stream1);
    REQUIRE(status == cudaSuccess);

    status = cudaStreamSynchronize(stream1);
    REQUIRE(status == cudaSuccess);

    status = cudaSetDevice(1);
    REQUIRE(status == cudaSuccess);
    
    status = cudaMemcpyAsync(host1, device2, size, cudaMemcpyHostToDevice, stream2);
    REQUIRE(status == cudaSuccess);

    status = cudaStreamSynchronize(stream2);
    REQUIRE(status == cudaSuccess);



    status = cudaSetDevice(0);
    REQUIRE(status == cudaSuccess);
    
    status = cudaMemcpyAsync(host2, device1, size, cudaMemcpyHostToDevice, stream2);
    REQUIRE(status == cudaSuccess);

    status = cudaStreamSynchronize(stream2);
    REQUIRE(status == cudaSuccess);

    status = cudaSetDevice(1);
    REQUIRE(status == cudaSuccess);
    
    status = cudaMemcpyAsync(host1, device2, size, cudaMemcpyHostToDevice, stream1);
    REQUIRE(status == cudaSuccess);

    status = cudaStreamSynchronize(stream1);
    REQUIRE(status == cudaSuccess);



    status = cudaSetDevice(0);
    REQUIRE(status == cudaSuccess);
    
    status = cudaMemcpyAsync(host2, device2, size, cudaMemcpyHostToDevice, stream1);
    REQUIRE(status == cudaSuccess);

    status = cudaStreamSynchronize(stream1);
    REQUIRE(status == cudaSuccess);

    status = cudaSetDevice(1);
    REQUIRE(status == cudaSuccess);
    
    status = cudaMemcpyAsync(host1, device1, size, cudaMemcpyHostToDevice, stream2);
    REQUIRE(status == cudaSuccess);

    status = cudaStreamSynchronize(stream2);
    REQUIRE(status == cudaSuccess);


    status = cudaSetDevice(0);
    REQUIRE(status == cudaSuccess);
    
    status = cudaMemcpyAsync(host2, device1, size, cudaMemcpyHostToDevice, stream1);
    REQUIRE(status == cudaSuccess);

    status = cudaSetDevice(1);
    REQUIRE(status == cudaSuccess);
    
    status = cudaMemcpyAsync(host1, device2, size, cudaMemcpyHostToDevice, stream1);
    REQUIRE(status == cudaSuccess);

    status = cudaStreamSynchronize(stream1);
    REQUIRE(status == cudaSuccess);
}

TEST_CASE("Profile memory limit for cudaMalloc", "[profile] [cudaMalloc]") {
    util::initializeCudaStream(0);
    size_t total_malloced = 0;
    size_t peak_usage = 0;
    fill_memory(total_malloced, peak_usage);
    std::cout << "cudaMalloc total=" << total_malloced << " plus " << (peak_usage - total_malloced) << " additional" << std::endl;

    void* ptr;
    cudaError_t status;
    status = cudaMalloc(&ptr, total_malloced);
    REQUIRE(status == cudaSuccess);
    status = cudaFree(ptr);
    REQUIRE(status == cudaSuccess);
}

void profile_model(std::string model_name, std::string model_path, int expected_blob_size) {
    std::string so_filename = model_path + ".1.so";
    so::SharedObject so(so_filename);
    
    const char* cuda_blob = reinterpret_cast<const char*>(so.GetSymbol(tvm::runtime::symbol::tvm_dev_mblob));
    REQUIRE(cuda_blob != nullptr);

    cudaError_t status;
    util::initializeCudaStream(0);

    size_t free, total;
    status = cudaMemGetInfo(&free, &total);
    REQUIRE(status == cudaSuccess);
    size_t initial_use = total-free;

    std::cout << "Profiling " << model_name << " with initial memory used=" << (total-free) << std::endl;

    std::vector<cuda::UnloadedCUDAModule*> unloaded;
    std::vector<cuda::LoadedCUDAModule*> loaded;

    int maxIterations = 1000000;
    for (unsigned i = 0; i < maxIterations; i++) {
        cuda::UnloadedCUDAModule* unloaded_cuda = new cuda::UnloadedCUDAModule(cuda_blob);
        REQUIRE(unloaded_cuda->data.size() == expected_blob_size);

        CUmodule module;
        CUresult result = cuModuleLoadFatBinary(&module, unloaded_cuda->data.c_str());
        REQUIRE((result == CUDA_SUCCESS || result == CUDA_ERROR_DEINITIALIZED || result == CUDA_ERROR_OUT_OF_MEMORY));

        if (result == CUDA_ERROR_OUT_OF_MEMORY) {
            size_t free, total;
            status = cudaMemGetInfo(&free, &total);
            REQUIRE(status == cudaSuccess);
            std::cout << model_name << ": limit at n=" << i << ", size=" << unloaded_cuda->data.size() << ", total used=" << (total-free) << ", average=" << ((total-free)/i) << std::endl;
            break;
        }

        cuda::LoadedCUDAModule* loaded_cuda = new cuda::LoadedCUDAModule(unloaded_cuda, module);

        unloaded.push_back(unloaded_cuda);
        loaded.push_back(loaded_cuda);

        if (i % 1000 == 0) {
            status = cudaMemGetInfo(&free, &total);
            REQUIRE(status == cudaSuccess);
            std::cout << " ... " << model_name << " iteration " << i << " memory used is " << (total-free) << std::endl;
        }
    }

    // See how much more memory we can malloc
    size_t total_malloced = 0;
    size_t peak_usage = 0;
    fill_memory(total_malloced, peak_usage);
    std::cout << "cudaMalloc additional total=" << total_malloced << " for peak usage of " << peak_usage << std::endl;;

    for (auto &l : loaded) {
        l->unload();
    }
    for (auto &u : unloaded) {
        delete u;
    }

    status = cudaMemGetInfo(&free, &total);
    REQUIRE(status == cudaSuccess);
    REQUIRE( (total-free) == initial_use );
}

TEST_CASE("Profile memory limit for cuModuleLoad - resnet18", "[profile] [resnet18] [cuModuleLoad]") {
    std::string model_name = "resnet18";
    std::string model_path = clockwork::util::get_example_model("resnet18_tesla-m40_batchsize1");
    // TODO: the current resnet18 example model is NOT a fatbin and does JIT compilation
    int expected_blob_size = 681463;
    profile_model(model_name, model_path, expected_blob_size);
}

TEST_CASE("Profile memory limit for cuModuleLoad - resnet50", "[profile] [resnet50] [cuModuleLoad]") {
    std::string model_name = "resnet50";
    std::string model_path = clockwork::util::get_example_model("resnet50_tesla-m40_batchsize1");
    int expected_blob_size = 403360;
    profile_model(model_name, model_path, expected_blob_size);
}