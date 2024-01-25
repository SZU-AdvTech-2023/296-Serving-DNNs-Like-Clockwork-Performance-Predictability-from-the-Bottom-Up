#include <cstdlib>
#include <unistd.h>
#include <libgen.h>
#include "clockwork/test/util.h"
#include <catch2/catch.hpp>
#include <nvml.h>
#include <iostream>

namespace clockwork{
namespace util {

std::string get_exe_location() {
    int bufsize = 1024;
    char buf[bufsize];
    int len = readlink("/proc/self/exe", buf, bufsize-1);
    buf[len] = '\0';
    return std::string(buf, len);
}

std::string get_clockwork_dir() {
    int bufsize = 1024;
    char buf[bufsize];
    int len = readlink("/proc/self/exe", buf, bufsize-1);
    buf[len] = '\0';
	return dirname(dirname(buf));
}

std::string get_example_model(std::string name) {
    return get_clockwork_dir() + "/resources/" + name + "/model";
}

std::string get_example_batched_model(std::string name) {
    return get_example_model(name);
}

bool is_cuda_cache_disabled() {
    const char* v = std::getenv("CUDA_CACHE_DISABLE");
    if (v != nullptr) return std::string(v) == "1";
    return false;
}

bool is_force_ptx_jit_enabled() {
    const char* v = std::getenv("CUDA_FORCE_PTX_JIT");
    if (v != nullptr) return std::string(v) == "1";
    return false;
}

bool is_gpu_exclusive(int deviceId) {
    nvmlReturn_t status;

    status = nvmlInit();
    CHECK(status == NVML_SUCCESS);

    nvmlDevice_t device;
    status = nvmlDeviceGetHandleByIndex(deviceId, &device);
    CHECK(status == NVML_SUCCESS);

    nvmlComputeMode_t computeMode;
    status = nvmlDeviceGetComputeMode(device, &computeMode);
    CHECK(status == NVML_SUCCESS);

    status = nvmlShutdown();
    CHECK(status == NVML_SUCCESS);

    return computeMode == NVML_COMPUTEMODE_EXCLUSIVE_PROCESS;
}

bool is_persistence_mode_enabled_on_gpu(int deviceId) {
    nvmlReturn_t status;

    status = nvmlInit();
    CHECK(status == NVML_SUCCESS);

    nvmlDevice_t device;
    status = nvmlDeviceGetHandleByIndex(deviceId, &device);
    CHECK(status == NVML_SUCCESS);

    nvmlEnableState_t mode;
    status = nvmlDeviceGetPersistenceMode(device, &mode);
    CHECK(status == NVML_SUCCESS);

    status = nvmlShutdown();
    CHECK(status == NVML_SUCCESS);

    return mode == NVML_FEATURE_ENABLED;
}

std::pair<int, int> get_compute_capability(unsigned device_id) {
    nvmlReturn_t status;

    status = nvmlInit();
    CHECK(status == NVML_SUCCESS);

    nvmlDevice_t device;
    status = nvmlDeviceGetHandleByIndex(device_id, &device);
    CHECK(status == NVML_SUCCESS);

    int major, minor;
    status = nvmlDeviceGetCudaComputeCapability(device, &major, &minor);

    status = nvmlShutdown();
    CHECK(status == NVML_SUCCESS);

    return std::make_pair(major, minor);
}

void nvml() {
    nvmlReturn_t status;

    status = nvmlInit();
    CHECK(status == NVML_SUCCESS);

    nvmlDevice_t device;
    status = nvmlDeviceGetHandleByIndex(0, &device);
    CHECK(status == NVML_SUCCESS);
    std::cout << " got device " << 0 << std::endl;

    nvmlComputeMode_t computeMode;
    status = nvmlDeviceGetComputeMode(device, &computeMode);
    CHECK(status == NVML_SUCCESS);
    std::cout << "compute mode is " << computeMode << std::endl;

    unsigned linkWidth;
    status = nvmlDeviceGetCurrPcieLinkWidth(device, &linkWidth);
    CHECK(status == NVML_SUCCESS);
    std::cout << "link width is " << linkWidth << std::endl;

    int major, minor;
    status = nvmlDeviceGetCudaComputeCapability(device, &major, &minor);
    std::cout << "Compute " << major << " " << minor << std::endl;

    status = nvmlShutdown();
    CHECK(status == NVML_SUCCESS);
}

}

namespace model {


cuda_page_alloc::cuda_page_alloc(int page_size, int num_pages) {
    cudaError_t status = cudaMalloc(&baseptr, page_size * num_pages);
    REQUIRE(status == cudaSuccess);
    ptr = static_cast<char*>(baseptr);

    for (unsigned i = 0; i < num_pages; i++) {
        pages.push_back(static_cast<char*>(ptr) + (i * page_size));
    }
}

cuda_page_alloc::~cuda_page_alloc() {
    cudaError_t status = cudaFree(baseptr);
    REQUIRE(status == cudaSuccess);
}

std::shared_ptr<cuda_page_alloc> make_cuda_pages(int page_size, int num_pages) {
    return std::make_shared<cuda_page_alloc>(page_size, num_pages);
}

void cuda_synchronize(cudaStream_t stream) {
	cudaError_t status = cudaStreamSynchronize(stream);
	REQUIRE(status == cudaSuccess);
}

}
}