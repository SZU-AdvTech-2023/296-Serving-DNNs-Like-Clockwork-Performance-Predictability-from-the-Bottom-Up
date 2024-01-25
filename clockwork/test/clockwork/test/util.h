#ifndef _CLOCKWORK_TEST_UTIL_H_
#define _CLOCKWORK_TEST_UTIL_H_

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <utility>
#include <memory>

namespace clockwork{
namespace util {

std::string get_exe_location();
std::string get_clockwork_dir();
std::string get_example_model(std::string name = "resnet18_tesla-m40_batchsize1"); 
std::string get_example_batched_model(std::string name = "resnet18_tesla-m40"); 

bool is_cuda_cache_disabled();
bool is_force_ptx_jit_enabled();

bool is_gpu_exclusive(int deviceId);
bool is_persistence_mode_enabled_on_gpu(int deviceId);
std::pair<int, int> get_compute_capability(unsigned device_id);
void nvml();

}

namespace model {

class cuda_page_alloc {
private:
	void* baseptr;
public:
	char* ptr;
	std::vector<char*> pages;

	cuda_page_alloc(int page_size, int num_pages);
	~cuda_page_alloc();
};

std::shared_ptr<cuda_page_alloc> make_cuda_pages(int page_size, int num_pages);

void cuda_synchronize(cudaStream_t stream);

}
}

#endif