#include "clockwork/thread.h"

#include <cuda_runtime.h>
#include <nvml.h>
#include <thread>
#include <algorithm>
#include <sstream>
#include <sched.h>

#include <dmlc/logging.h>

#include "clockwork/cuda_common.h"
#include "clockwork/util.h"

namespace clockwork {
namespace threading {

// The priority scheduler in use.  SCHED_FIFO or SCHED_RR
int scheduler = SCHED_FIFO;

/* 
Globally manages assignment of threads to cores, since RT priority is fragile
*/
class CoreManager {
public:
	const int init_pool_size = 2;
	const int default_pool_size = 2;

	std::vector<bool> in_use;
	std::vector<std::vector<unsigned>> gpu_affinity;

	std::vector<unsigned> init_pool;
	std::vector<unsigned> default_pool;

	CoreManager() : in_use(coreCount(), false) {
		for (int i = 0; i < default_pool_size; i++) {
			in_use[i] = true;
			default_pool.push_back(i);
		}

		for (int i = 0; i < init_pool_size; i++) {
			int core = i + default_pool_size;
			in_use[core] = true;
			init_pool.push_back(core);
		}

		unsigned gpu_count = util::get_num_gpus();
		for (unsigned i = 0; i < gpu_count; i++) {
			gpu_affinity.push_back(getGPUCoreAffinity(i));
		}
	}

	unsigned alloc(unsigned gpu_id) {
		if (gpu_id < gpu_affinity.size()) {
			for (unsigned i = 0; i < gpu_affinity[gpu_id].size(); i++) {
				unsigned core = gpu_affinity[gpu_id][i];
				if (!in_use[core]) {
					in_use[core] = true;
					return core;
				}
			}
		}
		// Couldn't get a core with GPU affinity; get a different core
		for (unsigned i = 0; i < in_use.size(); i++) {
			if (!in_use[i]) {
				in_use[i] = true;
				return i;
			}
		}
		CHECK(false) << "All cores exhausted for GPU " << gpu_id;
		return 0;
	}

	std::vector<unsigned> alloc(unsigned count, unsigned gpu_id) {
		// std::cout << "Alloc " << count << " on " << gpu_id << " " << str() << std::endl;
		std::vector<unsigned> result;
		for (unsigned i = 0; i < count; i++) {
			result.push_back(alloc(gpu_id));
		}
		return result;
	}

	std::string str() {
		unsigned allocated = 0;
		for (unsigned i = 0; i < in_use.size(); i++) {
			allocated += in_use[i];
		}
		std::stringstream ss;
		ss << (in_use.size() - allocated) << "/" << in_use.size() << " cores free";
		return ss.str();
	}

};


bool init = false;
CoreManager manager;

// Initializes a clockwork process
void initProcess() {
	// Bind to the init pool and set priority to max
	setCores(manager.init_pool, pthread_self());
	setPriority(scheduler, maxPriority(scheduler), pthread_self());
	init = true;
}

void initHighPriorityThread(int num_cores, int gpu_affinity, std::thread &thread) {
	if (init) {
		auto cores = manager.alloc(num_cores, gpu_affinity);
		setCores(cores, thread.native_handle());
		setPriority(scheduler, maxPriority(scheduler), thread.native_handle());
	} else {
		std::cout << "Warning: trying to initialize high priority thread without threading initialized" << std::endl;
	}
}

void initHighPriorityThread(std::thread &thread) {
	initHighPriorityThread(1, 1, thread);
}

void initHighPriorityThread(std::thread &thread, int num_cores) {
	initHighPriorityThread(num_cores, 1, thread);
}

void initLowPriorityThread(std::thread &thread) {
	setCores(manager.default_pool, thread.native_handle());
	setDefaultPriority(thread.native_handle());
}

void initNetworkThread(std::thread &thread) {
	initHighPriorityThread(1, 0, thread);
}

void initLoggerThread(std::thread &thread) {
	initLowPriorityThread(thread);
}

void initGPUThread(int gpu_id, std::thread &thread) {
	initHighPriorityThread(2, 0, thread);
}

unsigned coreCount() {
	return std::thread::hardware_concurrency();	
}

void setCore(unsigned core) {
	cpu_set_t cpuset;
	CPU_ZERO(&cpuset);
	CPU_SET(core, &cpuset);
	int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
	CHECK(rc == 0) << "Unable to set thread affinity: " << rc;
}

void setCores(std::vector<unsigned> cores, pthread_t thread) {
	CHECK(cores.size() > 0) << "Trying to bind to empty core set";
	cpu_set_t cpuset;
	CPU_ZERO(&cpuset);
	for (unsigned core : cores) {
		CPU_SET(core, &cpuset);
	}
	int rc = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
	CHECK(rc == 0) << "Unable to set thread affinity: " << rc;
}

void setAllCores() {
	cpu_set_t cpuset;
	CPU_ZERO(&cpuset);
	for (unsigned i = 0; i < coreCount(); i++) {
		CPU_SET(i, &cpuset);
	}
	int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
	CHECK(rc == 0) << "Unable to set thread affinity: " << rc;
}

void addCore(unsigned core) {
	auto cores = currentCores();
	cores.push_back(core);
	setCores(cores, pthread_self());
}

void removeCore(unsigned core) {
	auto cores = currentCores();
	auto it = std::remove(cores.begin(), cores.end(), core);
	cores.erase(it, cores.end());
	setCores(cores, pthread_self());
}

std::vector<unsigned> currentCores() {
	cpu_set_t cpuset;
	int rc = pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
	CHECK(rc == 0) << "Unable to get thread affinity: " << rc;
	std::vector<unsigned> cores;
	for (unsigned i = 0; i < coreCount(); i++) {
		if (CPU_ISSET(i, &cpuset) > 0) {
			cores.push_back(i);
		}
	}
	return cores;
}

unsigned getCurrentCore() {
	return sched_getcpu();
}

std::vector<unsigned> getGPUCoreAffinity(unsigned gpu_id) {
	unsigned len = (coreCount() + 63) / 64;

	std::vector<uint64_t> bitmaps(len);

	nvmlReturn_t status;

	status = nvmlInit();
	CHECK(status == NVML_SUCCESS);

	nvmlDevice_t device;
	status = nvmlDeviceGetHandleByIndex(gpu_id, &device);
	CHECK(status == NVML_SUCCESS);

	// Fill bitmaps with the ideal CPU affinity for the device
	// (see https://helpmanual.io/man3/nvmlDeviceGetCpuAffinity/)
	status = nvmlDeviceGetCpuAffinity(device, bitmaps.size(), bitmaps.data());
	CHECK(status == NVML_SUCCESS);

	std::vector<unsigned> cores;

	unsigned core = 0;
	for (unsigned i = 0; i < bitmaps.size(); i++) {
		for (unsigned j = 0; j < 64; j++) {
			if (((bitmaps[i] >> j) & 0x01) == 0x01) {
				cores.push_back(core);
			}
		core++;
		}
	}

	status = nvmlShutdown();
	CHECK(status == NVML_SUCCESS);

	return cores;
}

int minPriority(int scheduler) {
	return sched_get_priority_min(scheduler);
}

int maxPriority(int scheduler) {
	return 49; // 1 less than interrupt priority
	// return sched_get_priority_max(scheduler);
}

void setDefaultPriority() {
	setDefaultPriority(pthread_self());
}

void setDefaultPriority(pthread_t thread) {
	setPriority(SCHED_OTHER, 0, thread);
}

void setMaxPriority() {
	setPriority(SCHED_FIFO, maxPriority(SCHED_FIFO), pthread_self());
}

void setPriority(int scheduler, int priority, pthread_t thId) {
	struct sched_param params;
	params.sched_priority = sched_get_priority_max(scheduler);
	int ret = pthread_setschedparam(thId, scheduler, &params);
	CHECK(ret == 0) << "Unable to set thread priority.  Don't forget to set `rtprio` to unlimited in `limits.conf`.  See Clockwork README for instructions";

	int policy = 0;
	ret = pthread_getschedparam(thId, &policy, &params);
	CHECK(ret == 0) << "Unable to verify thread scheduler params";
	CHECK(policy == scheduler) << "Unable to verify thread scheduler params";
}

int currentScheduler() {
	pthread_t thId = pthread_self();

	struct sched_param params;
	int policy = 0;
	int ret = pthread_getschedparam(thId, &policy, &params);
	CHECK(ret == 0) << "Unable to get current thread scheduler params";

	return policy;	
}

int currentPriority() {
	pthread_t thId = pthread_self();

	struct sched_param params;
	int policy = 0;
	int ret = pthread_getschedparam(thId, &policy, &params);
	CHECK(ret == 0) << "Unable to get current thread scheduler params";

	return params.sched_priority;
}


}
}