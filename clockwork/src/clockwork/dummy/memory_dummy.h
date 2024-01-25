#ifndef _CLOCKWORK_MEMORY_DUMMY_H_
#define _CLOCKWORK_MEMORY_DUMMY_H_

#include <atomic>
#include <unordered_map>
#include "clockwork/api/worker_api.h"
#include "clockwork/config.h"

namespace clockwork {

class RuntimeModelDummy {
public:
    unsigned gpu_id;
    workerapi::ModelInfo* modelinfo;
    std::atomic_flag in_use;
    int version;
    bool weights;
    unsigned weightspagescount;

    RuntimeModelDummy(workerapi::ModelInfo* Modelinfo, unsigned gpu_id, unsigned weights_pages_count);

    bool try_lock();
    void lock();
    void unlock();
    int padded_batch_size_index(int batch_size); // return the index for padded batch size if batch_size is legal, return -1 otherwise, This index is the same as the index for batch_size_exec_times_nanos
    size_t input_size(unsigned batch_size);
    size_t output_size(unsigned batch_size); 

};

class ModelStoreDummy {
public:
    std::atomic_flag in_use;
    std::unordered_map<std::pair<int, unsigned>, RuntimeModelDummy*, util::hash_pair> models;

    ModelStoreDummy();

    // This will delete all models that are in the ModelStoreDummy
    ~ModelStoreDummy();

    RuntimeModelDummy* get(int model_id, unsigned gpu_id);
    bool contains(int model_id, unsigned gpu_id);
    void put(int model_id, unsigned gpu_id, RuntimeModelDummy* model);
    bool put_if_absent(int model_id, unsigned gpu_id, RuntimeModelDummy* model);
    void get_model_info(clockwork::workerapi::WorkerMemoryInfo &worker_memory_info);
    void clearWeights();

};
class PageCacheDummy{
public:
    std::atomic_flag in_use;
    const size_t size, page_size;
    const unsigned total_pages;
    unsigned n_free_pages;

    PageCacheDummy(size_t total_size, size_t page_size);


    bool try_lock();
    void lock();
    void unlock();
    bool alloc(unsigned n_pages); // Alloc n_pages from n_free_pages, fail if no enough pages available
    void free(unsigned n_pages);// free n_pages and add them to  n_free_pages
    void clear(); // Reclaim back all pages
};
class MemoryManagerDummy {
public:
    // Used for testing; Clockwork can be configured to generate model inputs server-side
    bool allow_zero_size_inputs = false;

    const size_t page_size;

    // Device-side GPU-specific page cache for model weights
    std::vector<PageCacheDummy*> weights_caches;

    // TODO: host-side weights cache


    // Device-side GPU-specific memory pools for inference inputs and outputs
    const size_t io_pool_size;

    // Device-side GPU-specific memory pools for inference workspace
    const size_t workspace_pool_size;

    // Host-side memory pool for inference inputs and outputs
    const size_t host_io_pool_size;
    

    ModelStoreDummy* models; // Models

    unsigned num_gpus;

    MemoryManagerDummy(ClockworkWorkerConfig &config);
    ~MemoryManagerDummy();

    void get_worker_memory_info(clockwork::workerapi::WorkerMemoryInfo &worker_memory_info);
};

class NoMeasureFile {
public:
    int status_code;
    std::string message;
    NoMeasureFile(int status_code, std::string message) : status_code(status_code), message(message) {}
};

struct ModelDataDummy {
    unsigned batch_size;
    std::string serialized_spec;
    uint64_t exec_measurement;
    uint64_t weights_measurement;
};

std::vector<ModelDataDummy> loadModelDataDummy(std::string base_filename);

}

#endif