#include "clockwork/dummy/memory_dummy.h"
#include <exception>
#include <libconfig.h++>
#include <algorithm>

#include <iostream>


namespace clockwork {

RuntimeModelDummy::RuntimeModelDummy(workerapi::ModelInfo* Modelinfo, unsigned gpu_id, unsigned weights_pages_count):
    modelinfo(Modelinfo), gpu_id(gpu_id), in_use(ATOMIC_FLAG_INIT), version(0), weights(false),weightspagescount(weights_pages_count) {
}

bool RuntimeModelDummy::try_lock() {
    return !in_use.test_and_set();
}

void RuntimeModelDummy::lock() {
    while (!try_lock());
}

void RuntimeModelDummy::unlock() {
    in_use.clear();
}

int RuntimeModelDummy::padded_batch_size_index(int batch_size){
    if( batch_size <= 0 || modelinfo->supported_batch_sizes.size() == 0)
        return -1;
    if(batch_size > modelinfo->supported_batch_sizes[modelinfo->supported_batch_sizes.size()-1] ||batch_size < modelinfo->supported_batch_sizes[0])
        return -1;
    int index = 0;
    for(unsigned size: modelinfo->supported_batch_sizes){
        if(batch_size <= int(size)){
            return index;
        }
        index++;
    }
    return -1;
}

size_t RuntimeModelDummy::input_size(unsigned batch_size){
    return modelinfo->input_size*batch_size;
}
size_t RuntimeModelDummy::output_size(unsigned batch_size){
    return modelinfo->output_size*batch_size;
}

PageCacheDummy::PageCacheDummy(size_t total_size, size_t page_size):in_use(ATOMIC_FLAG_INIT), page_size(page_size),total_pages(total_size/page_size),n_free_pages(total_size/page_size),size(total_size){}

bool PageCacheDummy::try_lock() {
    return !in_use.test_and_set();
}

void PageCacheDummy::lock() {
    while (!try_lock());
}

void PageCacheDummy::unlock() {
    in_use.clear();
}

bool PageCacheDummy::alloc(unsigned n_pages) {
    bool alloc_success = false;
    this->lock();
    if (n_pages <= n_free_pages) {
        n_free_pages -= n_pages;
        alloc_success = true;
    }
    this->unlock();
    return alloc_success;
}

void PageCacheDummy::free(unsigned n_pages) {
    this->lock();
    if(n_free_pages + n_pages <= total_pages){
        n_free_pages += n_pages;
    }
    this->unlock();
}

void PageCacheDummy::clear() {
    this->lock();
    n_free_pages = total_pages;
    this->unlock();
}

ModelStoreDummy::ModelStoreDummy() : in_use(ATOMIC_FLAG_INIT) {}

ModelStoreDummy::~ModelStoreDummy() {
    while (in_use.test_and_set());

    for (auto &p : models) {
        RuntimeModelDummy* rm = p.second;
        if (rm != nullptr) {
            // Do we want to delete models here? Probably?
            delete rm->modelinfo;
            delete rm;
        }
    }

    // Let callers hang here to aid in use-after-free
    // in_use.clear();
}

RuntimeModelDummy* ModelStoreDummy::get(int model_id, unsigned gpu_id) {
    while (in_use.test_and_set());

    std::unordered_map<std::pair<int, unsigned>, RuntimeModelDummy*, util::hash_pair>::iterator got = models.find(std::make_pair(model_id, gpu_id));

    RuntimeModelDummy* rm = nullptr;

    if ( got != models.end() )
        rm = got->second;

    in_use.clear();

    return rm;
}

bool ModelStoreDummy::contains(int model_id, unsigned gpu_id) {
    while (in_use.test_and_set());

    bool did_contain = true;

    std::unordered_map<std::pair<int, unsigned>, RuntimeModelDummy*, util::hash_pair>::iterator got = models.find(std::make_pair(model_id, gpu_id));

    if ( got == models.end() )
        did_contain = false;

    in_use.clear();

    return did_contain;
}

void ModelStoreDummy::put(int model_id, unsigned gpu_id, RuntimeModelDummy* model) {
    while (in_use.test_and_set());

    models[std::make_pair(model_id, gpu_id)] = model;

    in_use.clear();
}

bool ModelStoreDummy::put_if_absent(int model_id, unsigned gpu_id, RuntimeModelDummy* model) {
    while (in_use.test_and_set());

    bool did_put = false;
    std::pair<int, unsigned> key = std::make_pair(model_id, gpu_id);
    std::unordered_map<std::pair<int, unsigned>, RuntimeModelDummy*, util::hash_pair>::iterator got = models.find(key);

    if ( got == models.end() ){
        models[key] = model;
        did_put = true;
    }

    in_use.clear();

    return did_put;
}

void ModelStoreDummy::get_model_info(workerapi::WorkerMemoryInfo &info) {
    while (in_use.test_and_set());

    std::map<int, workerapi::ModelInfo> models_info;

    for (auto p : models) {
        int model_id = p.first.first;
        unsigned gpu_id = p.first.second;
        RuntimeModelDummy* rm = p.second;

        auto it = models_info.find(model_id);
        if (it == models_info.end()) {
            models_info[model_id] = *rm->modelinfo;
        }

        // Also store which models are loaded
        if ( rm->weights ) {
            info.gpus[gpu_id].models.push_back(model_id);
        }
    }

    // Add models to model info
    for (auto &p : models_info) {
        info.models.push_back(p.second);
    }

    // Sort model ids on GPU
    for (unsigned i = 0; i < info.gpus.size(); i++) {
        std::sort(info.gpus[i].models.begin(), info.gpus[i].models.end());
    }

    in_use.clear();
}

void ModelStoreDummy::clearWeights(){
    for (std::unordered_map<std::pair<int, unsigned>, RuntimeModelDummy*, util::hash_pair>::iterator got = models.begin(); got != models.end(); ++got){
        RuntimeModelDummy* rm = got->second;
        rm->lock();
        rm->weights = false;
        rm->version++;
        rm->unlock();
    }
}

MemoryManagerDummy::MemoryManagerDummy(ClockworkWorkerConfig &config) :
            host_io_pool_size(config.host_io_pool_size),
            models(new ModelStoreDummy()),
            num_gpus(config.num_gpus),
            page_size(config.weights_cache_page_size),workspace_pool_size(config.workspace_pool_size),io_pool_size(config.io_pool_size){

    for (unsigned gpu_id = 0; gpu_id < config.num_gpus; gpu_id++) {
        weights_caches.push_back(new PageCacheDummy(config.weights_cache_size,config.weights_cache_page_size));
    }
    allow_zero_size_inputs = config.allow_zero_size_inputs;
}

MemoryManagerDummy::~MemoryManagerDummy() {
    delete models;
}

void MemoryManagerDummy::get_worker_memory_info(workerapi::WorkerMemoryInfo &info) {
    // Store basic info
    info.page_size = page_size;
    info.host_weights_cache_size = ULONG_MAX; // Not currently fixed
    info.host_io_pool_size = host_io_pool_size;

    // Store GPU info
    for (unsigned i = 0; i < num_gpus; i++) {
        workerapi::GPUInfo gpu;
        gpu.id = i;
        gpu.weights_cache_size = weights_caches[i]->size;
        gpu.weights_cache_total_pages = weights_caches[i]->total_pages;
        gpu.io_pool_size = io_pool_size;
        gpu.workspace_pool_size = workspace_pool_size;
        // Add models later
        info.gpus.push_back(gpu);
    }

    // Store model info
    models->get_model_info(info);
}

void lookupValue_(libconfig::Config &config, std::string key, uint64_t &value) {
    unsigned long long v = 0;
    if (config.getRoot().lookupValue(key, v)) {
        value = v;
    }
}

std::vector<ModelDataDummy> loadModelDataDummy(std::string base_filename) {
    std::vector<ModelDataDummy> modeldata;

    for (unsigned batch_size = 1; ; batch_size *=2) {
        std::stringstream batch_filename_base;
        batch_filename_base << base_filename << "." << batch_size;

        std::string so_filename = batch_filename_base.str() + ".so";
        std::string clockwork_filename = batch_filename_base.str() + ".clockwork";

        if (!util::exists(so_filename) || !util::exists(clockwork_filename)) {
            break;
        }

        std::string serialized_spec;
        util::readFileAsString(clockwork_filename, serialized_spec);

        modeldata.push_back(ModelDataDummy{
            batch_size,
            serialized_spec,
            0,
            0
        });
    }
    
    CHECK(modeldata.size() != 0) << "No valid batch sizes found for " << base_filename;
    
    // Load measurements if they exist
    try {
        std::string measurements_file = base_filename + ".measurements";
        libconfig::Config measurements;
        measurements.readFile(measurements_file.c_str());

        uint64_t weights_measurement;
        lookupValue_(measurements, "weights", weights_measurement);
        for (auto &model : modeldata) {
            std::stringstream key;
            key << "b" << model.batch_size;
            lookupValue_(measurements, key.str(), model.exec_measurement);
            model.weights_measurement = weights_measurement;
        }
    } catch (const libconfig::FileIOException& e) {
        std::cerr<< "here2";
        throw NoMeasureFile(actionErrorUnknownModel,"No measurements file for " + base_filename);
    }

    return modeldata;
}



}
