#include <unistd.h>
#include <shared_mutex>
#include <algorithm>
#include <chrono>
#include <thread>
#include <catch2/catch.hpp>
#include <cuda_runtime.h>
#include "clockwork/cuda_common.h"
#include "clockwork/util.h"
#include "clockwork/test/util.h"
#include "clockwork/model/so.h"
#include "clockwork/model/cuda.h"
#include "clockwork/memory.h"
#include "clockwork/model/model.h"
#include <sys/mman.h>
#include "clockwork/thread.h"

using namespace clockwork;

model::Model* duplicate(model::Model* model, bool duplicate_weights) {
    Memfile so_memfile = Memfile::readFrom(model->so_memfile.filename);

    std::string serialized_spec = model->serialized_spec;

    void* weights_pinned_host_memory;
    if (duplicate_weights) {
        // weights_pinned_host_memory = malloc(model->weights_size);
        // REQUIRE(mlock(weights_pinned_host_memory, model->weights_size) == 0);
        // cudaError_t status = cudaHostRegister(weights_pinned_host_memory, model->weights_size, cudaHostRegisterDefault);
        // REQUIRE(status == cudaSuccess);
        CUDA_CALL(cudaMallocHost(&weights_pinned_host_memory, model->weights_size));
        std::memcpy(weights_pinned_host_memory, model->weights_pinned_host_memory, model->weights_size);
    } else {
        weights_pinned_host_memory = model->weights_pinned_host_memory;
    }

	// TODO Does it suffice if profiling is on GPU 0?
    return new model::Model(so_memfile, serialized_spec, model->weights_size, static_cast<char*>(weights_pinned_host_memory), GPU_ID_0);
}

class Experiment {
public:
    std::atomic_int progress;
    std::shared_mutex shared;
    std::mutex copy_weights_mutex, copy_inputs_mutex, exec_mutex, copy_output_mutex;
    cudaStream_t copy_weights_stream, copy_inputs_stream, exec_stream, copy_output_stream;

    Experiment() : progress(0) {
        cudaError_t status;
        status = cudaStreamCreateWithFlags(&copy_inputs_stream, cudaStreamNonBlocking);
        REQUIRE(status == cudaSuccess);
        status = cudaStreamCreateWithFlags(&copy_weights_stream, cudaStreamNonBlocking);
        REQUIRE(status == cudaSuccess);
        status = cudaStreamCreateWithFlags(&exec_stream, cudaStreamNonBlocking);
        REQUIRE(status == cudaSuccess);
        status = cudaStreamCreateWithFlags(&copy_output_stream, cudaStreamNonBlocking);
        REQUIRE(status == cudaSuccess);
    }
};

struct Measurement {
    std::vector<float> cuda;
    std::vector<uint64_t> host;
    Measurement(std::vector<cudaEvent_t> &events, std::vector<clockwork::time_point> &timestamps) : cuda(events.size()/2), host(timestamps.size()/2) {
        cudaError_t status;
        for (int i = 1; i < events.size(); i += 1) {
            float duration;
            status = cudaEventElapsedTime(&duration, events[i-1], events[i]);
            REQUIRE(status == cudaSuccess);
            cuda[i/2] = duration;
        }
        for (int i = 1; i < timestamps.size(); i += 2) {
            host[i/2] = util::nanos(timestamps[i]) - util::nanos(timestamps[i-1]);
        } 
    }
};

struct Series {
    std::vector<std::vector<uint64_t>> data;

    Series(std::vector<Measurement> &measurements) {
        data.resize(measurements[0].cuda.size() + measurements[0].host.size());

        for (unsigned i = 0; i < data.size(); i++) {
            data[i].resize(measurements.size());
        }

        for (unsigned i = 0; i < measurements.size(); i++) {
            unsigned next = 0;
            for (float &f : measurements[i].cuda) {
                data[next++][i] = 1000000 * f;
            }
            for (uint64_t &x : measurements[i].host) {
                data[next++][i] = x;
            }
        }
    }

    std::vector<uint64_t> medians() {
        std::vector<uint64_t> medians(data.size());
        for (unsigned i = 0; i < data.size(); i++) {
            std::vector<uint64_t> series(data[i]);
            std::sort(series.begin(), series.end());
            medians[i] = series[series.size()/2]; 
        }
        return medians;
    }

    std::vector<uint64_t> percentiles(float p) {
        std::vector<uint64_t> ps(data.size());
        for (unsigned i = 0; i < data.size(); i++) {
            std::vector<uint64_t> series(data[i]);
            std::sort(series.begin(), series.end());
            ps[i] = series[(int) ((series.size()-1)*p)]; 
        }
        return ps;
    }

};

class ModelExecWithModuleLoad {
public:
    std::atomic_int iterations;
    std::thread thread;
    std::atomic_bool started, ready, alive;
    PageCache* weights_cache;
    MemoryPool* workspace_pool;
    MemoryPool* io_pool;
    Experiment* experiment;
    std::vector<model::Model*> models;
    std::string input;


    std::vector<Measurement> measurements;

    void run_with_module_load() {
        util::initializeCudaStream();

        for (model::Model* model : models) {
            model->instantiate_model_on_host();
        }


        cudaError_t status;

        std::vector<cudaEvent_t> events(8);
        for (unsigned i = 0; i < events.size(); i++) {
            status = cudaEventCreate(&events[i]);
            REQUIRE(status == cudaSuccess);            
        }

        ready = true;
        while (!started) {}

        while (alive) {
            model::Model* model = models[rand() % models.size()];

            std::vector<clockwork::time_point> timestamps;
            timestamps.reserve(12);

            experiment->shared.lock();
            timestamps.push_back(util::hrt());
            model->instantiate_model_on_device();
            experiment->shared.unlock();
            timestamps.push_back(util::hrt());

            std::shared_ptr<Allocation> weights = weights_cache->alloc(model->num_weights_pages(weights_cache->page_size), []{});


            experiment->copy_weights_mutex.lock();
            experiment->shared.lock_shared();
            timestamps.push_back(util::hrt());
            status = cudaEventRecord(events[0], experiment->copy_weights_stream);
            REQUIRE(status == cudaSuccess);
            model->transfer_weights_to_device(weights->page_pointers, experiment->copy_weights_stream);
            status = cudaEventRecord(events[1], experiment->copy_weights_stream);
            REQUIRE(status == cudaSuccess);
            experiment->copy_weights_mutex.unlock();

            status = cudaEventSynchronize(events[1]);
            experiment->shared.unlock_shared();
            REQUIRE(status == cudaSuccess);
            timestamps.push_back(util::hrt());

            char* io_memory = io_pool->alloc(model->io_memory_size());

            experiment->copy_inputs_mutex.lock();
            experiment->shared.lock_shared();
            timestamps.push_back(util::hrt());
            status = cudaEventRecord(events[2], experiment->copy_inputs_stream);
            REQUIRE(status == cudaSuccess);
            model->transfer_input_to_device(input.data(), io_memory, experiment->copy_inputs_stream);
            status = cudaEventRecord(events[3], experiment->copy_inputs_stream);
            REQUIRE(status == cudaSuccess);
            experiment->copy_inputs_mutex.unlock();

            status = cudaEventSynchronize(events[3]);
            experiment->shared.unlock_shared();
            REQUIRE(status == cudaSuccess);
            timestamps.push_back(util::hrt());

            char* workspace_memory = workspace_pool->alloc(model->workspace_memory_size());

            experiment->exec_mutex.lock();
            experiment->shared.lock_shared();
            timestamps.push_back(util::hrt());
            status = cudaEventRecord(events[4], experiment->exec_stream);
            REQUIRE(status == cudaSuccess);
            model->call(weights->page_pointers, io_memory, workspace_memory, experiment->exec_stream);
            status = cudaEventRecord(events[5], experiment->exec_stream);
            REQUIRE(status == cudaSuccess);
            experiment->exec_mutex.unlock();

            REQUIRE(status == cudaSuccess);

            status = cudaEventSynchronize(events[5]);
            experiment->shared.unlock_shared();
            REQUIRE(status == cudaSuccess);
            timestamps.push_back(util::hrt());

            char output[model->output_size()];

            experiment->copy_output_mutex.lock();
            experiment->shared.lock_shared();
            timestamps.push_back(util::hrt());
            status = cudaEventRecord(events[6], experiment->copy_output_stream);
            REQUIRE(status == cudaSuccess);
            model->transfer_output_from_device(output, io_memory, experiment->copy_output_stream);
            status = cudaEventRecord(events[7], experiment->copy_output_stream);
            REQUIRE(status == cudaSuccess);
            experiment->copy_output_mutex.unlock();

            REQUIRE(status == cudaSuccess);

            status = cudaEventSynchronize(events[7]);
            experiment->shared.unlock_shared();
            REQUIRE(status == cudaSuccess);
            timestamps.push_back(util::hrt());

            workspace_pool->free(workspace_memory);
            io_pool->free(io_memory);
            weights_cache->unlock(weights);
            weights_cache->free(weights);


            experiment->shared.lock();
            timestamps.push_back(util::hrt());
            model->uninstantiate_model_on_device(); 
            experiment->shared.unlock();
            timestamps.push_back(util::hrt());

            iterations++;
            experiment->progress++;

            measurements.push_back(Measurement(events, timestamps));
        }

        for (model::Model* model : models) {
            model->uninstantiate_model_on_host();
            delete model;
        }

    }


    void run_without_module_load() {
        util::initializeCudaStream();

        for (model::Model* model : models) {
            model->instantiate_model_on_host();
            model->instantiate_model_on_device();
        }


        cudaError_t status;

        std::vector<cudaEvent_t> events(8);
        for (unsigned i = 0; i < events.size(); i++) {
            status = cudaEventCreate(&events[i]);
            REQUIRE(status == cudaSuccess);            
        }

        ready = true;
        while (!started) {}

        while (alive) {
            model::Model* model = models[rand() % models.size()];

            std::vector<clockwork::time_point> timestamps;
            timestamps.reserve(8);

            std::shared_ptr<Allocation> weights = weights_cache->alloc(model->num_weights_pages(weights_cache->page_size), []{});


            experiment->copy_weights_mutex.lock();
            timestamps.push_back(util::hrt());
            status = cudaEventRecord(events[0], experiment->copy_weights_stream);
            REQUIRE(status == cudaSuccess);
            model->transfer_weights_to_device(weights->page_pointers, experiment->copy_weights_stream);
            status = cudaEventRecord(events[1], experiment->copy_weights_stream);
            REQUIRE(status == cudaSuccess);
            experiment->copy_weights_mutex.unlock();

            status = cudaEventSynchronize(events[1]);
            REQUIRE(status == cudaSuccess);
            timestamps.push_back(util::hrt());

            char* io_memory = io_pool->alloc(model->io_memory_size());

            experiment->copy_inputs_mutex.lock();
            timestamps.push_back(util::hrt());
            status = cudaEventRecord(events[2], experiment->copy_inputs_stream);
            REQUIRE(status == cudaSuccess);
            model->transfer_input_to_device(input.data(), io_memory, experiment->copy_inputs_stream);
            status = cudaEventRecord(events[3], experiment->copy_inputs_stream);
            REQUIRE(status == cudaSuccess);
            experiment->copy_inputs_mutex.unlock();

            status = cudaEventSynchronize(events[3]);
            REQUIRE(status == cudaSuccess);
            timestamps.push_back(util::hrt());

            char* workspace_memory = workspace_pool->alloc(model->workspace_memory_size());

            experiment->exec_mutex.lock();
            timestamps.push_back(util::hrt());
            status = cudaEventRecord(events[4], experiment->exec_stream);
            REQUIRE(status == cudaSuccess);
            model->call(weights->page_pointers, io_memory, workspace_memory, experiment->exec_stream);
            status = cudaEventRecord(events[5], experiment->exec_stream);
            REQUIRE(status == cudaSuccess);
            experiment->exec_mutex.unlock();

            REQUIRE(status == cudaSuccess);

            status = cudaEventSynchronize(events[5]);
            REQUIRE(status == cudaSuccess);
            timestamps.push_back(util::hrt());

            char output[model->output_size()];

            experiment->copy_output_mutex.lock();
            timestamps.push_back(util::hrt());
            status = cudaEventRecord(events[6], experiment->copy_output_stream);
            REQUIRE(status == cudaSuccess);
            model->transfer_output_from_device(output, io_memory, experiment->copy_output_stream);
            status = cudaEventRecord(events[7], experiment->copy_output_stream);
            REQUIRE(status == cudaSuccess);
            experiment->copy_output_mutex.unlock();

            REQUIRE(status == cudaSuccess);

            status = cudaEventSynchronize(events[7]);
            REQUIRE(status == cudaSuccess);
            timestamps.push_back(util::hrt());

            workspace_pool->free(workspace_memory);
            io_pool->free(io_memory);
            weights_cache->unlock(weights);
            weights_cache->free(weights);

            iterations++;
            experiment->progress++;

            measurements.push_back(Measurement(events, timestamps));
        }

        for (model::Model* model : models) {
            model->uninstantiate_model_on_device();
            model->uninstantiate_model_on_host();
            delete model;
        }

    }

    ModelExecWithModuleLoad(int i, Experiment* experiment, PageCache* weights_cache, MemoryPool* io_pool, MemoryPool* workspace_pool, std::vector<model::Model*> models, std::string input) :
            experiment(experiment), weights_cache(weights_cache), io_pool(io_pool), workspace_pool(workspace_pool), models(models), alive(true), input(input), iterations(0),
            ready(false), started(false) {
        threading::setMaxPriority();
    }

    void start(bool with_module_load) {
        if (with_module_load) {
            thread = std::thread(&ModelExecWithModuleLoad::run_with_module_load, this);
        } else {
            thread = std::thread(&ModelExecWithModuleLoad::run_without_module_load, this);            
        }
    }

    void stop(bool awaitCompletion) {
        alive = false;
        if (awaitCompletion) {
            join();
        }
    }

    void join() {
        thread.join();
    }

    void awaitIterations(int iterations) {
        while (this->iterations.load() < iterations) {}
    }

};

model::Model* load_model_from_disk(std::string model_path) {
    std::string so_filename = model_path + ".1.so";
    std::string clockwork_filename = model_path + ".1.clockwork";
    std::string params_filename = model_path + ".clockwork_params";
	// TODO Does it suffice if profiling is from GPU 0?
    return model::Model::loadFromDisk(so_filename, clockwork_filename, params_filename, GPU_ID_0);
}

void get_model_inputs_and_outputs(std::string model_path, std::string &input, std::string &output) {
    std::string input_filename = model_path + ".input";
    std::string output_filename = model_path + ".output";
    clockwork::util::readFileAsString(input_filename, input);
    clockwork::util::readFileAsString(output_filename, output);
}

void warmup() {
    util::initializeCudaStream();

    std::string model_name = "resnet50";
    std::string model_path = clockwork::util::get_example_model("resnet50_tesla-m40_batchsize1");

    size_t weights_page_size = 16 * 1024L * 1024L;
    size_t weights_cache_size = 512L * weights_page_size;
    PageCache* weights_cache = make_GPU_cache(weights_cache_size, weights_page_size, GPU_ID_0);

    size_t io_pool_size = 128 * 1024L * 1024L;
    size_t workspace_pool_size = 512 * 1024L * 1024L;

    MemoryPool* io_pool = CUDAMemoryPool::create(io_pool_size, GPU_ID_0);
    MemoryPool* workspace_pool = CUDAMemoryPool::create(workspace_pool_size, GPU_ID_0);

    model::Model* model = load_model_from_disk(model_path);

    std::string input, expected_output;
    get_model_inputs_and_outputs(model_path, input, expected_output);

    cudaError_t status;

    model->instantiate_model_on_host();
    model->instantiate_model_on_device();
    
    std::shared_ptr<Allocation> weights = weights_cache->alloc(model->num_weights_pages(weights_page_size), []{});
    model->transfer_weights_to_device(weights->page_pointers, util::Stream());

    char* io_memory = io_pool->alloc(model->io_memory_size());
    model->transfer_input_to_device(input.data(), io_memory, util::Stream());
    
    char* workspace_memory = workspace_pool->alloc(model->workspace_memory_size());
    model->call(weights->page_pointers, io_memory, workspace_memory, util::Stream());

    char output[model->output_size()];
    model->transfer_output_from_device(output, io_memory, util::Stream());

    status = cudaStreamSynchronize(util::Stream());
    REQUIRE(status == cudaSuccess);


    io_pool->free(io_memory);
    workspace_pool->free(workspace_memory);
    weights_cache->unlock(weights);
    weights_cache->free(weights);

    model->uninstantiate_model_on_device();
    model->uninstantiate_model_on_host();

    delete model;
    delete io_pool;
    delete workspace_pool;
}

TEST_CASE("Warmup works", "[profile] [warmup]") {
    for (unsigned i = 0; i < 3; i++) {
        warmup();
    }
}

TEST_CASE("Test max concurrent models", "[so_limits]") {
    util::setCudaFlags();
    util::initializeCudaStream();

    std::string model_path = "/home/jcmace/clockwork-modelzoo-volta/others/cifar_resnet20_v1/model";

    model::Model* model = load_model_from_disk(model_path);
    std::vector<model::Model*> copies;
    for (unsigned i = 0; i < 100000; i++) {
        model::Model* copy = duplicate(model, false);
        // model::Model* copy = load_model_from_disk(model_path);
        copy->instantiate_model_on_host();
        //copy->instantiate_model_on_device();
        copies.push_back(copy);
        std::cout << i << std::endl;
    }

}

void runMultiClientExperiment(int num_execs, int models_per_exec, bool duplicate_weights, int iterations, bool with_module_load) {
    util::setCudaFlags();
    for (unsigned i = 0; i < 3; i++) {
        warmup();
    }
    
    util::initializeCudaStream();

    std::string model_name = "resnet50";
    std::string model_path = clockwork::util::get_example_model("resnet50_tesla-m40_batchsize1");

    size_t weights_page_size = 16 * 1024L * 1024L;
    size_t weights_cache_size = 512L * weights_page_size;
    PageCache* weights_cache = make_GPU_cache(weights_cache_size, weights_page_size, GPU_ID_0);

    size_t io_pool_size = 128 * 1024L * 1024L;
    size_t workspace_pool_size = 512 * 1024L * 1024L;

    MemoryPool* io_pool = CUDAMemoryPool::create(io_pool_size, GPU_ID_0);
    MemoryPool* workspace_pool = CUDAMemoryPool::create(workspace_pool_size, GPU_ID_0);

    Experiment* experiment = new Experiment();

    std::vector<ModelExecWithModuleLoad*> execs;

    model::Model* model;
    std::string input, output;
    get_model_inputs_and_outputs(model_path, input, output);

    uint64_t started = util::now();

    for (unsigned i = 0; i < num_execs; i++) {
        std::vector<model::Model*> models;
        for (unsigned j = 0; j < models_per_exec; j++) {
            if (i == 0 && j == 0) {
                model = load_model_from_disk(model_path);
            } else {
                model = duplicate(model, duplicate_weights);
            }
            models.push_back(model);

            float progress = ((float) ((i * models_per_exec) + j)) / ((float) (models_per_exec * num_execs));
            uint64_t seconds_left = 1000;
            if (progress != 0) {
                seconds_left = (1 - progress) * ((util::now() - started) / progress) / 1000000000;
            }

            std::cout << "Creating model: " << ((i * models_per_exec) + j) << " (" << ((int) (100*progress)) << "%) (" << seconds_left << ") \r";
            std::cout.flush();
        }

        ModelExecWithModuleLoad* exec = new ModelExecWithModuleLoad(i,
            experiment, weights_cache, io_pool, workspace_pool, models, input);

        execs.push_back(exec);
    }

    for (ModelExecWithModuleLoad* exec : execs) {
        exec->start(with_module_load);
    }
    for (ModelExecWithModuleLoad* exec : execs) {
        while (!exec->ready) {}
    }
    for (ModelExecWithModuleLoad* exec : execs) {
        exec->started = true;
    }


    std::cout << "Exec creation completed, awaiting termination" << std::endl;

    started = util::now();
    int progress;
    while ((progress = experiment->progress.load()) < iterations) {
        float fprogress = progress / ((float) iterations);
        uint64_t seconds_left = 1000;
        if (progress != 0) {
            seconds_left = (1 - fprogress) * ((util::now() - started) / fprogress) / 1000000000;
        }
        std::cout << progress << " (" << ((progress * 100) / iterations) << "%) (" << seconds_left << "s) \r";
        std::cout.flush();
        usleep(250000);
    }
    for (int i = 0; i < execs.size(); i++) {
        execs[i]->stop(false);
    }
    for (int i = 0; i < execs.size(); i++) {
        execs[i]->join();
    }

    std::vector<Measurement> measurements;
    for (int i = 0; i < execs.size(); i++) {
        measurements.insert(measurements.end(), execs[i]->measurements.begin(), execs[i]->measurements.end());
    }


    std::vector<std::string> series_names;
    if (with_module_load) {
        series_names = {
            "cWeights", "cInputs", "cKernel", "cOutputs",
            "hModuleLoad", "hWeights", "hInputs", "hKernel", "hOutputs", "hModuleUnload"
        };
    } else {
        series_names = {
            "cWeights", "cInputs", "cKernel", "cOutputs",
            "hWeights", "hInputs", "hKernel", "hOutputs"
        };
    }


    Series series(measurements);

    std::vector<uint64_t> medians = series.percentiles(0.5);
    std::vector<uint64_t> p99 = series.percentiles(0.99);
    std::vector<uint64_t> p999 = series.percentiles(0.999);
    std::vector<uint64_t> p9999 = series.percentiles(0.9999);
    std::vector<uint64_t> p99999 = series.percentiles(0.99999);

    for (unsigned i = 0; i < series_names.size(); i++) {
        std::cout << series_names[i] << ":  median " << medians[i];
        std::cout << "   p99 +";
        printf("%.2f", 100 * (((float) p99[i]) / ((float) medians[i]) - 1));
        std::cout << "   p99.9 +";
        printf("%.2f", 100 * (((float) p999[i]) / ((float) medians[i]) - 1));
        std::cout << "%   p99.99 +";
        printf("%.2f", 100 * (((float) p9999[i]) / ((float) medians[i]) - 1));
        std::cout << "%   p99.999 +";
        printf("%.2f", 100 * (((float) p99999[i]) / ((float) medians[i]) - 1));
        std::cout << "%" << std::endl;
    }

    for (int i = 0; i < execs.size(); i++) {
        delete execs[i];
    }

    delete experiment;
    delete weights_cache;
    delete io_pool;
    delete workspace_pool;
}



TEST_CASE("Profile resnet50 1 thread with module load", "[profile] [resnet50] [e2e-moduleload]") {
    runMultiClientExperiment(6, 100, true, 20000, true);
}
TEST_CASE("Profile resnet50 1 thread without module load", "[profile] [resnet50] [e2e]") {
    runMultiClientExperiment(1, 500, true, 1000, false);
}