#include <unistd.h>
#include <shared_mutex>
#include <algorithm>
#include <chrono>
#include <thread>
#include <catch2/catch.hpp>
#include <cuda_runtime.h>
#include "tvm/runtime/cuda_common.h"
#include "clockwork/util.h"
#include "clockwork/test/util.h"
#include "clockwork/model/so.h"
#include "clockwork/model/cuda.h"
#include "clockwork/cache.h"
#include "clockwork/model/model.h"
#include "clockwork/memory.h"
#include <sys/mman.h>
#include "clockwork/alternatives/model_manager.h"

namespace clockwork {
namespace alternatives {

std::vector<model::Model*> duplicate(model::Model* model, bool duplicate_weights, int num_duplicates) {
    if (num_duplicates == 0) {
        return std::vector<model::Model*>();
    }
    std::vector<model::Model*> models;

    std::vector<char*> weights_memories;
    if (duplicate_weights) {
        void* mega_memory;
        CUDA_CALL(cudaMallocHost(&mega_memory, model->weights_size * num_duplicates));
        for (unsigned i = 0; i < num_duplicates; i++) {
            size_t offset = model->weights_size * i;
            char* ptr = static_cast<char*>(mega_memory) + offset;
            std::memcpy(ptr, model->weights_pinned_host_memory, model->weights_size);
            weights_memories.push_back(ptr);
        }
    } else {
        for (unsigned i = 0; i < num_duplicates; i++) {
            weights_memories.push_back(model->weights_pinned_host_memory);
        }
    }

    for (unsigned i = 0; i < num_duplicates; i++) {
        Memfile so_memfile = Memfile::readFrom(model->so_memfile.filename);

        std::string serialized_spec = model->serialized_spec;

        models.push_back(new model::Model(
            so_memfile,
            serialized_spec,
            model->weights_size,
            weights_memories[i]
        ));
    }

    return models;
}

class ManagerLock {
public:
    std::atomic_flag in_use;
    ManagerLock() : in_use(ATOMIC_FLAG_INIT) {}

    bool lock() {
        return !in_use.test_and_set();
    }

    void unlock() {
        in_use.clear();
    }

};

class Exec {
public:
    std::atomic_flag in_use;
    std::atomic_int remainingToSubmit;
    std::atomic_int remaining;
    std::atomic_int tokens;
    std::vector<ModelManager*> managers;
    std::vector<ManagerLock*> managers_in_use;
    char* input = nullptr;

    Exec(int max_outstanding, std::vector<ModelManager*> managers) 
        : tokens(max_outstanding), managers(managers), in_use(ATOMIC_FLAG_INIT) {
        for (unsigned i = 0; i < managers.size(); i++) {
            managers_in_use.push_back(new ManagerLock());
        }
        CUDA_CALL(cudaMallocHost(&input, managers[0]->model.input_size()));
    }

    void submitSome() {
        while (in_use.test_and_set()); // spin

        while (remainingToSubmit.load() > 0 && tokens.load() > 0) {
            int i;
            while (true) {
                i = rand() % managers.size();
                if (managers_in_use[i]->lock()) break;
            }

            Request* r = new Request();
            r->user_request_id = 0;
            r->model_id = 0;
            r->input_size = managers[i]->model.input_size();
            r->input = input;
            r->batch_size = 1;
            r->callback = [this, i] {
                managers[i]->evict();
                managers_in_use[i]->unlock();
                this->remaining--;
                this->tokens++;
                this->submitSome();
            };
            r->errback = [this, i] (std::string message) {
                std::cout << "ERROR: " << message << std::endl;
                managers[i]->evict();
                managers_in_use[i]->unlock();
                this->remaining--;
                this->tokens++;
                this->submitSome();
            };
            managers[i]->submit(r);

            remainingToSubmit--;
            tokens--;
        }

        in_use.clear();
    }

    void run(int iterations) {
        remaining = iterations;
        remainingToSubmit = iterations;
        submitSome();
    }

    void await() {
        while (remaining.load() > 0) {}
    }

};

model::Model* load_model_from_disk(std::string model_path) {
    std::string so_filename = model_path + ".1.so";
    std::string clockwork_filename = model_path + ".1.clockwork";
    std::string params_filename = model_path + ".clockwork_params";
    return model::Model::loadFromDisk(so_filename, clockwork_filename, params_filename);    
}


struct Series {
    std::vector<std::vector<float>> data;

    Series(std::vector<RequestTelemetry*> &measurements, int warmups) {
        data.resize(measurements[0]->tasks.size());

        for (unsigned i = 0; i < data.size(); i++) {
            data[i].resize(measurements.size() - warmups);
        }

        for (unsigned i = warmups; i < measurements.size(); i++) {
            for (unsigned j = 0; j < measurements[i]->tasks.size(); j++) {
                data[j][i-warmups] = measurements[i]->tasks[j]->async_duration;
            }
        }
    }

    std::vector<float> medians() {
        std::vector<float> medians(data.size());
        for (unsigned i = 0; i < data.size(); i++) {
            std::vector<float> series(data[i]);
            std::sort(series.begin(), series.end());
            medians[i] = series[series.size()/2]; 
        }
        return medians;
    }

    std::vector<float> percentiles(float p) {
        std::vector<float> ps(data.size());
        for (unsigned i = 0; i < data.size(); i++) {
            std::vector<float> series(data[i]);
            std::sort(series.begin(), series.end());
            ps[i] = series[(int) ((series.size()-1)*p)]; 
        }
        return ps;
    }

};

size_t get_cudaMallocHost_max() {
    int increment = 1024 * 1024 * 128;

    cudaError_t status;

    int size = increment * 16;
    while (true) {
        void* ptr;
        status = cudaMallocHost(&ptr, size);
        if (status != cudaSuccess) {
            std::cout << "cudaMallocHost limit at " << (size - increment) << std::endl;
            return (size - increment);
        }
        size += increment;
        CUDA_CALL(cudaFreeHost(ptr));
    }    
}

void runMultiClientExperiment(int num_execs, int models_per_exec, int requests_per_exec, bool duplicate_weights, int iterations) {
    util::setCudaFlags();
    util::initializeCudaStream();

    size_t cudaMallocHost_max = get_cudaMallocHost_max();

    std::string model_name = "resnet50";
    std::string model_path = clockwork::util::get_example_model("resnet50_tesla-m40_batchsize1");

    size_t page_size = 16 * 1024L * 1024L;
    size_t cache_size = 512L * page_size;
    PageCache* cache = make_GPU_cache(cache_size, page_size);

    Runtime* runtime = clockwork::alternatives::newGreedyRuntime(1, 3);

    InMemoryTelemetryBuffer* logger = new InMemoryTelemetryBuffer();


    model::Model* model = load_model_from_disk(model_path);

    std::vector<Exec*> execs;

    for (unsigned i = 0; i < num_execs; i++) {
        std::cout << "Creating exec " << i << " \r";
        std::cout.flush();

        std::vector<model::Model*> models;
        if (i == 0) {
            models.push_back(model);
        }
        while (models.size() < models_per_exec) {
            int num_duplicates = cudaMallocHost_max / model->weights_size;
            if ((models_per_exec - models.size()) < num_duplicates) {
                num_duplicates = models_per_exec - models.size();
            }
            std::vector<model::Model*> duplicates = duplicate(model, duplicate_weights, num_duplicates);
            models.insert(models.end(), duplicates.begin(), duplicates.end());
        }
        std::vector<ModelManager*> managers;

        for (unsigned j = 0; j < models.size(); j++) {
            ModelManager* manager = new ModelManager(
                i * models_per_exec + j,
                runtime,
                cache,
                models[j],
                logger
            );
            managers.push_back(manager);
        }
        
        Exec* exec = new Exec(requests_per_exec, managers);
        execs.push_back(exec);
    }

    std::cout << "Created " << execs.size() << " execs.  Doing warmups" << std::endl;

    // Do some warmups first but only one of the execs
    int warmups = 0;
    execs[0]->run(warmups);
    while (execs[0]->remaining > 0) {
        std::cout << execs[0]->remaining << "      \r";
        std::cout.flush();
    }


    uint64_t started = util::now();

    for (Exec* exec : execs) {
        exec->run(iterations / execs.size());
    }
    std::cout << "Warmups complete.  Running" << std::endl;


    started = util::now();
    int progress;
    while (true) {
        int remaining = 0;
        for (Exec* exec : execs) {
            remaining += exec->remaining.load();
        }
        if (remaining == 0) {
            break;
        }

        std::cout << remaining << "     \r";
        std::cout.flush();
        usleep(100000);
    }

    std::vector<RequestTelemetry*> telemetry = logger->take_request();


    std::vector<std::string> series_names = {
        "cWeights", "cInputs", "cKernel", "cOutputs"
    };

    Series series(telemetry, 0);

    std::vector<float> medians = series.percentiles(0.5);
    std::vector<float> p99 = series.percentiles(0.99);
    std::vector<float> p999 = series.percentiles(0.999);
    std::vector<float> p9999 = series.percentiles(0.9999);
    std::vector<float> p99999 = series.percentiles(0.99999);

    for (unsigned i = 0; i < series_names.size(); i++) {
        std::cout << series_names[i] << ":  median " << medians[i];
        std::cout << "   p99 +";
        printf("%.2f", 100 * (p99[i] / medians[i] - 1));
        std::cout << "   p99.9 +";
        printf("%.2f", 100 * (p999[i] / medians[i] - 1));
        std::cout << "%   p99.99 +";
        printf("%.2f", 100 * (p9999[i] / medians[i] - 1));
        std::cout << "%   p99.999 +";
        printf("%.2f", 100 * (p99999[i] / medians[i] - 1));
        std::cout << "%" << std::endl;
    }
}


TEST_CASE("Profile cudaMallocHost limits", "[cudaMallocHost]") {
    get_cudaMallocHost_max();
}

TEST_CASE("Profile resnet50 greedy runtime", "[greedy]") {
    // runMultiClientExperiment(40, 100, 1, true, 1000000);
    // runMultiClientExperiment(40, 100, 1, true, 10000);
    // runMultiClientExperiment(10, 100, 1, true, 1000);
    // runMultiClientExperiment(1000, 1, 1, true, 1000);
    runMultiClientExperiment(1, 1000, 4, true, 1000);
}

}
}