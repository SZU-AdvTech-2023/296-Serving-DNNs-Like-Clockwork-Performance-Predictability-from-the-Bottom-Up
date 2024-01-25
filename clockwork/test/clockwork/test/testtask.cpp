#include <unistd.h>
#include <libgen.h>
#include <fstream>
#include <algorithm>
#include <memory>

#include <cuda_runtime.h>
#include "clockwork/api/worker_api.h"
#include "clockwork/test/util.h"
#include "clockwork/model/model.h"
#include "clockwork/task.h"
#include "clockwork/memory.h"
#include "clockwork/config.h"
#include <catch2/catch.hpp>
#include <stdio.h>

using namespace clockwork;
using namespace clockwork::model;

class TestLoadModelFromDiskTask : public LoadModelFromDiskTask {
public:
    bool is_success = false;
    bool is_error = false;
    bool is_cancelled = false;
    RuntimeModel* rm;
    int status_code;
    std::string error_message;

    TestLoadModelFromDiskTask(MemoryManager* cache, int model_id, std::string model_path, uint64_t earliest, uint64_t latest) : 
            LoadModelFromDiskTask(cache, model_id, model_path, earliest, latest) {
    }

    void run() {
        try {
            LoadModelFromDiskTask::run();
        } catch (TaskError &error) {
            this->error(error.status_code, error.message);
        }
    }

    void cancel() {
        is_cancelled = true;
    }

    void success(RuntimeModel* rm) {
        is_success = true;
        this->rm = rm;
    }

    void error(int status_code, std::string message) {
        is_error = true;
        this->status_code = status_code;
        this->error_message = message;
    }

};

class TestLoadWeightsTask : public LoadWeightsTask {
public:
    bool is_success = false;
    bool is_error = false;
    bool is_cancelled = false;
    RuntimeModel* rm;
    int status_code;
    std::string error_message;

    TestLoadWeightsTask(MemoryManager* cache, int model_id, uint64_t earliest,
		uint64_t latest, unsigned gpu_id, CudaEventPool* event_pool):
			LoadWeightsTask(cache, model_id, earliest, latest, gpu_id, event_pool) {}

    void run(cudaStream_t stream) {
        try {
            LoadWeightsTask::run(stream);
        } catch (TaskError &error) {
            this->error(error.status_code, error.message);
        }
    }

    void cancel() {
        is_cancelled = true;
    }

    void process_completion() {
        try {
            LoadWeightsTask::process_completion();
        } catch (TaskError &error) {
            this->error(error.status_code, error.message);
        }    
    }

    void success(RuntimeModel* rm) {
        is_success = true;
        this->rm = rm;
    }

    void error(int status_code, std::string message) {
        is_error = true;
        this->status_code = status_code;
        this->error_message = message;
    }

};

class TestEvictWeightsTask : public EvictWeightsTask {
public:
    bool is_success = false;
    bool is_error = false;
    bool is_cancelled = false;
    RuntimeModel* rm;
    int status_code;
    std::string error_message;

    TestEvictWeightsTask(MemoryManager* cache, int model_id, uint64_t earliest,
		uint64_t latest, unsigned gpu_id):
			EvictWeightsTask(cache, model_id, earliest, latest, gpu_id) {}

    void run(cudaStream_t stream) {
        try {
            EvictWeightsTask::run(stream);
        } catch (TaskError &error) {
            this->error(error.status_code, error.message);
        }
    }

    void cancel() {
        is_cancelled = true;
    }

    void success(RuntimeModel* rm) {
        is_success = true;
        this->rm = rm;
    }

    void error(int status_code, std::string message) {
        is_error = true;
        this->status_code = status_code;
        this->error_message = message;
    }

};


class TestCopyInputTask : public CopyInputTask {
public:
    bool is_success = false;
    bool is_error = false;
    bool is_cancelled = false;
    int status_code;
    std::string error_message;
    RuntimeModel* rm;
    char* io_memory;

    TestCopyInputTask(MemoryManager* cache, int model_id, uint64_t earliest,
		uint64_t latest, unsigned batch_size, size_t input_size, char* input,
		unsigned gpu_id, CudaEventPool* event_pool):
			CopyInputTask(cache, model_id, earliest, latest, batch_size,
				input_size, input, gpu_id, event_pool),
			io_memory(nullptr) {}

    void run(cudaStream_t stream) {
        try {
            CopyInputTask::run(stream);
        } catch (TaskError &error) {
            this->error(error.status_code, error.message);
        }
    }

    void cancel() {
        is_cancelled = true;
    }

    void process_completion() {
        try {
            CopyInputTask::process_completion();
        } catch (TaskError &error) {
            this->error(error.status_code, error.message);
        }    
    }

    void success(RuntimeModel* rm, char* io_memory) {
        is_success = true;
        this->rm = rm;
        this->io_memory = io_memory;
    }

    void error(int status_code, std::string message) {
        is_error = true;
        this->status_code = status_code;
        this->error_message = message;
    }

};

class TestInferTask : public ExecTask {
public:
    bool is_success = false;
    bool is_error = false;
    bool is_cancelled = false;
    int status_code;
    std::string error_message;

    TestInferTask(RuntimeModel* rm, MemoryManager* cache, uint64_t earliest,
		uint64_t latest, unsigned batch_size, char* io_memory, unsigned gpu_id,
		CudaEventPool* event_pool):
			ExecTask(rm, cache, earliest, latest, batch_size, io_memory,
				gpu_id, event_pool) {}

    void run(cudaStream_t stream) {
        try {
            ExecTask::run(stream);
        } catch (TaskError &error) {
            this->error(error.status_code, error.message);
        }
    }

    void cancel() {
        is_cancelled = true;
    }

    void process_completion() {
        try {
            ExecTask::process_completion();
        } catch (TaskError &error) {
            this->error(error.status_code, error.message);
        }    
    }

    void success() {
        is_success = true;
    }

    void error(int status_code, std::string message) {
        is_error = true;
        this->status_code = status_code;
        this->error_message = message;
    }

};

class TestCopyOutputTask : public CopyOutputTask {
public:
    bool is_success = false;
    bool is_error = false;
    bool is_cancelled = false;
    int status_code;
    std::string error_message;
    char* output;

    TestCopyOutputTask(RuntimeModel* rm, MemoryManager* manager,
		uint64_t earliest, uint64_t latest, unsigned batch_size,
		char* io_memory, unsigned gpu_id, CudaEventPool *event_pool):
			CopyOutputTask(rm, manager, earliest, latest, batch_size, io_memory,
			gpu_id, event_pool) {}

    void run(cudaStream_t stream) {
        try {
            CopyOutputTask::run(stream);
        } catch (TaskError &error) {
            this->error(error.status_code, error.message);
        }
    }

    void cancel() {
        is_cancelled = true;
    }

    void process_completion() {
        try {
            CopyOutputTask::process_completion();
        } catch (TaskError &error) {
            this->error(error.status_code, error.message);
        }    
    }

    void success(char* output) {
        is_success = true;
        this->output = output;
    }

    void error(int status_code, std::string message) {
        is_error = true;
        this->status_code = status_code;
        this->error_message = message;
    }

};

// Models get deleted by the MemoryManager
Model* make_model() {
    std::string f = clockwork::util::get_example_model();
    Model* model = Model::loadFromDisk(f+".1.so", f+".1.clockwork", f+".clockwork_params", GPU_ID_0);
    return model;
}

// Models get deleted by the MemoryManager
BatchedModel* make_batched_model(int batch_size, Model* model) {
    std::vector<std::pair<unsigned, Model*>> models = {{batch_size, model}};
    BatchedModel* batched = new BatchedModel(model->weights_size, model->weights_pinned_host_memory, models, GPU_ID_0);
    batched->instantiate_models_on_host();
    batched->instantiate_models_on_device();
    return batched;    
}

class Autostream {
public:
    cudaStream_t stream;
	unsigned gpu_id = 0;
    Autostream(unsigned gpu_id = 0): gpu_id(gpu_id) {
		REQUIRE(cudaSetDevice(gpu_id) == cudaSuccess);
		REQUIRE(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, 0)
			== cudaSuccess);
        //REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
	}
    ~Autostream() {
		REQUIRE(cudaSetDevice(gpu_id) == cudaSuccess);
        REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
    }
};

std::shared_ptr<MemoryManager> make_manager(
        size_t weights_cache_size, size_t weights_page_size, 
        size_t io_pool_size,
        size_t workspace_pool_size,
        size_t host_io_pool_size,
		unsigned num_gpus) {

	ClockworkWorkerConfig config = ClockworkWorkerConfig();
	config.weights_cache_size = weights_cache_size;
	config.weights_cache_page_size = weights_page_size;
	config.io_pool_size = io_pool_size;
	config.workspace_pool_size = workspace_pool_size;
	config.host_io_pool_size = host_io_pool_size;
	config.num_gpus = num_gpus;

	return std::make_shared<MemoryManager>(config);
}

std::shared_ptr<MemoryManager> make_manager() {
    return make_manager(
        10 * 1024L * 1024L * 1024L,
		16L * 1024L * 1024L,
        128L * 1024L * 1024L,
        256L * 1024L * 1024L,
        128L * 1024L * 1024L,
		NUM_GPUS_2); // since Volta machines have 2 GPUs
}

TEST_CASE("Load Model From Disk", "[task] [loadmodel]") {
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager();

    int model_id = 0;
    std::string model_path = clockwork::util::get_example_model();

    TestLoadModelFromDiskTask task(manager.get(), model_id, model_path, util::now(), util::now()+1000000000);

    task.run();

    INFO(task.status_code << ": " << task.error_message);
    REQUIRE(!task.is_error);
    REQUIRE(task.is_success);
}

TEST_CASE("Load Non-Existent Model From Disk", "[task] [loadmodel]") {
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager();

    int model_id = 0;
    std::string model_path = clockwork::util::get_example_model() + "bad";

    TestLoadModelFromDiskTask task(manager.get(),model_id, model_path, util::now(), util::now()+1000000000);

    task.run();

    INFO(task.status_code << ": " << task.error_message);
    REQUIRE(task.is_error);
    REQUIRE(!task.is_success);
}

TEST_CASE("Load Weights", "[task]") {
	CudaEventPool* event_pool = new CudaEventPool(GPU_ID_0);
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(make_batched_model(1, model), GPU_ID_0);
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager();
    manager->models->put(0, GPU_ID_0, rm);

    uint64_t now = util::now();
    TestLoadWeightsTask task(manager.get(), 0, now, now+1000000000, GPU_ID_0, event_pool);

    REQUIRE(task.eligible() == now);


    task.run(stream->stream);

    REQUIRE(!task.is_complete());

    while (!task.is_complete());

    task.process_completion();

    REQUIRE(task.is_success);
    REQUIRE(!task.is_error);
}

TEST_CASE("Load Weights Nonexistent Model", "[task]") {
	CudaEventPool* event_pool = new CudaEventPool(GPU_ID_0);
    Model* model = make_model();
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager();

    uint64_t now = util::now();
    TestLoadWeightsTask task(manager.get(), 0, now, now+1000000000, GPU_ID_0, event_pool);

    REQUIRE(task.eligible() == now);


    task.run(stream->stream);
    REQUIRE(task.is_error);
    REQUIRE(!task.is_success);
    REQUIRE(task.status_code == loadWeightsUnknownModel);
}

TEST_CASE("Load Weights Earliest", "[task]") {
	CudaEventPool* event_pool = new CudaEventPool(GPU_ID_0);
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(make_batched_model(1, model), GPU_ID_0);
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager();
    manager->models->put(0, GPU_ID_0, rm);

    uint64_t now = util::now();

    TestLoadWeightsTask task(manager.get(), 0, now+1000000000, now+1000000000, GPU_ID_0, event_pool);

    task.run(stream->stream);

    REQUIRE(task.is_complete());
    REQUIRE(!task.is_success);
    REQUIRE(task.is_error);
    REQUIRE(task.status_code == loadWeightsTooEarly);
}

TEST_CASE("Load Weights Latest", "[task]") {
	CudaEventPool* event_pool = new CudaEventPool(GPU_ID_0);
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(make_batched_model(1, model), GPU_ID_0);
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager();
    manager->models->put(0, GPU_ID_0, rm);

    uint64_t now = util::now();

    TestLoadWeightsTask task(manager.get(), 0, 0, now - 1000000, GPU_ID_0, event_pool);

    task.run(stream->stream);

    REQUIRE(task.is_complete());
    REQUIRE(!task.is_success);
    REQUIRE(task.is_error);
    REQUIRE(task.status_code == loadWeightsTooLate);
}

TEST_CASE("Load Weights Insufficient Cache", "[task]") {
	CudaEventPool* event_pool = new CudaEventPool(GPU_ID_0);
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(make_batched_model(1, model), GPU_ID_0);
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager(16 * 1024 * 1024, 16 * 1024 * 1024, 64 * 1024 * 1024, 64 * 1024 * 1024,64 * 1024 * 1024, NUM_GPUS_2);
    manager->models->put(0, GPU_ID_0, rm);

    uint64_t now = util::now();

    TestLoadWeightsTask task(manager.get(), 0, 0, now + 1000000000L, GPU_ID_0, event_pool);

    task.run(stream->stream);

    REQUIRE(task.is_complete());
    REQUIRE(!task.is_success);
    REQUIRE(task.is_error);
    REQUIRE(task.status_code == loadWeightsInsufficientCache);
}

TEST_CASE("Load Weights Version Update", "[task]") {
	CudaEventPool* event_pool = new CudaEventPool(GPU_ID_0);
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(make_batched_model(1, model), GPU_ID_0);
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager();
    manager->models->put(0, GPU_ID_0, rm);

    uint64_t now = util::now();

    TestLoadWeightsTask task(manager.get(), 0, 0, now + 1000000000L, GPU_ID_0, event_pool);

    task.run(stream->stream);
    rm->lock();

    REQUIRE(!task.is_complete());

    rm->version++;
    rm->unlock();

    while (!task.is_complete());
    task.process_completion();

    REQUIRE(!task.is_success);
    REQUIRE(task.is_error);
    REQUIRE(task.status_code == loadWeightsConcurrentModification);
}

TEST_CASE("Double Load Weights", "[task]") {
	CudaEventPool* event_pool = new CudaEventPool(GPU_ID_0);
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(make_batched_model(1, model), GPU_ID_0);
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager();
    manager->models->put(0, GPU_ID_0, rm);

    // Load weights 1
    TestLoadWeightsTask load1(manager.get(), 0, 0, util::now() + 1000000000L, GPU_ID_0, event_pool);
    load1.run(stream->stream);

    rm->lock();

    int invalid_version = rm->version;
    std::shared_ptr<Allocation> invalid_weights = rm->weights;

    rm->unlock();

    // Load weights 2
    TestLoadWeightsTask load2(manager.get(), 0, 0, util::now() + 1000000000L, GPU_ID_0, event_pool);
    load2.run(stream->stream);

    while (!load1.is_complete());
    load1.process_completion();
    while (!load2.is_complete());
    load2.process_completion();
    
    REQUIRE(!load1.is_success);
    REQUIRE(load1.is_error);
    REQUIRE(load2.is_success);
    REQUIRE(!load2.is_error);
}



TEST_CASE("Evict Weights", "[task]") {
	CudaEventPool* event_pool = new CudaEventPool(GPU_ID_0);
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(make_batched_model(1, model), GPU_ID_0);
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager();
    manager->models->put(0, GPU_ID_0, rm);

    // Load weights
    uint64_t now = util::now();
    TestLoadWeightsTask load(manager.get(), 0, 0, util::now() + 1000000000L, GPU_ID_0, event_pool);
    load.run(stream->stream);
    while (!load.is_complete());
    load.process_completion();
    
    REQUIRE(load.is_success);
    REQUIRE(!load.is_error);

    // Now evict them
    TestEvictWeightsTask evict(manager.get(), 0, 0, util::now() + 1000000000, GPU_ID_0);
    evict.run(stream->stream);

    REQUIRE(evict.is_success);
    REQUIRE(!evict.is_error);
}

TEST_CASE("Evict Non-Existent Weights", "[task]") {
	CudaEventPool* event_pool = new CudaEventPool(GPU_ID_0);
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(make_batched_model(1, model), GPU_ID_0);
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager();
    manager->models->put(0, GPU_ID_0, rm);

    // Evict weights
    TestEvictWeightsTask evict(manager.get(), 0, 0, util::now() + 1000000000, GPU_ID_0);
    evict.run(stream->stream);

    REQUIRE(!evict.is_success);
    REQUIRE(evict.is_error);
    REQUIRE(evict.status_code == evictWeightsNotInCache);
}

TEST_CASE("Double Evict", "[task]") {
	CudaEventPool* event_pool = new CudaEventPool(GPU_ID_0);
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(make_batched_model(1, model), GPU_ID_0);
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager();
    manager->models->put(0, GPU_ID_0, rm);

    // Load weights
    uint64_t now = util::now();
    TestLoadWeightsTask load(manager.get(), 0, 0, util::now() + 1000000000L, GPU_ID_0, event_pool);
    load.run(stream->stream);
    while (!load.is_complete());
    load.process_completion();
    
    REQUIRE(load.is_success);
    REQUIRE(!load.is_error);

    // Now evict them
    TestEvictWeightsTask evict(manager.get(), 0, 0, util::now() + 1000000000, GPU_ID_0);
    evict.run(stream->stream);

    REQUIRE(evict.is_success);
    REQUIRE(!evict.is_error);

    // Now evict them
    TestEvictWeightsTask evict2(manager.get(), 0, 0, util::now() + 1000000000, GPU_ID_0);
    evict2.run(stream->stream);

    REQUIRE(!evict2.is_success);
    REQUIRE(evict2.is_error);
    REQUIRE(evict2.status_code == evictWeightsNotInCache);
}

TEST_CASE("Evict Weights Nonexistent Model", "[task]") {
	CudaEventPool* event_pool = new CudaEventPool(GPU_ID_0);
    Model* model = make_model();
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager();

    uint64_t now = util::now();
    TestEvictWeightsTask task(manager.get(), 0, now, now+1000000000, GPU_ID_0);

    REQUIRE(task.eligible() == now);

    task.run(stream->stream);
    REQUIRE(task.is_error);
    REQUIRE(!task.is_success);
    REQUIRE(task.status_code == evictWeightsUnknownModel);
}

TEST_CASE("Copy Input", "[task]") {
	CudaEventPool* event_pool = new CudaEventPool(GPU_ID_0);
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(make_batched_model(1, model), GPU_ID_0);
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager();
    manager->models->put(0, GPU_ID_0, rm);

    char* input = static_cast<char*>(malloc(model->input_size()));

    TestCopyInputTask copyinput(manager.get(), 0, 0, util::now() + 1000000000, 1, model->input_size(), input, GPU_ID_0, event_pool);
    copyinput.run(stream->stream);
    while (!copyinput.is_complete());
    copyinput.process_completion();

    INFO("Error " << copyinput.status_code << ": " << copyinput.error_message);
    REQUIRE(copyinput.is_success);
    REQUIRE(!copyinput.is_error);

    free(input);
}

TEST_CASE("Copy Input Wrong Size", "[task] [wrongsize]") {
	CudaEventPool* event_pool = new CudaEventPool(GPU_ID_0);
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(make_batched_model(1, model), GPU_ID_0);
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager();
    manager->models->put(0, GPU_ID_0, rm);

    char* input = static_cast<char*>(malloc(10));

    TestCopyInputTask copyinput(manager.get(), 0, 0, util::now() + 1000000000, 1, 10, input, GPU_ID_0, event_pool);
    copyinput.run(stream->stream);
    while (!copyinput.is_complete());

    REQUIRE(!copyinput.is_success);
    REQUIRE(copyinput.is_error);
    REQUIRE(copyinput.status_code == copyInputInvalidInput);

    free(input);
}

TEST_CASE("Copy Input Nonexistent Model", "[task]") {
	CudaEventPool* event_pool = new CudaEventPool(GPU_ID_0);
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(make_batched_model(1, model), GPU_ID_0);
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager();

    char* input = static_cast<char*>(malloc(model->input_size()));

    uint64_t now = util::now();
    TestCopyInputTask copyinput(manager.get(), 0, now, util::now() + 1000000000, 1, model->input_size(), input, GPU_ID_0, event_pool);

    REQUIRE(copyinput.eligible() == now);


    copyinput.run(stream->stream);
    REQUIRE(copyinput.is_error);
    REQUIRE(!copyinput.is_success);
    REQUIRE(copyinput.status_code == copyInputUnknownModel);
}


TEST_CASE("Input and Infer", "[task]") {
	CudaEventPool* event_pool = new CudaEventPool(GPU_ID_0);
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(make_batched_model(1, model), GPU_ID_0);
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager();
    manager->models->put(0, GPU_ID_0, rm);


    TestLoadWeightsTask loadweights(manager.get(), 0, 0, util::now()+1000000000, GPU_ID_0, event_pool);
    loadweights.run(stream->stream);
    REQUIRE(!loadweights.is_error);
    
    while (!loadweights.is_complete());
    loadweights.process_completion();

    REQUIRE(loadweights.is_success);
    REQUIRE(!loadweights.is_error);

    char* input = static_cast<char*>(malloc(model->input_size()));

    TestCopyInputTask copyinput(manager.get(), 0, 0, util::now() + 1000000000, 1, model->input_size(), input, GPU_ID_0, event_pool);
    copyinput.run(stream->stream);
    REQUIRE(!copyinput.is_error);
    
    while (!copyinput.is_complete());
    copyinput.process_completion();

    REQUIRE(copyinput.is_success);
    REQUIRE(!copyinput.is_error);

    TestInferTask infer(rm, manager.get(), 0, util::now() + 1000000000, 1, copyinput.io_memory, GPU_ID_0, event_pool);
    infer.run(stream->stream);
    REQUIRE(!infer.is_error);

    while (!infer.is_complete());
    infer.process_completion();

    REQUIRE(infer.is_success);
    REQUIRE(!infer.is_error);

    free(input);
}

TEST_CASE("Infer Without Weights", "[task]") {
	CudaEventPool* event_pool = new CudaEventPool(GPU_ID_0);
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(make_batched_model(1, model), GPU_ID_0);
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager();
    manager->models->put(0, GPU_ID_0, rm);

    char* input = static_cast<char*>(malloc(model->input_size()));

    TestCopyInputTask copyinput(manager.get(), 0, 0, util::now() + 1000000000, 1, model->input_size(), input, GPU_ID_0, event_pool);
    copyinput.run(stream->stream);
    REQUIRE(!copyinput.is_error);
    
    while (!copyinput.is_complete());
    copyinput.process_completion();

    REQUIRE(copyinput.is_success);
    REQUIRE(!copyinput.is_error);

    TestInferTask infer(rm, manager.get(), 0, util::now() + 1000000000, 1, copyinput.io_memory, GPU_ID_0, event_pool);
    infer.run(stream->stream);
    REQUIRE(!infer.is_success);
    REQUIRE(infer.is_error);
    REQUIRE(infer.status_code == execWeightsMissing);

    free(input);
}



TEST_CASE("Input Infer and Output", "[task]") {
	CudaEventPool* event_pool = new CudaEventPool(GPU_ID_0);
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager();

    std::string model_path = clockwork::util::get_example_model();

    TestLoadModelFromDiskTask loadmodel(manager.get(), 0, model_path, util::now(), util::now()+1000000000);

    loadmodel.run();
    REQUIRE(loadmodel.is_success);
    REQUIRE(!loadmodel.is_error);

    RuntimeModel* rm = manager->models->get(0, GPU_ID_0);
    REQUIRE(rm != nullptr);
    model::BatchedModel* model = rm->model;

    TestLoadWeightsTask loadweights(manager.get(), 0, 0, util::now()+1000000000, GPU_ID_0, event_pool);
    loadweights.run(stream->stream);
    REQUIRE(!loadweights.is_error);
    
    while (!loadweights.is_complete());
    loadweights.process_completion();

    REQUIRE(loadweights.is_success);
    REQUIRE(!loadweights.is_error);

    char* input = static_cast<char*>(malloc(model->input_size(1)));

    TestCopyInputTask copyinput(manager.get(), 0, 0, util::now() + 1000000000, 1, model->input_size(1), input, GPU_ID_0, event_pool);
    copyinput.run(stream->stream);
    REQUIRE(!copyinput.is_error);
    
    while (!copyinput.is_complete());
    copyinput.process_completion();

    REQUIRE(copyinput.is_success);
    REQUIRE(!copyinput.is_error);
    REQUIRE(copyinput.io_memory != nullptr);

    TestInferTask infer(rm, manager.get(), 0, util::now() + 1000000000, 1, copyinput.io_memory, GPU_ID_0, event_pool);
    infer.run(stream->stream);
    REQUIRE(!infer.is_error);

    while (!infer.is_complete());
    infer.process_completion();

    REQUIRE(infer.is_success);
    REQUIRE(!infer.is_error);

    TestCopyOutputTask copyoutput(rm, manager.get(), 0, util::now() + 1000000000, 1, copyinput.io_memory, GPU_ID_0, event_pool);
    copyoutput.run(stream->stream);
    REQUIRE(!copyoutput.is_error);

    while (!copyoutput.is_complete());
    copyoutput.process_completion();

    REQUIRE(copyoutput.is_success);
    REQUIRE(!copyoutput.is_error);

    free(input);
}

TEST_CASE("Input Infer and Output Batched", "[task] [taskiobatched]") {
	CudaEventPool* event_pool = new CudaEventPool(GPU_ID_0);
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager();

    std::string model_path = clockwork::util::get_example_batched_model();

    TestLoadModelFromDiskTask loadmodel(manager.get(), 0, model_path, util::now(), util::now()+1000000000);

    loadmodel.run();
    REQUIRE(loadmodel.is_success);
    REQUIRE(!loadmodel.is_error);

    RuntimeModel* rm = manager->models->get(0, GPU_ID_0);
    REQUIRE(rm != nullptr);
    model::BatchedModel* model = rm->model;

    TestLoadWeightsTask loadweights(manager.get(), 0, 0, util::now()+1000000000, GPU_ID_0, event_pool);
    loadweights.run(stream->stream);
    REQUIRE(!loadweights.is_error);
    
    while (!loadweights.is_complete());
    loadweights.process_completion();

    REQUIRE(loadweights.is_success);
    REQUIRE(!loadweights.is_error);

    for (unsigned batch_size = 1; batch_size <= 16; batch_size++) {

        char* input = manager->host_io_pool->alloc(model->input_size(batch_size));

        TestCopyInputTask copyinput(manager.get(), 0, 0, util::now() + 1000000000, batch_size, model->input_size(batch_size), input, GPU_ID_0, event_pool);
        copyinput.run(stream->stream);
        INFO("B-" << batch_size << " Error " << copyinput.status_code << ": " << copyinput.error_message);
        REQUIRE(!copyinput.is_error);
        
        while (!copyinput.is_complete());
        copyinput.process_completion();

        INFO("B-" << batch_size << " Error " << copyinput.status_code << ": " << copyinput.error_message);
        REQUIRE(!copyinput.is_error);
        REQUIRE(copyinput.is_success);
        REQUIRE(copyinput.io_memory != nullptr);

        TestInferTask infer(rm, manager.get(), 0, util::now() + 1000000000, batch_size, copyinput.io_memory, GPU_ID_0, event_pool);
        infer.run(stream->stream);
        INFO("B-" << batch_size << " Error " << infer.status_code << ": " << infer.error_message);
        REQUIRE(!infer.is_error);

        while (!infer.is_complete());
        infer.process_completion();

        INFO("B-" << batch_size << " Error " << infer.status_code << ": " << infer.error_message);
        REQUIRE(!infer.is_error);
        REQUIRE(infer.is_success);

        TestCopyOutputTask copyoutput(rm, manager.get(), 0, util::now() + 1000000000, batch_size, copyinput.io_memory, GPU_ID_0, event_pool);
        copyoutput.run(stream->stream);
        INFO("B-" << batch_size << " Error " << copyoutput.status_code << ": " << copyoutput.error_message);
        REQUIRE(!copyoutput.is_error);

        while (!copyoutput.is_complete());
        copyoutput.process_completion();

        INFO("B-" << batch_size << " Error " << copyoutput.status_code << ": " << copyoutput.error_message);
        REQUIRE(!copyoutput.is_error);
        REQUIRE(copyoutput.is_success);

        manager->host_io_pool->free(copyoutput.output);
		manager->io_pools[GPU_ID_0]->free(copyinput.io_memory);
    }
}

TEST_CASE("Input Infer and Output Multiple GPUs", "[task]") {
	unsigned num_gpus = clockwork::util::get_num_gpus();

	std::vector<CudaEventPool *> event_pools;
	for (unsigned gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
		event_pools.push_back(new CudaEventPool(gpu_id));
	}

	std::vector<std::shared_ptr<Autostream> > streams;
	for (unsigned i = 0; i < num_gpus; i++) {
		streams.push_back(std::make_shared<Autostream>(i));
	}

	auto manager = make_manager();
	std::string model_path = clockwork::util::get_example_model();

	// Test LoadModelFromDiskTask
	// The task creates necessary data structures RuntimeModel and BatchedModel
	// for each GPU separately, and therefore does not need to be invoked
	// separately for each GPU
	TestLoadModelFromDiskTask loadmodel(manager.get(), 0, model_path,
		util::now(), util::now()+1000000000);
	loadmodel.run();
	REQUIRE(loadmodel.is_success);
	REQUIRE(!loadmodel.is_error);

	// Obtain GPU-specific RuntimeModel and BatchedModel objects from the
	// ModelStore (which is obtained from the MemoryManager)
	std::vector<RuntimeModel*> rms;
	std::vector<BatchedModel*> models;
	for (unsigned i = 0; i < num_gpus; i++) {
		RuntimeModel* rm = manager->models->get(0, i);
		model::BatchedModel* model = rm->model;
		REQUIRE(rm != nullptr);
		REQUIRE(model != nullptr);
		rms.push_back(rm);
		models.push_back(model);
	}

	// Test LoadWeightTask
	std::vector<TestLoadWeightsTask *> load_weights_tasks;
	for (unsigned i = 0; i < num_gpus; i++) {
		TestLoadWeightsTask* load_weights_task = new TestLoadWeightsTask(
			manager.get(), 0, 0, util::now()+1000000000, i, event_pools[i]);
		load_weights_task->run(streams[i]->stream);
		REQUIRE(!load_weights_task->is_error);
		load_weights_tasks.push_back(load_weights_task);
	}

	// Test LoadWeightTask completion
	for (unsigned i = 0; i < num_gpus; i++) {
		while (!load_weights_tasks[i]->is_complete());
		load_weights_tasks[i]->process_completion();
		REQUIRE(load_weights_tasks[i]->is_success);
		REQUIRE(!load_weights_tasks[i]->is_error);
	}

	// Test CopyInputTask
	std::vector<TestCopyInputTask *> copy_input_tasks;
	std::vector<char*> inputs;
	for (unsigned i = 0; i < num_gpus; i++) {
		char* input = static_cast<char*>(malloc(models[i]->input_size(1)));
		TestCopyInputTask* copy_input_task = new TestCopyInputTask(
			manager.get(), 0, 0, util::now() + 1000000000, 1,
			models[i]->input_size(1), input, i, event_pools[i]);
		copy_input_task->run(streams[i]->stream);
		REQUIRE(!copy_input_task->is_error);
		copy_input_tasks.push_back(copy_input_task);
		inputs.push_back(input);
	}

	// Test CopyInputTask completion
	for (unsigned i = 0; i < num_gpus; i++) {
		while (!copy_input_tasks[i]->is_complete());
		copy_input_tasks[i]->process_completion();
	    REQUIRE(copy_input_tasks[i]->is_success);
	    REQUIRE(!copy_input_tasks[i]->is_error);
	    REQUIRE(copy_input_tasks[i]->io_memory != nullptr);
	}

	// Test InferTask
	std::vector<TestInferTask *> infer_tasks;
	for (unsigned i = 0; i < num_gpus; i++) {
		TestInferTask* infer_task = new TestInferTask(rms[i], manager.get(), 0,
			util::now() + 1000000000, 1, copy_input_tasks[i]->io_memory, i,
			event_pools[i]);
		infer_task->run(streams[i]->stream);
		REQUIRE(!infer_task->is_error);
		infer_tasks.push_back(infer_task);
	}

	// Test InferTask completion
	for (unsigned i = 0; i < num_gpus; i++) {
		while (!infer_tasks[i]->is_complete());
		infer_tasks[i]->process_completion();
		REQUIRE(infer_tasks[i]->is_success);
		REQUIRE(!infer_tasks[i]->is_error);
	}

	// Test CopyOutputTask
	std::vector<TestCopyOutputTask *> copy_output_tasks;
	for (unsigned i = 0; i < num_gpus; i++) {
		TestCopyOutputTask* copy_output_task = new TestCopyOutputTask(
			rms[i], manager.get(), 0, util::now() + 1000000000, 1,
			copy_input_tasks[i]->io_memory, i, event_pools[i]);
		copy_output_task->run(streams[i]->stream);
		REQUIRE(!copy_output_task->is_error);
		copy_output_tasks.push_back(copy_output_task);
	}

	// Test InferTask completion
	for (unsigned i = 0; i < num_gpus; i++) {
		while (!copy_output_tasks[i]->is_complete());
		copy_output_tasks[i]->process_completion();
		REQUIRE(copy_output_tasks[i]->is_success);
		REQUIRE(!copy_output_tasks[i]->is_error);
	}

	// Free up all task objects
	for (unsigned i = 0; i < num_gpus; i++) {
		free(inputs[i]);
		free(load_weights_tasks[i]);
		free(copy_input_tasks[i]);
		free(infer_tasks[i]);
		free(copy_output_tasks[i]);
	}
}
