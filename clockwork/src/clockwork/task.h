#ifndef _CLOCKWORK_TASK_H_
#define _CLOCKWORK_TASK_H_

#include <memory>
#include <atomic>
#include <cuda_runtime.h>
#include "clockwork/telemetry.h"
#include "clockwork/cache.h"
#include "clockwork/model/model.h"
#include "clockwork/memory.h"
#include "clockwork/cuda_common.h"

/*
This file contains logic for executing models directly
*/

namespace clockwork {

class CudaEventPool {
public:
	unsigned gpu_id;
	tbb::concurrent_queue<cudaEvent_t> events;

	CudaEventPool(unsigned gpu_id) : gpu_id(gpu_id) {
	}

	cudaEvent_t get_or_create() {
		cudaEvent_t event;
		if (!events.try_pop(event)) {
			CUDA_CALL(cudaSetDevice(gpu_id));
			CUDA_CALL(cudaEventCreate(&event));
		}
		return event;
	}

	void release(cudaEvent_t event) {
		events.push(event);
	}

};

class Task {
public:
	std::shared_ptr<TaskTelemetry> telemetry;
	unsigned gpu_id = -1;

	Task() : telemetry(std::make_shared<TaskTelemetry>()){}

	Task(unsigned gpu_id): gpu_id(gpu_id), telemetry(std::make_shared<TaskTelemetry>()) {}

	virtual uint64_t eligible() = 0;
	virtual void run(cudaStream_t stream) = 0;
	virtual void cancel() = 0;
};

class AsyncTask : public Task {
public:
	AsyncTask(unsigned gpu_id) : Task(gpu_id) {}

	virtual bool is_complete() = 0;
	virtual void await_completion() = 0;
	virtual void process_completion() = 0;
};


class CudaAsyncTask : public AsyncTask {
private:
	std::atomic_bool async_begin_submitted, async_end_submitted;
	cudaEvent_t async_begin_event, async_end_event;
public:
	CudaEventPool* event_pool;

	CudaAsyncTask(unsigned gpu_id, CudaEventPool* event_pool);
	~CudaAsyncTask();

	void record_async_begin(cudaStream_t stream);
	void record_async_end(cudaStream_t stream);
	float async_duration();

	// AsyncTask
	bool is_complete();
	void await_completion();
	virtual void process_completion() = 0;
};

class TaskError {
public:
	int status_code;
	std::string message;
	TaskError(int status_code, std::string message) : status_code(status_code), message(message) {}
};

/* For now, load, deserialize, and instantiate on host and device all in one.  TODO: split up. */
class LoadModelFromDiskTask : public Task {
private:
	MemoryManager* manager;
	uint64_t earliest, latest;
	int no_of_copies;
	unsigned max_batch_size;
	uint64_t max_exec_duration;

public:
	int model_id;
	std::string model_path;

	LoadModelFromDiskTask(MemoryManager* manager, int model_id, std::string model_path, uint64_t earliest, uint64_t latest, int no_of_copies = 1, unsigned max_batch_size = 32, uint64_t max_exec_duration = 1000000000UL);
	~LoadModelFromDiskTask();

	// Task
	uint64_t eligible();
	void run(cudaStream_t stream = 0);
	virtual void cancel() = 0;

	// Callbacks
	virtual void success(RuntimeModel* rm) = 0;

};

class LoadWeightsTask : public CudaAsyncTask {
private:
	int model_id;
	RuntimeModel* rm;
	MemoryManager* manager;
	uint64_t earliest, latest;

	int new_version;
	std::shared_ptr<Allocation> new_weights;

public:

	LoadWeightsTask(MemoryManager* manager, int model_id, uint64_t earliest,
		uint64_t latest, unsigned gpu_id, CudaEventPool* event_pool);
	~LoadWeightsTask();

	// Task
	uint64_t eligible();
	void run(cudaStream_t stream);
	virtual void cancel() = 0;

	// CudaAsyncTask
	void process_completion();

	// Callbacks
	virtual void success(RuntimeModel* rm) = 0;

};

class EvictWeightsTask : public Task {
private:
	int model_id;
	RuntimeModel* rm;
	MemoryManager* manager;
	uint64_t earliest, latest;

public:

	EvictWeightsTask(MemoryManager* manager, int model_id, uint64_t earliest,
		uint64_t latest, unsigned gpu_id);

	// Task
	uint64_t eligible();
	void run(cudaStream_t stream);
	virtual void cancel() = 0;

	// Callbacks
	virtual void success(RuntimeModel* rm) = 0;

};

class CopyInputTask : public CudaAsyncTask {
private:
	MemoryManager* manager;

	int model_id;
	uint64_t earliest, latest;
	unsigned batch_size;
	size_t input_size;
	char* &input;
	std::vector<size_t> compressed_input_sizes;

	RuntimeModel* rm;
	char* io_memory;

public:

	CopyInputTask(MemoryManager* manager, int model_id, uint64_t earliest,
		uint64_t latest, unsigned batch_size, size_t input_size, char* &input,
		unsigned gpu_id, CudaEventPool* event_pool);
	CopyInputTask(MemoryManager* manager, int model_id, uint64_t earliest,
		uint64_t latest, unsigned batch_size, size_t input_size, char* &input,
		std::vector<size_t> &compressed_input_sizes, unsigned gpu_id, CudaEventPool* event_pool);
	~CopyInputTask();

	// Task
	uint64_t eligible();
	void run(cudaStream_t stream);
	virtual void cancel() = 0;

	// CudaAsyncTask
	void process_completion();

	// Callbacks
	virtual void success(RuntimeModel* rm, char* io_memory) = 0;
};

class ExecTask : public CudaAsyncTask {
private:
	RuntimeModel* rm;
	MemoryManager* manager;
	uint64_t earliest, latest;

	unsigned batch_size;
	int weights_version;
	std::shared_ptr<Allocation> weights;
	char* io_memory;
	char* workspace_memory;

public:

	ExecTask(RuntimeModel* rm, MemoryManager* manager, uint64_t earliest,
		uint64_t latest, unsigned batch_size, char* io_memory, unsigned gpu_id,
		CudaEventPool* event_pool);
	~ExecTask();

	// Task
	uint64_t eligible();
	void run(cudaStream_t stream);
	virtual void cancel() = 0;

	// CudaAsyncTask
	void process_completion();

	// Callbacks
	virtual void success() = 0;
};

class CopyOutputTask : public CudaAsyncTask {
private:
	RuntimeModel* rm;
	MemoryManager* manager;

	uint64_t earliest, latest;
	unsigned batch_size;
	char* output;

	char* io_memory;

public:
	CopyOutputTask(RuntimeModel* rm, MemoryManager* manager, uint64_t earliest,
		uint64_t latest, unsigned batch_size, char* io_memory, unsigned gpu_id,
		CudaEventPool* event_pool);
	~CopyOutputTask();

	// Task
	uint64_t eligible();
	void run(cudaStream_t stream);
	virtual void cancel() = 0;

	// CudaAsyncTask
	void process_completion();

	// Callbacks
	virtual void success(char* output) = 0;
};

}

#endif
