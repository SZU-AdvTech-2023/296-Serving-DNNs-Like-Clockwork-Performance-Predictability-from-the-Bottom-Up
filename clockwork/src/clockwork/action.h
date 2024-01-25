#ifndef _CLOCKWORK_ACTION_H_
#define _CLOCKWORK_ACTION_H_

#include <thread>
#include <limits>
#include <algorithm>
#include <memory>
#include <atomic>
#include <cuda_runtime.h>
#include "clockwork/cuda_common.h"
#include "clockwork/telemetry.h"
#include "clockwork/cache.h"
#include "clockwork/model/model.h"
#include "clockwork/priority_queue.h"
#include "clockwork/common.h"
#include "tbb/concurrent_queue.h"
#include "clockwork/task.h"
#include "clockwork/runtime.h"
#include "clockwork/api/worker_api.h"

/*
This file defines how to execute tasks (defined in task.h) within the clockwork scheduling
and thread-pool framework (defined in runtime.h).
*/

namespace clockwork {

class LoadModelFromDiskAction {
protected:

	class LoadModelFromDiskTaskImpl : public LoadModelFromDiskTask {
	public:
		LoadModelFromDiskAction* load_model;

		LoadModelFromDiskTaskImpl(LoadModelFromDiskAction* load_model);

		void run(cudaStream_t stream);
		void success(RuntimeModel* rm);
		void cancel();
		int duplicate(RuntimeModel* rm, int model_id, int no_of_copies);
	};

	ClockworkRuntime* runtime;
	std::shared_ptr<workerapi::LoadModelFromDisk> action;
	LoadModelFromDiskTaskImpl* task;

public:
	LoadModelFromDiskAction(ClockworkRuntime* runtime, std::shared_ptr<workerapi::LoadModelFromDisk> action);
	~LoadModelFromDiskAction();

	void submit();
	void handle_error(TaskError &error);

	virtual void success(std::shared_ptr<workerapi::LoadModelFromDiskResult> result) = 0;
	virtual void error(std::shared_ptr<workerapi::ErrorResult> result) = 0;
};

class LoadWeightsAction {
protected:

	class LoadWeightsTaskImpl : public LoadWeightsTask {
	public:
		LoadWeightsAction* load_weights;

		LoadWeightsTaskImpl(LoadWeightsAction* load_weights);

		void run(cudaStream_t stream);
		void process_completion();
		void success(RuntimeModel* rm);
		void cancel();
	};

	ClockworkRuntime* runtime;
	std::shared_ptr<workerapi::LoadWeights> action;
	LoadWeightsTaskImpl* task;

public:
	LoadWeightsAction(ClockworkRuntime* runtime, std::shared_ptr<workerapi::LoadWeights> action);
	~LoadWeightsAction();

	void submit();
	void handle_error(TaskError &error);

	virtual void success(std::shared_ptr<workerapi::LoadWeightsResult> result) = 0;
	virtual void error(std::shared_ptr<workerapi::ErrorResult> result) = 0;
};

class EvictWeightsAction {
protected:

	class EvictWeightsTaskImpl : public EvictWeightsTask {
	public:
		EvictWeightsAction* evict_weights;

		EvictWeightsTaskImpl(EvictWeightsAction* evict_weights);

		void run(cudaStream_t stream);
		void success(RuntimeModel* rm);
		void cancel();
	};

	ClockworkRuntime* runtime;
	std::shared_ptr<workerapi::EvictWeights> action;
	EvictWeightsTaskImpl* task;

public:
	EvictWeightsAction(ClockworkRuntime* runtime, std::shared_ptr<workerapi::EvictWeights> action);
	~EvictWeightsAction();

	void submit();
	void handle_error(TaskError &error);

	virtual void success(std::shared_ptr<workerapi::EvictWeightsResult> result) = 0;
	virtual void error(std::shared_ptr<workerapi::ErrorResult> result) = 0;
};


class InferAction {
public:

	class CopyInputTaskImpl : public CopyInputTask {
	public:
		InferAction* infer;

		CopyInputTaskImpl(InferAction* infer);

		void run(cudaStream_t stream);
		void process_completion();
		void success(RuntimeModel* rm, char* io_memory);
		void cancel();
	};

	class ExecTaskImpl : public ExecTask {
	public:
		InferAction* infer;
		unsigned gpu_clock_before;

		ExecTaskImpl(InferAction* infer);

		void run(cudaStream_t stream);
		void process_completion();
		void success();
		void cancel();
	};

	class CopyOutputTaskImpl : public CopyOutputTask {
	public:
		InferAction* infer;

		CopyOutputTaskImpl(InferAction* infer);

		void run(cudaStream_t stream);
		void process_completion();
		void success(char* output);
		void cancel();
	};

	ClockworkRuntime* runtime;

	std::shared_ptr<workerapi::Infer> action;

	RuntimeModel* rm;
	char* io_memory;

	bool zero_size = false;

	CopyInputTaskImpl* copy_input = nullptr;
	ExecTaskImpl* exec = nullptr;
	CopyOutputTaskImpl* copy_output = nullptr;
	
	/* Task types */
	const int copyInputTask= 0;
	const int execTask = 1;
	const int copyOutputTask = 2;

	uint64_t copy_input_earliest();

public:
	InferAction(ClockworkRuntime* runtime, std::shared_ptr<workerapi::Infer> action);
	~InferAction();

	void submit();
	void handle_completion(char* output);
	void handle_error(TaskError &error);

	virtual void success(std::shared_ptr<workerapi::InferResult> result) = 0;
	virtual void error(std::shared_ptr<workerapi::ErrorResult> result) = 0;

};



}

#endif
