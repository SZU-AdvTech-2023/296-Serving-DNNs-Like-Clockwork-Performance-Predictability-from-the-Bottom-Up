#include "clockwork/alternatives/worker.h"
#include "clockwork/alternatives/model_manager.h"
#include <iostream>
#include <atomic>
#include "clockwork/util.h"
#include <cuda_runtime.h>
#include "tvm/runtime/cuda_common.h"
#include "clockwork/util.h"
#include "clockwork/telemetry.h"

using namespace clockwork::alternatives;

ModelManager::ModelManager(const int id, Runtime* runtime, PageCache* cache, model::Model* model, TelemetryLogger* logger) : id(id), runtime(runtime), model(cache, model), logger(logger), request_id_seed(0) {
	CUDA_CALL(cudaMallocHost(&output, model->output_size()));
}

ModelManager::~ModelManager() {
	while (pending_requests.size() > 0) {
		Request* r = pending_requests.front();
		pending_requests.pop_front();
		free(r->output);
		delete r;
	}
}

bool ModelManager::evict() {
	std::lock_guard<std::mutex> lock(queue_mutex);

	return model.evict_weights();
}


void ModelManager::submit(Request* r) {
	r->id = request_id_seed++;
	r->output_size = model.output_size();
	r->output = output; // TODO: should be allocated per request from a host-pinned buffer pool
	r->telemetry = new RequestTelemetry();
	r->telemetry->model_id = id;
	r->telemetry->arrived = clockwork::util::hrt();

	Request* toSubmit = nullptr;
	{
		std::lock_guard<std::mutex> lock(queue_mutex);

		pending_requests.push_back(r);
		if (model.try_lock()) {
			toSubmit = pending_requests.front();
			pending_requests.pop_front();
		}
	}

	if (toSubmit != nullptr) {
		execute(toSubmit);
	}
}

void ModelManager::execute(Request* request) {
	RequestBuilder* builder = runtime->newRequest();

	builder->setTelemetry(request->telemetry);

	if (!model.has_weights()) {
		builder->addTask(TaskType::PCIe_H2D_Weights, [this] {
			this->model.transfer_weights(util::Stream()); // TODO: pass stream as argument to function
		});
	}

	builder->addTask(TaskType::PCIe_H2D_Inputs, [this, request] {
		this->model.set_input(request->input, util::Stream());
    });

	builder->addTask(TaskType::GPU, [this] {
		this->model.call(util::Stream());
	});

	builder->addTask(TaskType::PCIe_D2H_Output, [this, request] {
		this->model.get_output(request->output, util::Stream());
	});

	// Task is unnecessary since onComplete callback won't run until async part of previous task is completed
	// builder->addTask(TaskType::Sync, [this, request] {
	// 	// cudaStreamSynchronize might not be necessary -- it waits for the PCIe_D2H_Output to complete,
	// 	// but some executor types might already guarantee it's completed.  Some, however, will not
	// 	// provide this guarantee, and only do a cudaStreamWaitEvent on the current stream.
	// 	CUDA_CALL(cudaStreamSynchronize(util::Stream()));
	// });

	builder->setCompletionCallback([this, request] {
		this->handle_response(request);
	});

	request->telemetry->submitted = clockwork::util::hrt();

	builder->submit();
}


void ModelManager::handle_response(Request* r) {
	r->telemetry->complete = clockwork::util::hrt();

	Request* toSubmit = nullptr;
	{
		std::lock_guard<std::mutex> lock(queue_mutex);

		if (pending_requests.size() > 0) {
			toSubmit = pending_requests.front();
			pending_requests.pop_front();
		}
	}

	if (toSubmit != nullptr) {
		execute(toSubmit);
	} else {
		model.unlock();
	}

	r->callback();
	this->logger->log(r->telemetry);
	delete r;
}