#include "clockwork/alternatives/worker.h"
#include <iostream>
#include <atomic>
#include "clockwork/util.h"
#include <cuda_runtime.h>
#include "tvm/runtime/cuda_common.h"
#include "clockwork/util.h"
#include "clockwork/telemetry.h"

using namespace clockwork::alternatives;

Worker::Worker(Runtime* runtime, PageCache* cache, TelemetryLogger *logger) : runtime(runtime), cache(cache), logger(logger) {}

Worker::~Worker() {
	delete runtime;
	delete cache;
	delete logger;
	for (ModelManager* manager : managers) {
		delete manager;
	}
}

void Worker::shutdown() {
	runtime->shutdown(true);
	logger->shutdown(true);
}

void Worker::uploadModel(clientapi::UploadModelRequest &request, std::function<void(clientapi::UploadModelResponse&)> callback) {
	throw "Not implemented yet";	
}

void Worker::infer(clientapi::InferenceRequest &request, std::function<void(clientapi::InferenceResponse&)> callback) {
	if (request.batch_size != 1) {
		std::stringstream errorMsg;
		errorMsg << "Batch size " << request.batch_size << " not yet supported";
		clientapi::InferenceResponse response{
			ResponseHeader{request.header.user_request_id, clockworkError, errorMsg.str()}
		};
		callback(response);
		return;
	}

	ModelManager* manager = nullptr;
	{
		std::lock_guard<std::mutex> lock(managers_mutex);

		if (request.model_id >= 0 && request.model_id < managers.size()) {
			manager = managers[request.model_id];
		}
	}

	if (manager == nullptr) {
		std::stringstream errorMsg;
		errorMsg << "No model exists with ID " << request.model_id;
		clientapi::InferenceResponse response{
			ResponseHeader{request.header.user_request_id, clockworkError, errorMsg.str()}
		};
		callback(response);
		return;
	}

	if (manager->model.input_size() != request.input_size) {
		std::stringstream errorMsg;
		errorMsg << "Mismatched input size, expected " << manager->model.input_size() << ", got " << request.input_size;
		clientapi::InferenceResponse response{
			ResponseHeader{request.header.user_request_id, clockworkError, errorMsg.str()}
		};
		callback(response);
		return;
	}

	Request* r = new Request();
	r->user_request_id = request.header.user_request_id;
	r->model_id = request.model_id;
	r->input_size = request.input_size;
	r->input = static_cast<char*>(request.input);
	r->batch_size = request.batch_size;
	r->callback = [r, callback] () {
		clientapi::InferenceResponse response{
			ResponseHeader{r->user_request_id, clockworkSuccess, ""},
			r->model_id,
			r->batch_size,
			r->output_size,
			r->output
		};
		callback(response);
	};
	r->errback = [r, callback] (std::string message) {
		clientapi::InferenceResponse response{
			ResponseHeader{r->user_request_id, clockworkError, message}
		};
		callback(response);
	};

	return manager->submit(r);
}

void Worker::evict(clientapi::EvictRequest &request, std::function<void(clientapi::EvictResponse&)> callback) {
	ModelManager* manager = nullptr;
	{
		std::lock_guard<std::mutex> lock(managers_mutex);

		if (request.model_id >= 0 && request.model_id < managers.size()) {
			manager = managers[request.model_id];
		}
	}

	if (manager == nullptr) {
		std::stringstream errorMsg;
		errorMsg << "No model exists with ID " << request.model_id;
		clientapi::EvictResponse response{
			ResponseHeader{request.header.user_request_id, clockworkError, errorMsg.str()}
		};
		callback(response);
		return;
	}

	if (!manager->evict()) {
		std::stringstream errorMsg;
		errorMsg << "Unable to evict locked model " << request.model_id;
		clientapi::EvictResponse response{
			ResponseHeader{request.header.user_request_id, clockworkError, errorMsg.str()}
		};
		callback(response);
		return;
	}

	clientapi::EvictResponse response{
		ResponseHeader{request.header.user_request_id, clockworkSuccess, ""}
	};
	callback(response);
}

void Worker::loadRemoteModel(clientapi::LoadModelFromRemoteDiskRequest &request, std::function<void(clientapi::LoadModelFromRemoteDiskResponse&)> callback) {
	// Synchronous for now since this is not on critical path
	ModelManager* manager = nullptr;
	int id;
	{
		std::lock_guard<std::mutex> lock(managers_mutex);

		model::Model* model = model::Model::loadFromDisk(
			request.remote_path + ".1.so", // TODO: proper batch size here
			request.remote_path + ".1.clockwork",
			request.remote_path + ".clockwork_params"
		);
		id = managers.size();
		manager = new ModelManager(id, runtime, cache, model, logger);
		managers.push_back(manager);
	}

	clientapi::LoadModelFromRemoteDiskResponse response{
		ResponseHeader{request.header.user_request_id, clockworkSuccess, ""},
		id,
		manager->model.input_size(),
		manager->model.output_size()		
	};
	callback(response);
}