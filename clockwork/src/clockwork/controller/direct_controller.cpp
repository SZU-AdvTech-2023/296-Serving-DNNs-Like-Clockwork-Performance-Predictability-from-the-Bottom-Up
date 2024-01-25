#include <atomic>
#include "clockwork/network/controller.h"
#include "direct_controller.h"
#include "clockwork/api/client_api.h"
#include "clockwork/api/worker_api.h"
#include <cstdlib>
#include <unistd.h>
#include <libgen.h>
#include "clockwork/test/util.h"
#include <nvml.h>
#include <iostream>
#include "clockwork/util.h"
#include <functional>
#include <memory>
#include <unordered_map>
#include <dmlc/logging.h>

using namespace clockwork;

DirectControllerImpl::DirectControllerImpl(int client_port, std::vector<std::pair<std::string, std::string>> worker_host_port_pairs) :
	Controller(client_port, worker_host_port_pairs), model_id_seed(0), action_id_seed(0) {}

void DirectControllerImpl::save_callback(int id, std::function<void(std::shared_ptr<workerapi::Result>)> callback) {
	std::lock_guard<std::mutex> lock(actions_mutex);

	auto it = action_callbacks.find(id);
	CHECK(it == action_callbacks.end()) << "ID " << id << " already exists";

	action_callbacks.insert(std::make_pair(id, callback));
}

std::function<void(std::shared_ptr<workerapi::Result>)> DirectControllerImpl::get_callback(int id) {
	std::lock_guard<std::mutex> lock(DirectControllerImpl::actions_mutex);

	auto it = DirectControllerImpl::action_callbacks.find(id);

	if (it == action_callbacks.end()) return nullptr;

	auto callback = it->second;
	DirectControllerImpl::action_callbacks.erase(it);
	return callback;
}

// workerapi::Controller::sendResult
void DirectControllerImpl::sendResult(std::shared_ptr<workerapi::Result> result) {
	auto callback = get_callback(result->id);
	CHECK(callback != nullptr) << "Received result without a corresponding action: " << result->str();
	callback(result);
}

void DirectControllerImpl::uploadModel(clientapi::UploadModelRequest &request, std::function<void(clientapi::UploadModelResponse&)> callback) {
	std::cout << "Client  -> " << request.str() << std::endl;
	clientapi::UploadModelResponse response;
	std::cout << "Client <-  " << response.str() << std::endl;
	callback(response);
}

void DirectControllerImpl::infer(clientapi::InferenceRequest &request, std::function<void(clientapi::InferenceResponse&)> callback) {
	std::cout << "Client  -> " << request.str() << std::endl;

	int user_request_id = request.header.user_request_id;
	int model_id = request.model_id;
	std::vector<std::shared_ptr<workerapi::Action>> actions;

	unsigned gpu_id = 0;
	if (user_request_id % 2 == 0) {
		gpu_id = 1;
	}

	// If weights aren't loaded, send a load_weights action
	if (weights_available_at.find(std::make_pair(request.model_id, gpu_id)) == weights_available_at.end()) {
		auto load_weights = std::make_shared<workerapi::LoadWeights>();
		load_weights->id = action_id_seed++;
		load_weights->model_id = request.model_id;
		load_weights->gpu_id = gpu_id;
		load_weights->earliest = util::now();
		load_weights->latest = load_weights->earliest + 10000000000L;

		auto load_weights_complete = [this, user_request_id, model_id, gpu_id] (std::shared_ptr<workerapi::Result> result) {
			std::cout << "Worker  -> " << result->str() << std::endl;

			// If the result wasn't a success, then mark the weights as unavailble again
			if (auto error_result = std::dynamic_pointer_cast<workerapi::ErrorResult>(result)) {
				weights_available_at.erase(std::make_pair(model_id, gpu_id));
			}
		};
		save_callback(load_weights->id, load_weights_complete);
		actions.push_back(load_weights);

		// Store when the weights will be available
		weights_available_at[std::make_pair(model_id, gpu_id)] = util::now() + 7000000UL; // Assume it'll take 7 ms to load weights
	}
		
	// Translate clientapi request into a workerapi action
	auto infer = std::make_shared<workerapi::Infer>();
	infer->id = action_id_seed++;
	infer->model_id = request.model_id;
	infer->gpu_id = gpu_id;
	infer->batch_size = request.batch_size;
	infer->input_size = request.input_size;
	infer->input = static_cast<char*>(request.input);
	infer->earliest = weights_available_at[std::make_pair(model_id, gpu_id)];
	infer->latest = util::now() + 10000000000UL;

	// When the infer result is received, call this callback
	auto infer_complete = [this, callback, user_request_id, model_id] (std::shared_ptr<workerapi::Result> result) {
		std::cout << "Worker  -> " << result->str() << std::endl;

		// Translate workerapi result into a clientapi response
		clientapi::InferenceResponse response;
		response.header.user_request_id = user_request_id;
		if (auto error = std::dynamic_pointer_cast<workerapi::ErrorResult>(result)) {
			response.header.status = clockworkError;
			response.header.message = error->message;
			response.output_size = 0;
			response.output = nullptr;
		} else if (auto infer_result = std::dynamic_pointer_cast<workerapi::InferResult>(result)) {
			response.header.status = clockworkSuccess;
			response.header.message = "";
			response.model_id = model_id;
			response.batch_size = 1;
			response.output_size = infer_result->output_size;
			response.output = infer_result->output;
		} else {
			response.header.status = clockworkError;
			response.header.message = "Internal Controller Error";
			response.output_size = 0;
			response.output = nullptr;
		}

		// Send the response
		std::cout << "Client <-  " << response.str() << std::endl;
		callback(response);
	};
	DirectControllerImpl::save_callback(infer->id, infer_complete);
	actions.push_back(infer);

	// Send the action
	for (auto &action : actions) {
		std::cout << "Worker <-  " << action->str() << std::endl;
	}
	DirectControllerImpl::workers[0]->sendActions(actions);
}

/** This is a 'backdoor' API function for ease of experimentation */
void DirectControllerImpl::evict(clientapi::EvictRequest &request, std::function<void(clientapi::EvictResponse&)> callback) {
	std::cout << "Client  -> " << request.str() << std::endl;
	clientapi::EvictResponse response;
	std::cout << "Client <-  " << response.str() << std::endl;
	callback(response);
}

/** This is a 'backdoor' API function for ease of experimentation */
void DirectControllerImpl::loadRemoteModel(clientapi::LoadModelFromRemoteDiskRequest &request, std::function<void(clientapi::LoadModelFromRemoteDiskResponse&)> callback) {
	std::cout << "Client  -> " << request.str() << std::endl;

	int user_request_id = request.header.user_request_id;
	int action_id = DirectControllerImpl::action_id_seed++;
	int model_id = DirectControllerImpl::model_id_seed++;
	int no_of_copies = request.no_of_copies;
	DirectControllerImpl::model_id_seed += no_of_copies - 1;

	// Translate clientapi request into a workerapi action
	auto load_model = std::make_shared<workerapi::LoadModelFromDisk>();
	load_model->id = action_id;
	load_model->model_id = model_id;
	load_model->no_of_copies = no_of_copies;
	load_model->model_path = request.remote_path;
	load_model->earliest = 0;
	load_model->latest = util::now() + 10000000000UL;

	// When the result is received, call this callback
	auto result_callback = [this, callback, user_request_id, model_id, no_of_copies] (std::shared_ptr<workerapi::Result> result) {
		std::cout << "Worker  -> " << result->str() << std::endl;

		// Translate workerapi result into a clientapi response
		clientapi::LoadModelFromRemoteDiskResponse response;
		response.header.user_request_id = user_request_id;
		if (auto error = std::dynamic_pointer_cast<workerapi::ErrorResult>(result)) {
			response.header.status = clockworkError;
			response.header.message = error->message;
		} else if (auto load_model_result = std::dynamic_pointer_cast<workerapi::LoadModelFromDiskResult>(result)) {
			response.header.status = clockworkSuccess;
			response.header.message = "";
			response.model_id = model_id;
			response.copies_created = load_model_result->copies_created;
			response.input_size = load_model_result->input_size;
			response.output_size = load_model_result->output_size;
		} else {
			response.header.status = clockworkError;
			response.header.message = "Internal Controller Error";
		}

		// Send the response
		std::cout << "Client <-  " << response.str() << std::endl;
		callback(response);
	};
	DirectControllerImpl::save_callback(action_id, result_callback);

	// Send the action
	std::cout << "Worker <-  " << load_model->str() << std::endl;
	std::vector<std::shared_ptr<workerapi::Action>> actions;
	actions = {load_model};
	DirectControllerImpl::workers[0]->sendActions(actions);
}

