#include "clockwork/controller/controller.h"
#include <sstream>
#include "clockwork/thread.h"

using namespace clockwork;
using namespace clockwork::controller;
using namespace clockwork::controller::startup;


void Controller::ls(clientapi::LSRequest &request, std::function<void(clientapi::LSResponse&)> callback) {
	clientapi::LSResponse response;
	response.header.user_request_id = request.header.user_request_id;
	response.header.status = clockworkError;
	response.header.message = "ls not supported by this controller";
	callback(response);
}

float as_gb(size_t size) {
	return size / ((float) (1024*1024*1024));
}
float as_mb(size_t size) {
	return size / ((float) (1024*1024));	
}
float as_ms(uint64_t duration_nanos) {
	return duration_nanos / 1000000.0;
}

std::string BatchedModelState::str() {
	std::stringstream ss;
	ss.precision(1);
	ss << std::fixed;
	ss << "M-" << id
	   << " src=" << model_path
	   << " input=" << input_size
	   << " output=" << output_size
	   << " weights=" << as_mb(weights_size) << " MB (" << num_weights_pages << " pages)"
	   << " xfer=" << as_ms(weights_transfer_duration);
	for (auto &batch_size : supported_batch_sizes) {
		ss << " b" << batch_size << "=" << as_ms(exec_duration[batch_size]);
	}
	ss << std::endl;
	return ss.str();	
}

std::string GPUState::str() {
	std::stringstream ss;
	ss << "GPU " << id 
	   << " " << as_gb(weights_cache_size) << " GB (" << weights_cache_total_pages << " pages)"
	   << " " << loaded_models.size() << " loaded models" << std::endl;
	return ss.str();
}

std::string WorkerState::str() {
	std::stringstream ss;
	ss << "Worker " << id << " " << gpus.size() << " GPUs " << models.size() << " models" << std::endl;
	for (auto &gpu : gpus) {
		ss << gpu.str();
	}
	for (auto &p : models) {
		ss << p.second.str();
	}
	return ss.str();
	
}

std::string ClockworkState::str() {
	std::stringstream ss;
	ss << "Clockwork page_size=" << page_size << std::endl;
	for (auto &worker : workers) {
		ss << worker.str();
	}
	return ss.str();
	
}

void QueryWorkerStage::populate_model_state(BatchedModelState &model, workerapi::ModelInfo &info) {
	model.id = info.id;
	model.model_path = info.source;
	model.input_size = info.input_size;
	model.output_size = info.output_size;
	model.weights_size = info.weights_size;
	model.num_weights_pages = info.num_weights_pages;
	model.weights_transfer_duration = info.weights_load_time_nanos;
	model.supported_batch_sizes = info.supported_batch_sizes;
	for (unsigned i = 0; i < info.supported_batch_sizes.size(); i++) {
		auto batch_size = info.supported_batch_sizes[i];
		auto duration = info.batch_size_exec_times_nanos[i];
		model.exec_duration[batch_size] = duration;
	}
}

void QueryWorkerStage::populate_gpu_state(GPUState &gpu, workerapi::GPUInfo &info) {
	gpu.id = info.id;
	gpu.weights_cache_size = info.weights_cache_size;
	gpu.weights_cache_total_pages = info.weights_cache_total_pages;
	gpu.loaded_models = info.models;
}

void QueryWorkerStage::populate_worker_state(WorkerState &worker, workerapi::WorkerMemoryInfo &info) {
	for (auto &gpuinfo : info.gpus) {
		GPUState gpu;
		populate_gpu_state(gpu, gpuinfo);
		worker.gpus.push_back(gpu);
	}
	for (auto &modelinfo : info.models) {
		BatchedModelState model;
		populate_model_state(model, modelinfo);
		worker.models[model.id] = model;
	}
}

ClockworkState QueryWorkerStage::run(std::vector<network::controller::WorkerConnection*> workers,
		 		   tbb::concurrent_queue<std::shared_ptr<workerapi::Result>> &worker_results_queue) {
	std::unordered_set<unsigned> outstanding;

	// The state we will construct
	ClockworkState state;

	// Send actions to all workers
	for (auto &worker : workers) {
		// Create and send an action
		auto action = std::make_shared<workerapi::GetWorkerState>();
		action->id = outstanding.size();
		std::vector<std::shared_ptr<workerapi::Action>> actions{action};
		worker->sendActions(actions);

		// Save the action as outstanding
		outstanding.insert(action->id);

		// Create a WorkerState for this worker
		WorkerState workerstate;
		workerstate.id = action->id;
		state.workers.push_back(workerstate);
	}

	// Await results
	while (outstanding.size() > 0) {
		std::shared_ptr<workerapi::Result> result;
		while (!worker_results_queue.try_pop(result)) usleep(10000);

		// Check and remove action
		unsigned worker_id = result->id;
		auto it = outstanding.find(worker_id);
		CHECK(it != outstanding.end()) << "Received result for non-existent action " << result->str();

		// Validate result
		auto state_result = std::dynamic_pointer_cast<workerapi::GetWorkerStateResult>(result);
		CHECK(state_result) << "Fetching worker state failed " << result->str();

		// Process result
		if (outstanding.size() == workers.size()) {
			state.page_size = state_result->worker.page_size;
		}
		CHECK(state.page_size == state_result->worker.page_size) << "Found workers with inconsistent page sizes " << state_result->str();
		populate_worker_state(state.workers[worker_id], state_result->worker);

		// Remove outstanding action
		outstanding.erase(it);
	}

	return state;
}

LoadingStage::Pending::Pending(unsigned model_id, std::shared_ptr<LoadModelRequest> request) :
	model_id(model_id), request(request) {
}

void LoadingStage::Pending::add_action(unsigned worker_id, std::shared_ptr<workerapi::LoadModelFromDisk> action) {
	actions.push_back(action);
	action_worker_mapping[action->id] = worker_id;
}

void LoadingStage::Pending::result_received(std::shared_ptr<workerapi::Result> result) {
	std::cout << "Worker  --> " << result->str() << std::endl;
	results.push_back(result);
}

void LoadingStage::Pending::add_to_state(ClockworkState &state, std::shared_ptr<workerapi::LoadModelFromDiskResult> result) {
	for (unsigned j = 0; j < result->copies_created; j++) {
		BatchedModelState b;
		b.id = model_id + j;
		b.model_path = request->request.remote_path;
		b.input_size = result->input_size;
		b.output_size = result->output_size;
		b.weights_size = result->weights_size_in_cache;
		b.num_weights_pages = result->num_weights_pages;
		b.weights_transfer_duration = result->weights_load_time_nanos;
		b.supported_batch_sizes = result->supported_batch_sizes;

		for (unsigned i = 0; i < result->supported_batch_sizes.size(); i++) {
			auto batch_size = result->supported_batch_sizes[i];
			auto duration = result->batch_size_exec_times_nanos[i];
			b.exec_duration[batch_size] = duration;
		}

		unsigned worker_id = action_worker_mapping[result->id];
		state.workers[worker_id].models[model_id + j] = b;
	}
}

void LoadingStage::Pending::check_results(std::shared_ptr<workerapi::LoadModelFromDiskResult> a, std::shared_ptr<workerapi::LoadModelFromDiskResult> b) {
	CHECK(a->input_size == b->input_size) << "Inconsistent input_size in LoadModelFromDiskResult instances";
	CHECK(a->output_size == b->output_size) << "Inconsistent output_size in LoadModelFromDiskResult instances";
	CHECK(a->supported_batch_sizes == b->supported_batch_sizes) << "Inconsistent supported_batch_sizes in LoadModelFromDiskResult instances";
	CHECK(a->weights_size_in_cache == b->weights_size_in_cache) << "Inconsistent weights_size_in_cache in LoadModelFromDiskResult instances";
}

void LoadingStage::Pending::check_completion(ClockworkState &state) {
	if (results.size() < actions.size()) return;

	std::vector<std::shared_ptr<workerapi::ErrorResult>> errors;
	std::vector<std::shared_ptr<workerapi::LoadModelFromDiskResult>> loadresults;
	std::vector<std::shared_ptr<workerapi::Result>> unexpected;

	for (auto &result : results) {
		if (auto error = std::dynamic_pointer_cast<workerapi::ErrorResult>(result)) {
			errors.push_back(error);
		} else if (auto lresult = std::dynamic_pointer_cast<workerapi::LoadModelFromDiskResult>(result)) {
			loadresults.push_back(lresult);
		} else {
			unexpected.push_back(result);
		}
	}

	clientapi::LoadModelFromRemoteDiskResponse response;
	if (errors.size() == results.size()) {
		response.header.user_request_id = request->request.header.user_request_id;
		response.header.status = clockworkError;
		response.header.message = errors[0]->message;
	} else if (loadresults.size() == results.size()) {
		response.header.status = clockworkSuccess;
		response.header.message = "";
		response.model_id = model_id;
		response.copies_created = loadresults[0]->copies_created;
		response.input_size = loadresults[0]->input_size;
		response.output_size = loadresults[0]->output_size;

		for (unsigned i = 1; i < loadresults.size(); i++) {
			check_results(loadresults[0], loadresults[i]);
		}

		// Success; add to state
		for (auto &result : loadresults) {
			add_to_state(state, result);
		}
	} else {
		CHECK(unexpected.size() == 0) << "Received unexpected results from Clockwork worker, e.g. " << unexpected[0]->str();
		CHECK(false) << "Received " << errors.size() << " errors and " << loadresults.size() << " success for request " << request->request.str();
	}

	std::cout << "Client <--  " << response.str() << std::endl;

	request->callback(response);
}

LoadingStage::Worker::Worker(network::controller::WorkerConnection* worker) : worker(worker) {}

void LoadingStage::Worker::add_action(std::shared_ptr<workerapi::LoadModelFromDisk> action) {
	action_queue.push(action);
	check();
}

void LoadingStage::Worker::result_received() {
	outstanding--;
	check();
}

void LoadingStage::Worker::check() {
	if (outstanding == max_outstanding || action_queue.empty()) {
		return;
	}

	std::vector<std::shared_ptr<workerapi::Action>> actions;
	while (outstanding < max_outstanding && !action_queue.empty()) {
		actions.push_back(action_queue.front());
		action_queue.pop();
		outstanding++;
	}

	for (auto &action : actions) {
		std::cout << "Worker <--  " << action->str() << std::endl;
	}

	worker->sendActions(actions);
}

LoadingStage::LoadingStage(
	ClockworkState &state, 
	std::vector<network::controller::WorkerConnection*> worker_connections,
	uint64_t timeout, unsigned max_batch_size, uint64_t max_exec_duration) : 
		timeout(timeout), state(state), max_batch_size(max_batch_size), 
		max_exec_duration(max_exec_duration) {
	// Determine model_id_seed
	for (auto &worker : state.workers) {
		for (auto &p : worker.models) {
			model_id_seed = std::max(model_id_seed, p.first+1);
		}
	}

	for (auto &connection : worker_connections) {
		workers.push_back(Worker(connection));
	}

	// If the workers already have some models loaded, make sure we start from a higher ID
	for (auto &worker : state.workers) {
		for (auto &p : worker.models) {
			model_id_seed = std::max(p.first+1, model_id_seed);
		}
	}
}

void LoadingStage::on_request(std::shared_ptr<LoadModelRequest> &request) {
	int id = model_id_seed;
	model_id_seed += request->request.no_of_copies;
	std::shared_ptr<Pending> p = std::make_shared<Pending>(id, request);

	for (unsigned i = 0; i < workers.size(); i++) {
		auto load_model = std::make_shared<workerapi::LoadModelFromDisk>();
		load_model->id = action_id_seed++;
		load_model->model_id = p->model_id;
		load_model->model_path = request->request.remote_path;
		load_model->no_of_copies = request->request.no_of_copies;
		load_model->earliest = 0;
		load_model->latest = ULONG_MAX;
		load_model->max_batch_size = max_batch_size;
		load_model->max_exec_duration = max_exec_duration;

		p->add_action(i, load_model);
		workers[i].add_action(load_model);
		actions.insert(std::make_pair<unsigned, PendingAction>(load_model->id, PendingAction{workers[i], p}));
	}
}

void LoadingStage::on_result(std::shared_ptr<workerapi::Result> &result) {
	auto it = actions.find(result->id);
	CHECK(it != actions.end()) << "Received a result for a non-existent action " << result->str();

	auto &p = it->second;
	p.worker.result_received();
	p.pending->result_received(result);
	p.pending->check_completion(state);

	actions.erase(it);
}

bool LoadingStage::is_loading_stage_complete() {
	return last_action != 0 &&
		   actions.size() == 0 &&
		   (last_action + timeout) < util::now();
}	

ClockworkState LoadingStage::run(tbb::concurrent_queue<std::shared_ptr<LoadModelRequest>> &load_model_request_queue,
		 		   tbb::concurrent_queue<std::shared_ptr<workerapi::Result>> &worker_results_queue) {
	uint64_t warn_at = 0;
	while (!is_loading_stage_complete()) {
		// Check for new requests
		std::shared_ptr<LoadModelRequest> request = nullptr;
		while (load_model_request_queue.try_pop(request)) {
			if (last_action == 0) {
				std::cout << "(Startup-4) LoadModelStage has begun" << std::endl;
			}
			on_request(request);
			last_action = util::now();
			warn_at = std::max(warn_at, last_action + (timeout / 2));
		}

		// Check for results
		std::shared_ptr<workerapi::Result> result = nullptr;
		while (worker_results_queue.try_pop(result)) {
			on_result(result);
			last_action = util::now();
			warn_at = std::max(warn_at, last_action + (timeout / 2));
		}

		// Print a timeout-warning
		uint64_t now = util::now();
		if (warn_at != 0 && actions.size() == 0 && (last_action + timeout) > now && warn_at < now) {
			uint64_t seconds_remaining = (last_action + timeout - now) / 1000000000;
			std::stringstream ss;
			ss << std::fixed << "(Startup-5) LoadModelStage ending in " << seconds_remaining << " seconds...";
			std::cout << ss.str() << "\r" << std::flush;
			warn_at += 1000000000UL; // Warn every second until end
		}
	}

	return state;
}

void ControllerStartup::bounceLSRequest(std::shared_ptr<LSRequest> &request) {
	clientapi::LSResponse response;
	response.header.user_request_id = request->request.header.user_request_id;
	response.header.status = clockworkInitializing;
	response.header.message = "Controller initializing";

	std::cout << "Client <--  " << response.str() << std::endl;
	request->callback(response);
}

void ControllerStartup::bounceInferRequest(std::shared_ptr<InferRequest> &request) {
	clientapi::InferenceResponse response;
	response.header.user_request_id = request->request.header.user_request_id;
	response.header.status = clockworkInitializing;
	response.header.message = "Controller initializing";
	response.output_size = 0;
	response.output = nullptr;

	std::cout << "Client <--  " << response.str() << std::endl;
	request->callback(response);
}

void ControllerStartup::bounceLoadModelRequest(std::shared_ptr<LoadModelRequest> &request) {
	clientapi::LoadModelFromRemoteDiskResponse response;
	response.header.user_request_id = request->request.header.user_request_id;
	response.header.status = clockworkInvalidRequest;
	response.header.message = "loadModel phase has completed";

	std::cout << "Client <--  " << response.str() << std::endl;
	request->callback(response);
}

ClockworkState ControllerStartup::run(uint64_t timeout, std::vector<network::controller::WorkerConnection*> workers) {
	// Create a fetcher, call run directly, receive loaded model info
	// Create a loader, call run directly, pass it loaded model info, receive loaded model info
	// Create a profiler, call run directly, pass it loaded model info, receive profiled model info
	// Return profiled model info

	std::cout << "(Startup) Running ControllerStartup" << std::endl;

	// Immediately bounce infer requests

	std::cout << "(Startup-1) Bouncing LS and Infer requests until startup is complete" << std::endl;
	Bouncer<std::shared_ptr<InferRequest>> infer_bouncer(infer_request_queue, std::bind(&ControllerStartup::bounceInferRequest, this, std::placeholders::_1));
	Bouncer<std::shared_ptr<LSRequest>> ls_bouncer(ls_request_queue, std::bind(&ControllerStartup::bounceLSRequest, this, std::placeholders::_1));

	// Let loadModel requests buffer while querying worker state
	std::cout << "(Startup-2) Querying current worker state" << std::endl;
	ClockworkState state = QueryWorkerStage().run(workers, worker_results_queue);

	std::cout << state.str() << std::endl;

	// TODO: fetch existing models
	// // Create and run fetcher
	// Fetcher f();
	// LoadedModels loaded = f.run(workers, worker_results_queue);

	// Create and run loader

	std::cout << "(Startup-3) Awaiting LoadModel requests from clients" << std::endl;
	state = LoadingStage(state, workers, timeout, max_batch_size, max_exec_duration).run(load_model_request_queue, worker_results_queue);
	std::cout << "(Startup-6) LoadModelStage complete.  Printing loaded models: " << std::endl;
	std::cout << state.str() << std::endl;


	std::cout << "(Startup-end) Transitioning to scheduler" << std::endl;

	infer_bouncer.shutdown();
	ls_bouncer.shutdown();
	return state;
}

/*
Initiates an orderly shutdown of the ControllerStartup.  Call this after calling `run`.

The main purpose of this function is to drain any pending request queues
*/
void ControllerStartup::shutdown() {
	// There shouldn't be any pending worker results
	std::shared_ptr<workerapi::Result> result;
	CHECK(!worker_results_queue.try_pop(result)) << "Found pending worker results during Controller transition";

	// Bouncers are already shut down but stuff could have come in
	std::shared_ptr<InferRequest> infer = nullptr;
	while (infer_request_queue.try_pop(infer)) {
		bounceInferRequest(infer);
	}

	// Bouncers are already shut down but stuff could have come in
	std::shared_ptr<LSRequest> ls = nullptr;
	while (ls_request_queue.try_pop(ls)) {
		bounceLSRequest(ls);
	}

	// Bouncers are already shut down but stuff could have come in
	std::shared_ptr<LoadModelRequest> load_model = nullptr;
	while (load_model_request_queue.try_pop(load_model)) {
		bounceLoadModelRequest(load_model);
	}

	std::cout << " * Admitting inference requests" << std::endl;
}

void ControllerStartup::infer(clientapi::InferenceRequest &request, 
		   std::function<void(clientapi::InferenceResponse&)> callback) {
	std::cout << "Client  --> " << request.str() << std::endl;
	infer_request_queue.push(std::make_shared<InferRequest>(request, callback));
}

void ControllerStartup::loadRemoteModel(clientapi::LoadModelFromRemoteDiskRequest &request, 
					 std::function<void(clientapi::LoadModelFromRemoteDiskResponse&)> callback) {
	std::cout << "Client  --> " << request.str() << std::endl;
	load_model_request_queue.push(std::make_shared<LoadModelRequest>(request, callback));
}

void ControllerStartup::ls(clientapi::LSRequest &request, std::function<void(clientapi::LSResponse&)> callback) {
	std::cout << "Client  --> " << request.str() << std::endl;
	ls_request_queue.push(std::make_shared<LSRequest>(request, callback));
}

void ControllerStartup::sendResult(std::shared_ptr<workerapi::Result> result) {
	worker_results_queue.push(result);
}

ControllerWithStartupPhase::ControllerWithStartupPhase(
			int client_port, 
			std::vector<std::pair<std::string, std::string>> worker_host_port_pairs,
			uint64_t load_stage_timeout,
			ControllerStartup* startup,
			Scheduler* scheduler,
			RequestTelemetryLogger* request_telemetry
		) : 
		Controller(client_port, worker_host_port_pairs),
		timeout(load_stage_timeout),
		startup(startup),
		scheduler(scheduler),
		startup_thread(&ControllerWithStartupPhase::runStartup, this),
		request_telemetry(request_telemetry) {
	threading::initLowPriorityThread(startup_thread);
}

void ControllerWithStartupPhase::runStartup() {
	this->state = startup->run(timeout, workers);

	std::lock_guard<std::mutex> lock(startup_mutex);

	scheduler->start(this->workers, state);

	startup_phase = false;
	startup->shutdown();
	delete startup;
	startup = nullptr;
}

void ControllerWithStartupPhase::sendResult(std::shared_ptr<workerapi::Result> result) {
	if (startup_phase) {
		std::lock_guard<std::mutex> lock(startup_mutex);

		if (startup_phase) {
			startup->sendResult(result);
			return;
		}
	}
	scheduler->resultFromWorker(result);
}

void ControllerWithStartupPhase::infer(clientapi::InferenceRequest &request, std::function<void(clientapi::InferenceResponse&)> callback) {
	if (startup_phase) {
		std::lock_guard<std::mutex> lock(startup_mutex);

		if (startup_phase) {
			startup->infer(request, callback);
			return;
		}
	}
	if (request_telemetry != nullptr) {
		ControllerRequestTelemetry* telemetry = new ControllerRequestTelemetry();
		telemetry->set(request);
		scheduler->clientInfer(request, [this, telemetry, callback](clientapi::InferenceResponse &response) {
			telemetry->set(response);
			callback(response);
			request_telemetry->log(*telemetry);
			delete telemetry;
		});
	} else {
		scheduler->clientInfer(request, callback);
	}
}

void ControllerWithStartupPhase::loadRemoteModel(clientapi::LoadModelFromRemoteDiskRequest &request, std::function<void(clientapi::LoadModelFromRemoteDiskResponse&)> callback) {
	if (startup_phase) {
		std::lock_guard<std::mutex> lock(startup_mutex);

		if (startup_phase) {
			startup->loadRemoteModel(request, callback);
			return;
		}
	} else {
		clientapi::LoadModelFromRemoteDiskResponse response;
		response.header.user_request_id = request.header.user_request_id;
		response.header.status = clockworkInvalidRequest;
		response.header.message = "Controller startup phase has completed";
		callback(response);
	}
}

void ControllerWithStartupPhase::ls(clientapi::LSRequest &request, std::function<void(clientapi::LSResponse&)> callback) {
	if (startup_phase) {
		std::lock_guard<std::mutex> lock(startup_mutex);

		if (startup_phase) {
			startup->ls(request, callback);
			return;
		}
	}

	// Extract the model info, preventing duplicates
	std::map<unsigned, clientapi::ClientModelInfo> models;
	for (auto &worker : state.workers) {
		for (auto &p : worker.models) {
			auto &model = p.second;
			if (models.find(model.id) == models.end()) {
				clientapi::ClientModelInfo info;
				info.model_id = model.id;
				info.remote_path = model.model_path;
				info.input_size = model.input_size;
				info.output_size = model.output_size;
				models[model.id] = info;
			}
		}
	}

	// Send response
	clientapi::LSResponse response;
	response.header.user_request_id = request.header.user_request_id;
	response.header.status = clockworkSuccess;
	for (auto &p : models) {
		response.models.push_back(p.second);
	}

	callback(response);
}

void ControllerWithStartupPhase::evict(clientapi::EvictRequest &request, std::function<void(clientapi::EvictResponse&)> callback) {
	CHECK(false) << "evict not supported by ControllerWithStartupPhase";
}

// clientapi -- requests from clients call these functions
void ControllerWithStartupPhase::uploadModel(clientapi::UploadModelRequest &request, std::function<void(clientapi::UploadModelResponse&)> callback) {
	CHECK(false) << "uploadModel not supported by ControllerWithStartupPhase";
}
