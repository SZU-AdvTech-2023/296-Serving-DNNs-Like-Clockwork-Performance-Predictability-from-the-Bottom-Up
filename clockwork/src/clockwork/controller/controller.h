#ifndef _CLOCKWORK_CONTROLLER_CONTROLLER_H_
#define _CLOCKWORK_CONTROLLER_CONTROLLER_H_

#include "clockwork/controller/scheduler.h"
#include "clockwork/network/controller.h"
#include "clockwork/api/worker_api.h"
#include "clockwork/telemetry/controller_request_logger.h"
#include <limits>
#include <algorithm>

namespace clockwork {

class Controller : public workerapi::Controller, public clientapi::ClientAPI {
public:
	network::controller::Server* client_facing_server;
	network::controller::WorkerManager* worker_manager;
	std::vector<network::controller::WorkerConnection*> workers;
	Controller(int client_port, std::vector<std::pair<std::string, std::string>> worker_host_port_pairs) {
		client_facing_server = new network::controller::Server(this, client_port);
		worker_manager = new network::controller::WorkerManager();

		for (auto host_port_pair : worker_host_port_pairs) {
			network::controller::WorkerConnection* connection = worker_manager->connect(host_port_pair.first, host_port_pair.second, this);
			workers.push_back(connection);
		}
	}

	void shutdown(bool awaitShutdown = false) {
		// TODO
		if (awaitShutdown) {
			join();
		}
	}

	void join() {
		// TODO
		worker_manager->join();
	}

	// workerapi -- results received from workers call these functions
	virtual void sendResult(std::shared_ptr<workerapi::Result> result) = 0;

	// clientapi -- requests from clients call these functions
	virtual void uploadModel(clientapi::UploadModelRequest &request, std::function<void(clientapi::UploadModelResponse&)> callback) = 0;
	virtual void infer(clientapi::InferenceRequest &request, std::function<void(clientapi::InferenceResponse&)> callback) = 0;
	virtual void evict(clientapi::EvictRequest &request, std::function<void(clientapi::EvictResponse&)> callback) = 0;
	virtual void loadRemoteModel(clientapi::LoadModelFromRemoteDiskRequest &request, std::function<void(clientapi::LoadModelFromRemoteDiskResponse&)> callback) = 0;
	virtual void ls(clientapi::LSRequest &request, std::function<void(clientapi::LSResponse&)> callback);
};

namespace controller {
namespace startup {

template<typename ReqType, typename RspType> class Request {
public:
	uint64_t arrival;
	ReqType request;
	std::function<void(RspType&)> callback;
	Request(ReqType &request, std::function<void(RspType&)> &callback) : 
		arrival(util::now()), request(request), callback(callback) {
	}
};
typedef Request<clientapi::InferenceRequest, clientapi::InferenceResponse> InferRequest;
typedef Request<clientapi::LoadModelFromRemoteDiskRequest, clientapi::LoadModelFromRemoteDiskResponse> LoadModelRequest;
typedef Request<clientapi::LSRequest, clientapi::LSResponse> LSRequest;

/* Handles fetching information from workers about currently-loaded models */
class QueryWorkerStage {
public:

	QueryWorkerStage() {}

	void populate_model_state(BatchedModelState &model, workerapi::ModelInfo &info);
	void populate_gpu_state(GPUState &gpu, workerapi::GPUInfo &info);
	void populate_worker_state(WorkerState &worker, workerapi::WorkerMemoryInfo &info);

	ClockworkState run(std::vector<network::controller::WorkerConnection*> workers,
			 		   tbb::concurrent_queue<std::shared_ptr<workerapi::Result>> &worker_results_queue);
};

/* Handles model loading in the Controller Startup phase */
class LoadingStage {
public:
	class Pending {
	public:
		unsigned model_id;
		std::shared_ptr<LoadModelRequest> request;
		std::unordered_map<unsigned, unsigned> action_worker_mapping;
		std::vector<std::shared_ptr<workerapi::LoadModelFromDisk>> actions;
		std::vector<std::shared_ptr<workerapi::Result>> results;

		Pending(unsigned model_id, std::shared_ptr<LoadModelRequest> request);

		void add_action(unsigned worker_id, std::shared_ptr<workerapi::LoadModelFromDisk> action);
		void result_received(std::shared_ptr<workerapi::Result> result);
		void add_to_state(ClockworkState &state, std::shared_ptr<workerapi::LoadModelFromDiskResult> result);
		void check_results(std::shared_ptr<workerapi::LoadModelFromDiskResult> a, std::shared_ptr<workerapi::LoadModelFromDiskResult> b);
		void check_completion(ClockworkState &state);
	};

	class Worker {
	public:
		network::controller::WorkerConnection* worker;
		std::queue<std::shared_ptr<workerapi::LoadModelFromDisk>> action_queue;
		unsigned outstanding = 0;
		unsigned max_outstanding = 4;

		Worker(network::controller::WorkerConnection* worker);

		void add_action(std::shared_ptr<workerapi::LoadModelFromDisk> action);
		void result_received();

		void check();
	};

	unsigned action_id_seed = 0;
	unsigned model_id_seed = 0;

	uint64_t last_action = 0;
	uint64_t timeout; // For now, 10s hard-coded loading stage timeout

	ClockworkState state;

	struct PendingAction {
		Worker &worker;
		std::shared_ptr<Pending> pending;
	};

	std::unordered_map<unsigned, PendingAction> actions;
	std::vector<Worker> workers;
	unsigned max_batch_size;
	uint64_t max_exec_duration;


	LoadingStage(
		ClockworkState &state, 
		std::vector<network::controller::WorkerConnection*> worker_connections,
		uint64_t timeout = 10000000000UL, 
		unsigned max_batch_size = 32, 
		uint64_t max_exec_duration = 1000000000UL
	);

	void on_request(std::shared_ptr<LoadModelRequest> &request);
	void on_result(std::shared_ptr<workerapi::Result> &result);
	bool is_loading_stage_complete();
	ClockworkState run(tbb::concurrent_queue<std::shared_ptr<LoadModelRequest>> &load_model_request_queue,
			 		   tbb::concurrent_queue<std::shared_ptr<workerapi::Result>> &worker_results_queue);
};
}


/*
ControllerStartup handles three steps of the Controller starting up before handing over to the Scheduler:

(1) Fetches information about currently-loaded models from the workers
(2) Loads models requested by clients during an initial Loading stage
(3) After a pre-configured period of inactivity, transitions to a Profiling stage to collect
    initial statistics about models
(4) Once profiling is complete, switches over to the scheduler, passing the profiled model information

During startup, infer requests will fail with clockworkInitializing.  To prevent spamming from clients,
requests will time out before returning clockworkInitializing.

After transitioning to the scheduler, loadModelFromDisk requests will return clockworkInvalidRequest.
*/
class ControllerStartup {
public:

	/* Simple class that runs a thread, bouncing requests
	   Expects template parameter T to be a Request type */
	template<typename T> class Bouncer {
	public:
		uint64_t timeout_nanos = 12000000000UL; // 12 seconds
		std::atomic_bool alive = true;
		tbb::concurrent_queue<T> &queue;
		std::function<void(T&)> bounce;
		std::thread thread;
		Bouncer(tbb::concurrent_queue<T> &queue, std::function<void(T&)> bounce) :
		    queue(queue), bounce(bounce), thread(&Bouncer<T>::run, this) {
		}

		void run() {
			while (alive) {
				usleep(10000);

				T next;
				while (queue.try_pop(next)) {
					while (alive && (next->arrival + timeout_nanos) > util::now()) {
						usleep(10000);
					}
					bounce(next);
				}
			}
		}

		void shutdown() {
			alive = false;
			thread.join();
		}
	};

	unsigned max_batch_size;
	uint64_t max_exec_duration;

	ControllerStartup(unsigned max_batch_size = 32, uint64_t max_exec_duration = 1000000000UL)
		: max_batch_size(max_batch_size), max_exec_duration(max_exec_duration) {}

	void bounceLSRequest(std::shared_ptr<startup::LSRequest> &request);
	void bounceInferRequest(std::shared_ptr<startup::InferRequest> &request);
	void bounceLoadModelRequest(std::shared_ptr<startup::LoadModelRequest> &request);

	tbb::concurrent_queue<std::shared_ptr<startup::InferRequest>> infer_request_queue;
	tbb::concurrent_queue<std::shared_ptr<startup::LoadModelRequest>> load_model_request_queue;
	tbb::concurrent_queue<std::shared_ptr<startup::LSRequest>> ls_request_queue;
	tbb::concurrent_queue<std::shared_ptr<workerapi::Result>> worker_results_queue;

	/*
	Runs the whole startup stage.  Blocks until complete.
	*/
	ClockworkState run(uint64_t timeout, std::vector<network::controller::WorkerConnection*> workers);

	/*
	Initiates an orderly shutdown of the ControllerStartup.  Call this after calling `run`.

	The main purpose of this function is to drain any pending request queues
	*/
	void shutdown();

	void infer(clientapi::InferenceRequest &request, std::function<void(clientapi::InferenceResponse&)> callback);
	void loadRemoteModel(clientapi::LoadModelFromRemoteDiskRequest &request, std::function<void(clientapi::LoadModelFromRemoteDiskResponse&)> callback);
	void ls(clientapi::LSRequest &request, std::function<void(clientapi::LSResponse&)> callback);
	void sendResult(std::shared_ptr<workerapi::Result> result);

};

class ControllerWithStartupPhase : public Controller {
private:
	RequestTelemetryLogger* request_telemetry;
	bool startup_phase = true;
	uint64_t timeout;
	ControllerStartup* startup;
	std::thread startup_thread;
	std::mutex startup_mutex;

	ClockworkState state;
	Scheduler* scheduler;

public:

	ControllerWithStartupPhase(
				int client_port, 
				std::vector<std::pair<std::string, std::string>> worker_host_port_pairs,
				uint64_t load_stage_timeout,
				ControllerStartup* startup,
				Scheduler* scheduler,
				RequestTelemetryLogger* request_telemetry = nullptr
			);

	void runStartup();
	void sendResult(std::shared_ptr<workerapi::Result> result);
	void infer(clientapi::InferenceRequest &request, std::function<void(clientapi::InferenceResponse&)> callback);
	void loadRemoteModel(clientapi::LoadModelFromRemoteDiskRequest &request, std::function<void(clientapi::LoadModelFromRemoteDiskResponse&)> callback);
	void evict(clientapi::EvictRequest &request, std::function<void(clientapi::EvictResponse&)> callback);
	void uploadModel(clientapi::UploadModelRequest &request, std::function<void(clientapi::UploadModelResponse&)> callback);
	void ls(clientapi::LSRequest &request, std::function<void(clientapi::LSResponse&)> callback);
};
}
}

#endif 

