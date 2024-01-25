#ifndef _CLOCKWORK_CONTROLLER_SCHEDULER_H_
#define _CLOCKWORK_CONTROLLER_SCHEDULER_H_

#include <map>
#include <string>
#include <vector>
#include "clockwork/api/client_api.h"
#include "clockwork/api/worker_api.h"
#include "clockwork/network/controller.h"

namespace clockwork {

struct BatchedModelState {
	unsigned id;
	std::string model_path;
	size_t input_size;
	size_t output_size;
	size_t weights_size; // Total size or size in pages?
	unsigned num_weights_pages;
	uint64_t weights_transfer_duration;
	std::vector<unsigned> supported_batch_sizes;
	std::map<unsigned, uint64_t> exec_duration; // map of batch size to exec duration

	std::string str();
};

struct GPUState {
  unsigned id;
  size_t weights_cache_size;
  unsigned weights_cache_total_pages;   // Number of pages in GPU weights cache
  std::vector<unsigned> loaded_models;  // Models loaded into GPU memory

  std::string str();
};

struct WorkerState {
  unsigned id;
  std::vector<GPUState> gpus;
  std::map<unsigned, BatchedModelState> models;

  std::string str();
};

struct ClockworkState {
  size_t page_size;
  std::vector<WorkerState> workers;

  std::string str();
};

class Scheduler {
 public:
  // Called when model loading has completed
  virtual void start(
      std::vector<network::controller::WorkerConnection *> workers,
      ClockworkState &state) = 0;

  // The actual controller logic once model loading has completed
  virtual void resultFromWorker(std::shared_ptr<workerapi::Result> result) = 0;
  virtual void clientInfer(
      clientapi::InferenceRequest &request,
      std::function<void(clientapi::InferenceResponse &)> callback) = 0;
};


/* A dummy scheduler implementation that just echos commands received */
class EchoScheduler : public Scheduler {
 public:
  void start(std::vector<network::controller::WorkerConnection *> workers,
             ClockworkState &state) {
    // TODO: print all the info
    std::cout << "EchoScheduler started" << std::endl;
  }

  void resultFromWorker(std::shared_ptr<workerapi::Result> result) {
    std::cout << "Unexpectedly received a result from a worker: "
              << result->str() << std::endl;
  }

  void clientInfer(
      clientapi::InferenceRequest &request,
      std::function<void(clientapi::InferenceResponse &)> callback) {
    // std::cout << "Received: " << request.str() << std::endl;

    clientapi::InferenceResponse response;
    response.header.user_request_id = request.header.user_request_id;
    response.header.status = clockworkSuccess;
    response.output_size = 0;
    response.output = nullptr;

    callback(response);
  }
};

/* The SimpleScheduler does the following:

(1) If a model exists on multiple workers GPUs, assigns requests to workers
round-robin (2) If a model isn't on any worker GPUs, selects a worker,
round-robin, to load the GPU (3) If a GPU is out of memory, evicts the LRU model
(4) Workers execute requests FIFO
(5) Controller does not batch requests to the same model
(6) Controller only forwards 3 requests at a time to a worker

TODO: implement

 */

class SimpleScheduler : public Scheduler {
public:
 	void start(std::vector<network::controller::WorkerConnection*> workers,
		ClockworkState &state) {}

 	void clientInfer(clientapi::InferenceRequest &request,
		std::function<void(clientapi::InferenceResponse&)> callback) {}

 	void resultFromWorker(std::shared_ptr<workerapi::Result> result) {}
};


}

#endif
