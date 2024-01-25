#include <atomic>
#include "clockwork/network/controller.h"
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
#include "controller.h"

using namespace clockwork;

// Simple controller implementation that forwards all client requests to workers
class DirectControllerImpl : public Controller {
public:
	std::atomic_int action_id_seed;
	std::atomic_int model_id_seed;

	std::mutex actions_mutex;
	std::unordered_map<int, std::function<void(std::shared_ptr<workerapi::Result>)>> action_callbacks;

	std::unordered_map<std::pair<int, unsigned>, uint64_t, util::hash_pair> weights_available_at;

	DirectControllerImpl(int client_port, std::vector<std::pair<std::string, std::string>> worker_host_port_pairs);
	void save_callback(int id, std::function<void(std::shared_ptr<workerapi::Result>)> callback);

	std::function<void(std::shared_ptr<workerapi::Result>)> get_callback(int id);

	virtual void sendResult(std::shared_ptr<workerapi::Result> result);
	virtual void uploadModel(clientapi::UploadModelRequest &request, std::function<void(clientapi::UploadModelResponse&)> callback);
	virtual void infer(clientapi::InferenceRequest &request, std::function<void(clientapi::InferenceResponse&)> callback);
	virtual void evict(clientapi::EvictRequest &request, std::function<void(clientapi::EvictResponse&)> callback);
	virtual void loadRemoteModel(clientapi::LoadModelFromRemoteDiskRequest &request, std::function<void(clientapi::LoadModelFromRemoteDiskResponse&)> callback);
};

