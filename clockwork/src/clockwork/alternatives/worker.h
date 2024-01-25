#ifndef _CLOCKWORK_ALTERNATIVES_WORKER_H_
#define _CLOCKWORK_ALTERNATIVES_WORKER_H_

#include <mutex>
#include <deque>
#include <unordered_map>
#include <future>
#include <sstream>

#include "clockwork/common.h"
#include "clockwork/alternatives/runtime_model.h"
#include "clockwork/cache.h"
#include "clockwork/model/model.h"
#include "clockwork/telemetry.h"
#include "clockwork/telemetry_logger.h"
#include "clockwork/api/client_api.h"
#include "clockwork/alternatives/model_manager.h"

namespace clockwork {
namespace alternatives {

/** This alternatives::Worker is NOT part of Clockwork; it is a baseline for comparison.
The alternatives::Worker implements the Client API but not the worker API of clockwork.
It uses simple threadpools, fair queueing, etc. to execute models */
class Worker : public clockwork::clientapi::ClientAPI {
private:
	Runtime* runtime;
	PageCache* cache;
	TelemetryLogger* logger;

	std::mutex managers_mutex;
	std::vector<ModelManager*> managers;

public:

	Worker(Runtime* runtime, PageCache* cache, TelemetryLogger* logger);
	~Worker();

	void shutdown();

	void uploadModel(clientapi::UploadModelRequest &request, std::function<void(clientapi::UploadModelResponse&)> callback);

	void infer(clientapi::InferenceRequest &request, std::function<void(clientapi::InferenceResponse&)> callback);

	/** This is a 'backdoor' API function for ease of experimentation */
	void evict(clientapi::EvictRequest &request, std::function<void(clientapi::EvictResponse&)> callback);

	/** This is a 'backdoor' API function for ease of experimentation */
	void loadRemoteModel(clientapi::LoadModelFromRemoteDiskRequest &request, std::function<void(clientapi::LoadModelFromRemoteDiskResponse&)> callback);

};



}
}

#endif
