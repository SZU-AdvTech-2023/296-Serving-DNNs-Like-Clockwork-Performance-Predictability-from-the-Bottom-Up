#ifndef _CLOCKWORK_ALTERNATIVES_MODEL_MANAGER_H_
#define _CLOCKWORK_ALTERNATIVES_MODEL_MANAGER_H_

#include <mutex>
#include <deque>
#include <unordered_map>
#include <future>
#include <sstream>

#include "clockwork/alternatives/alternatives.h"
#include "clockwork/alternatives/runtime_model.h"
#include "clockwork/cache.h"
#include "clockwork/model/model.h"
#include "clockwork/telemetry.h"
#include "clockwork/telemetry_logger.h"
#include "clockwork/api/client_api.h"

namespace clockwork {
namespace alternatives {

class Request {
public:
	unsigned id;
	int user_request_id;
	int model_id;
	int batch_size;
	char* input;
	size_t input_size;
	char* output;
	size_t output_size;
	RequestTelemetry* telemetry;
	std::function<void()> callback;
	std::function<void(std::string)> errback;
};

/** Manages a specific model instance */
class ModelManager {
public:
	
	std::atomic_int request_id_seed;
	const int id;
	Runtime* runtime;
	TelemetryLogger* logger;

	// The model being managed
	RuntimeModel model;

	std::mutex queue_mutex;
	std::deque<Request*> pending_requests;

	char* output;

	ModelManager(const int id, Runtime* runtime, PageCache* cache, model::Model* cold, TelemetryLogger* logger);
	~ModelManager();

	void submit(Request* request);

	bool evict();

private:

	void handle_response(Request* request);

	void execute(Request* request);

};

}
}

#endif
