#ifndef _CLOCKWORK_TELEMETRY_CONTROLLER_ACTION_LOGGER_H_
#define _CLOCKWORK_TELEMETRY_CONTROLLER_ACTION_LOGGER_H_

#include <thread>
#include <atomic>
#include "tbb/concurrent_queue.h"
#include <algorithm>
#include <unistd.h>
#include <sstream>
#include <iostream>
#include <numeric>
#include <fstream>
#include "clockwork/util.h"
#include "clockwork/telemetry.h"
#include <dmlc/logging.h>
#include <pods/pods.h>
#include <pods/binary.h>
#include <pods/buffers.h>
#include <pods/streams.h>
#include <tbb/concurrent_queue.h>
#include <iomanip>
#include "clockwork/api/api_common.h"
#include <tuple>
#include "clockwork/api/worker_api.h"
#include "clockwork/thread.h"
#include <fstream>


namespace clockwork {

class AsyncControllerActionTelemetryLogger;
struct ControllerActionTelemetry {
	// Values set explicitly by scheduler
	float goodput = 1.0;

	// Values set automatically by `set` methods on actions
	int action_id;
	int worker_id;
	int gpu_id;
	int action_type;
	int batch_size;
	int model_id;
	uint64_t earliest;
	uint64_t latest;
	uint64_t expected_duration;
	unsigned expected_gpu_clock;
	uint64_t expected_exec_complete;
	uint64_t action_sent;

	// Values set automatically by `set` methods on results
	uint64_t result_received;
	uint64_t result_processing;
	int status;
	unsigned gpu_clock_before;
	unsigned gpu_clock;
	uint64_t worker_action_received;
	uint64_t worker_duration;
	uint64_t worker_exec_complete;
	uint64_t worker_copy_output_complete;
	uint64_t worker_result_sent;

	// Set manually
	unsigned requests_queued = 0;
	unsigned copies_loaded = 0;

	void set(std::shared_ptr<workerapi::Infer> &infer);
	void set(std::shared_ptr<workerapi::LoadWeights> &load);
	void set(std::shared_ptr<workerapi::EvictWeights> &evict);
	void set(std::shared_ptr<workerapi::ErrorResult> &result);
	void set(std::shared_ptr<workerapi::InferResult> &result);
	void set(std::shared_ptr<workerapi::LoadWeightsResult> &result);
	void set(std::shared_ptr<workerapi::EvictWeightsResult> &result);

	static AsyncControllerActionTelemetryLogger* summarize(uint64_t print_interval);
	static AsyncControllerActionTelemetryLogger* log_and_summarize(std::string filename, uint64_t print_interval);
};

class ControllerActionTelemetryLogger {
public:
	virtual void log(ControllerActionTelemetry &telemetry) = 0;
	virtual void shutdown(bool awaitCompletion) = 0;
};

class NoOpControllerActionTelemetryLogger : public ControllerActionTelemetryLogger {
public:
	virtual void log(ControllerActionTelemetry &telemetry) {}
	virtual void shutdown(bool awaitCompletion) {}
};

class ControllerActionTelemetryFileLogger : public ControllerActionTelemetryLogger {
private:
	uint64_t begin = util::now();
	std::ofstream f;

public:
	ControllerActionTelemetryFileLogger(std::string filename);

	void write_headers();
	void log(ControllerActionTelemetry &t);
	void shutdown(bool awaitCompletion);
};

class AsyncControllerActionTelemetryLogger : public ControllerActionTelemetryLogger {
private:
	std::atomic_bool alive = true;
	std::thread thread;
	tbb::concurrent_queue<ControllerActionTelemetry> queue;
	std::vector<ControllerActionTelemetryLogger*> loggers;

public:
	AsyncControllerActionTelemetryLogger();

	void addLogger(ControllerActionTelemetryLogger* logger);
	void start();
	void log(ControllerActionTelemetry &telemetry);
	void shutdown(bool awaitCompletion);

private:
	void run();

};

class ActionPrinter : public ControllerActionTelemetryLogger {
private:
	uint64_t last_print;
	const uint64_t print_interval;
	std::queue<ControllerActionTelemetry> buffered;

public:
	ActionPrinter(uint64_t print_interval);

	void log(ControllerActionTelemetry &telemetry);
	void shutdown(bool awaitCompletion);

	virtual void print(uint64_t interval, std::queue<ControllerActionTelemetry> &buffered) = 0;
};

class SimpleActionPrinter : public ActionPrinter {
public:

	SimpleActionPrinter(uint64_t print_interval);

	typedef std::tuple<int,int,int> Group;

	void print(uint64_t interval, const Group &group, std::queue<ControllerActionTelemetry> &buffered);
	void print(uint64_t interval, std::queue<ControllerActionTelemetry> &buffered);
};


}

#endif