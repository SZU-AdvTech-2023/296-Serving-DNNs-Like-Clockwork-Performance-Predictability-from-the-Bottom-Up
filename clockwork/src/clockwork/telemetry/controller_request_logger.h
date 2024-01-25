#ifndef _CLOCKWORK_TELEMETRY_CONTROLLER_REQUEST_LOGGER_H_
#define _CLOCKWORK_TELEMETRY_CONTROLLER_REQUEST_LOGGER_H_

#include <thread>
#include <atomic>
#include "tbb/concurrent_queue.h"
#include <algorithm>
#include <unistd.h>
#include <sstream>
#include <iostream>
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
#include "clockwork/api/client_api.h"
#include <iostream>
#include "clockwork/thread.h"


namespace clockwork {

class RequestTelemetryLogger;
struct ControllerRequestTelemetry {
	int request_id;
	int user_id;
	int model_id;
	uint64_t arrival;
	uint64_t departure;
	uint64_t deadline;
	float slo_factor;
	int arrival_count;
	int departure_count;
	int result;

	void set(clientapi::InferenceRequest &request);
	void set(clientapi::InferenceResponse &response);

	static RequestTelemetryLogger* summarize(uint64_t print_interval);
	static RequestTelemetryLogger* log_and_summarize(std::string filename, uint64_t print_interval);
};

class RequestTelemetryLogger {
public:
	virtual void log(ControllerRequestTelemetry &telemetry) = 0;
	virtual void shutdown(bool awaitCompletion) = 0;
};

class NoOpRequestTelemetryLogger : public RequestTelemetryLogger {
public:
	virtual void log(ControllerRequestTelemetry &telemetry) {}
	virtual void shutdown(bool awaitCompletion) {}
};

class RequestTelemetryFileLogger : public RequestTelemetryLogger {
private:
	uint64_t begin = util::now();
	std::ofstream f;

public:

	RequestTelemetryFileLogger(std::string filename);

	void write_headers();
	void log(ControllerRequestTelemetry &t);
	void shutdown(bool awaitCompletion);
};

class AsyncRequestTelemetryLogger : public RequestTelemetryLogger {
private:
	std::atomic_bool alive = true;
	std::thread thread;
	tbb::concurrent_queue<ControllerRequestTelemetry> queue;
	std::vector<RequestTelemetryLogger*> loggers;

public:	

	AsyncRequestTelemetryLogger();

	void addLogger(RequestTelemetryLogger* logger);
	void start();
	void run();
	void log(ControllerRequestTelemetry &telemetry);
	void shutdown(bool awaitCompletion);

};

class RequestTelemetryPrinter : public RequestTelemetryLogger {
private:
	uint64_t last_print;
	const uint64_t print_interval;
	std::queue<ControllerRequestTelemetry> buffered;

public:

	RequestTelemetryPrinter(uint64_t print_interval);

	void print(uint64_t interval);
	void log(ControllerRequestTelemetry &telemetry);
	void shutdown(bool awaitCompletion);

};

}

#endif
