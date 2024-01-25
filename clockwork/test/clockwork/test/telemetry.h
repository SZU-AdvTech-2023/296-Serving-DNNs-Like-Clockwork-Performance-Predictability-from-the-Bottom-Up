#include <unistd.h>
#include <libgen.h>
#include <fstream>
#include <algorithm>

#include <cuda_runtime.h>
#include "clockwork/api/worker_api.h"
#include "clockwork/test/util.h"
#include "clockwork/model/model.h"
#include "clockwork/task.h"
#include "clockwork/worker.h"
#include "clockwork/action.h"
#include <catch2/catch.hpp>
#include "clockwork/test/actions.h"
#include "clockwork/test/controller.h"
#include "tbb/concurrent_queue.h"


using namespace clockwork;
using namespace clockwork::model;

int tasks_logged = 0;
int actions_logged = 0;
int action_id = -1;
int model_id = -1;
int batch_size = -1;
uint64_t action_timestamp;
tbb::concurrent_queue<std::shared_ptr<TaskTelemetry>> task_queue;

class TestActionTelemetryLogger : public ActionTelemetryLogger {
public:
	void log(std::shared_ptr<ActionTelemetry> telemetry) {
		actions_logged ++;
		action_id = telemetry->action_id;
		action_timestamp = util::nanos(telemetry->timestamp);
	}

	void shutdown(bool awaitCompletion){}

};

class TestTaskTelemetryLogger : public TaskTelemetryLogger {
public:
	void log(std::shared_ptr<TaskTelemetry> telemetry) {
		tasks_logged ++;
		action_id = telemetry->action_id;
		model_id = telemetry->model_id;
		batch_size = telemetry->batch_size;
		task_queue.push(telemetry);
	}

	void log(RequestTelemetry* telemetry) {}
	void shutdown(bool awaitCompletion){}

};

