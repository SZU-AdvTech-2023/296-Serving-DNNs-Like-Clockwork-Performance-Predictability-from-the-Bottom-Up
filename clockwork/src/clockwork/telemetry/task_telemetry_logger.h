#ifndef _CLOCKWORK_TELEMETRY_TASK_TELEMETRY_LOGGER_H_
#define _CLOCKWORK_TELEMETRY_TASK_TELEMETRY_LOGGER_H_

#include <thread>
#include <atomic>
#include "tbb/concurrent_queue.h"
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
#include "clockwork/thread.h"


namespace clockwork {

class TaskTelemetryLogger {
public:
	virtual void log(std::shared_ptr<TaskTelemetry> telemetry) = 0;
	virtual void log(RequestTelemetry* telemetry) = 0;
	virtual void shutdown(bool awaitCompletion) = 0;
};

class TaskTelemetryDummyLogger : public TaskTelemetryLogger {
	void log(std::shared_ptr<TaskTelemetry> telemetry){}
 	void log(RequestTelemetry* telemetry) {}
	void shutdown(bool awaitCompletion) {}
};

class TaskTelemetryFileLogger : public TaskTelemetryLogger {
private:
	const std::string output_filename;
	std::atomic_bool alive;
	std::thread thread;
	tbb::concurrent_queue<std::shared_ptr<TaskTelemetry>> task_queue;
	tbb::concurrent_queue<RequestTelemetry*> request_queue;

public:	
	TaskTelemetryFileLogger(std::string output_filename) : output_filename(output_filename), alive(true) {
		thread = std::thread(&TaskTelemetryFileLogger::main, this);
		threading::initLoggerThread(thread);
	}

	void shutdown(bool awaitCompletion) {
		alive = false;
		if (awaitCompletion) {
			thread.join();
		}
	}

	void log(std::shared_ptr<TaskTelemetry> telemetry) {
		task_queue.push(telemetry);
	}

	void log(RequestTelemetry* telemetry) {
		request_queue.push(telemetry);
	}


	void convert(std::shared_ptr<TaskTelemetry> telemetry, SerializedTaskTelemetry *converted) {
		converted->action_type = telemetry->action_type;
		converted->task_type = telemetry->task_type;
		converted->executor_id = telemetry->executor_id;
		converted->gpu_id = telemetry->gpu_id;
		converted->model_id = telemetry->model_id;
		converted->batch_size = telemetry->batch_size;
		converted->status = telemetry->status;
		converted->action_id = telemetry->action_id;
		converted->enqueued = util::nanos(telemetry->enqueued);
		converted->eligible_for_dequeue = telemetry->eligible_for_dequeue;
		converted->dequeued = util::nanos(telemetry->dequeued);
		converted->exec_complete = util::nanos(telemetry->exec_complete);
		converted->async_complete = util::nanos(telemetry->async_complete);
		converted->async_wait = telemetry->async_wait * 1000000;
		converted->async_duration = telemetry->async_duration * 1000000;
	}

	void convert(TaskTelemetry* telemetry, SerializedTaskTelemetry *converted) {
		converted->task_type = telemetry->task_type;
		converted->executor_id = telemetry->executor_id;
		converted->model_id = telemetry->model_id;
		converted->batch_size = telemetry->batch_size;
		converted->status = telemetry->status;
		converted->action_id = telemetry->action_id;
		converted->enqueued = util::nanos(telemetry->enqueued);
		converted->eligible_for_dequeue = telemetry->eligible_for_dequeue;
		converted->dequeued = util::nanos(telemetry->dequeued);
		converted->exec_complete = util::nanos(telemetry->exec_complete);
		converted->async_complete = util::nanos(telemetry->async_complete);
		converted->async_wait = telemetry->async_wait * 1000000;
		converted->async_duration = telemetry->async_duration * 1000000;
	}

	void convert(RequestTelemetry* telemetry, SerializedRequestTelemetry &converted) {
		converted.model_id = telemetry->model_id;
		converted.request_id = telemetry->request_id;
		converted.arrived = util::nanos(telemetry->arrived);
		converted.submitted = util::nanos(telemetry->submitted);
		converted.complete = util::nanos(telemetry->complete);
		converted.tasks.resize(telemetry->tasks.size());
		for (unsigned i = 0; i < telemetry->tasks.size(); i++) {
			convert(telemetry->tasks[i], &converted.tasks[i]);
		}
	}


	void main() {
		std::ofstream outfile;
		outfile.open(output_filename);
	    pods::OutputStream out(outfile);
	    pods::BinarySerializer<decltype(out)> serializer(out);

		while (alive) {
			std::shared_ptr<TaskTelemetry> srcTelemetry;
			if (!task_queue.try_pop(srcTelemetry)) {
				usleep(10000);
				continue;
			}

			SerializedTaskTelemetry telemetry;
			convert(srcTelemetry, &telemetry);
			
			//TODO: delete srcTelemetry;
			CHECK(serializer.save(telemetry) == pods::Error::NoError) << "Unable to serialize task telemetry";
		
			// std::stringstream msg;
			// msg << "Logging request " << telemetry.request_id 
			//     << " model=" << telemetry.model_id
			//     << " latency=" << (telemetry.complete - telemetry.submitted)
			//     << " totallatency=" << (telemetry.complete - telemetry.arrived)
			//     << std::endl;
			// for (int i = 0; i < telemetry.tasks.size(); i++) {
			// 	msg << "   Task " << i << std::endl
			// 	    << "     queue=" << (telemetry.tasks[i].dequeued - telemetry.tasks[i].enqueued) << std::endl
			// 	    << "     exec="  << (telemetry.tasks[i].exec_complete - telemetry.tasks[i].dequeued) << std::endl
			// 	    << "     async_wait=" << telemetry.tasks[i].async_wait << std::endl
			// 	    << "     async_duration=" << telemetry.tasks[i].async_duration << std::endl;
			// }
			// std::cout << msg.str();
		}
		outfile.close();
	}

};

class InMemoryTelemetryBuffer : public TaskTelemetryLogger {
public:
	tbb::concurrent_queue<std::shared_ptr<TaskTelemetry>> task_queue;
	tbb::concurrent_queue<RequestTelemetry*> request_queue;

public:	
	InMemoryTelemetryBuffer() {}

	void shutdown(bool awaitCompletion) {}

	void log(std::shared_ptr<TaskTelemetry> telemetry) {
		task_queue.push(telemetry);
	}

	void log(RequestTelemetry* telemetry) {
		request_queue.push(telemetry);
	}

	std::vector<std::shared_ptr<TaskTelemetry>> take_task() {
		std::vector<std::shared_ptr<TaskTelemetry>> telemetry;
		std::shared_ptr<TaskTelemetry> next;
		while (task_queue.try_pop(next)) {
			telemetry.push_back(next);
		}
		return telemetry;
	}

	std::vector<RequestTelemetry*> take_request() {
		std::vector<RequestTelemetry*> telemetry;
		RequestTelemetry* next;
		while (request_queue.try_pop(next)) {
			telemetry.push_back(next);
		}
		return telemetry;
	}


};

}

#endif
