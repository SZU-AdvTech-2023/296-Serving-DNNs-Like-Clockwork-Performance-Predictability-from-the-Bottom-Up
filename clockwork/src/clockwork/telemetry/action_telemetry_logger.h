#ifndef _CLOCKWORK_TELEMETRY_ACTION_TELEMETRY_LOGGER_H_
#define _CLOCKWORK_TELEMETRY_ACTION_TELEMETRY_LOGGER_H_

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

class ActionTelemetryLogger {
public:
	virtual void log(std::shared_ptr<ActionTelemetry> telemetry) = 0;
	virtual void shutdown(bool awaitCompletion) = 0;
};

class ActionTelemetryDummyLogger : public ActionTelemetryLogger {

	void log(std::shared_ptr<ActionTelemetry> telemetry) {}
	void shutdown(bool awaitCompletion) {}

};

class ActionTelemetryFileLogger : public ActionTelemetryLogger {
private:
	const std::string output_filename;
	std::atomic_bool alive;
	std::thread thread;
	tbb::concurrent_queue<std::shared_ptr<ActionTelemetry>> action_queue;

public:	
	ActionTelemetryFileLogger(std::string output_filename) : output_filename(output_filename), alive(true) {
		thread = std::thread(&ActionTelemetryFileLogger::main, this);
		threading::initLoggerThread(thread);
	}

	void shutdown(bool awaitCompletion) {
		alive = false;
		if (awaitCompletion) {
			thread.join();
		}
	}
	
	void log(std::shared_ptr<ActionTelemetry> telemetry) {
		action_queue.push(telemetry);
	}

	void convert(std::shared_ptr<ActionTelemetry> telemetry, SerializedActionTelemetry *converted) {
		converted->telemetry_type = telemetry->telemetry_type;
		converted->action_id = telemetry->action_id;
		converted->action_type = telemetry->action_type;
		converted->status = telemetry->status;
		converted->timestamp = util::nanos(telemetry->timestamp);
	}

	void main() {
		std::ofstream outfile;
		outfile.open(output_filename);
	    pods::OutputStream out(outfile);
	    pods::BinarySerializer<decltype(out)> serializer(out);

		while (alive) {
			std::shared_ptr<ActionTelemetry> srcActionTelemetry;
			if (!action_queue.try_pop(srcActionTelemetry)) {
				usleep(10000);
				continue;
			}

			SerializedActionTelemetry actionTelemetry;
			convert(srcActionTelemetry, &actionTelemetry);
		   	CHECK(serializer.save(actionTelemetry) == pods::Error::NoError) << "Unable to serialize action telemetry";

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

}

#endif