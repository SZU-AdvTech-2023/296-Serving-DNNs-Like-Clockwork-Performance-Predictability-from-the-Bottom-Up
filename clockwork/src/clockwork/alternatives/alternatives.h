#ifndef _CLOCKWORK_ALTERNATIVES_ALTERNATIVES_H_
#define _CLOCKWORK_ALTERNATIVES_ALTERNATIVES_H_

#include <functional>
#include <array>
#include "clockwork/common.h"
#include "clockwork/telemetry.h"

/*
Alternatives are non-clockwork implementations that exist for the sake of comparison
and evaluation.  The main two alternatives are a thread-per-request runtime (ThreadPoolRuntime) and a 
thread-per-task runtime (GreedyRuntime)
*/

namespace clockwork {
namespace alternatives {

class RequestBuilder {
public:
	virtual RequestBuilder* setTelemetry(RequestTelemetry* telemetry) = 0;;
	virtual RequestBuilder* addTask(TaskType type, std::function<void(void)> operation) = 0;
	virtual RequestBuilder* setCompletionCallback(std::function<void(void)> onComplete) = 0;
	virtual void submit() = 0;
};

class Runtime {
public:
	virtual RequestBuilder* newRequest() = 0;
	virtual void shutdown(bool awaitShutdown) = 0;
	virtual void join() = 0;
};

/**
The threadpool runtime has a fixed-size threadpool for executing requests.
Each thread executes the entirety of a request at a time, e.g. all the tasks
of the request.
**/
Runtime* newFIFOThreadpoolRuntime(const unsigned numThreads);

/**
The Greedy runtime has an executor for each resource type.

An executor consists of a self-contained threadpool and queue.

numThreadsPerExecutor specifies the size of the threadpool

Threadpools do not block on asynchronous cuda work.  Use maxOutstandingPerExecutor to specify
a maximum number of incomplete asynchronous tasks before an executor will block.
**/
Runtime* newGreedyRuntime(const unsigned numThreadsPerExecutor, const unsigned maxOutstandingPerExecutor);

}
}

#endif
