#ifndef _CLOCKWORK_THREADPOOLRUNTIME_H_
#define _CLOCKWORK_THREADPOOLRUNTIME_H_

#include <functional>
#include <thread>
#include <atomic>
#include "tbb/concurrent_queue.h"
#include "clockwork/alternatives/alternatives.h"

namespace clockwork {
namespace alternatives {

namespace threadpoolruntime {

struct Task {
	TaskType type;
	std::function<void(void)> f;
	TaskTelemetry* telemetry;
};

struct Request {
	std::vector<Task> tasks;
	std::function<void(void)> onComplete;
};

class Queue {
public:
	virtual void enqueue(Request* request) = 0;
	virtual bool try_dequeue(Request* &request) = 0;
};

class FIFOQueue : public Queue {
private:
	tbb::concurrent_queue<Request*> queue;
public:
	virtual void enqueue(Request* request);
	virtual bool try_dequeue(Request* &request);
};

/** A simple manager based on a threadpool.
Each threadpool thread executes the request in its entirety */
class ThreadpoolRuntime : public clockwork::alternatives::Runtime {
private:
	const int numThreads;
	Queue* queue;
	std::atomic_bool alive;
	std::vector<std::thread> threads;

public:
	ThreadpoolRuntime(const unsigned numThreads, Queue* queue);
	~ThreadpoolRuntime();

	void threadpoolMain(int threadNumber);
	void shutdown(bool awaitShutdown);
	void join();
	virtual clockwork::alternatives::RequestBuilder* newRequest();
	void submit(Request* request);

};

class RequestBuilder : public clockwork::alternatives::RequestBuilder {
private:
	RequestTelemetry* telemetry = nullptr;
	std::function<void(void)> onComplete = nullptr;
	ThreadpoolRuntime* runtime;
	std::vector<Task> tasks;
public:
	RequestBuilder(ThreadpoolRuntime *runtime) : runtime(runtime) {}

	RequestBuilder* setTelemetry(RequestTelemetry* telemetry);
	RequestBuilder* addTask(TaskType type, std::function<void(void)> operation);
	RequestBuilder* setCompletionCallback(std::function<void(void)> onComplete);
	void submit();
};

}
}
}

#endif