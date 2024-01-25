#ifndef _CLOCKWORK_GREEDYRUNTIME_H_
#define _CLOCKWORK_GREEDYRUNTIME_H_

#include <cuda_runtime.h>
#include <functional>
#include <thread>
#include <atomic>
#include "clockwork/alternatives/alternatives.h"
#include "tbb/concurrent_queue.h"
#include "clockwork/telemetry.h"

namespace clockwork {
namespace alternatives {
	
namespace greedyruntime {


class Task {
public:
	TaskType type;
	std::function<void(void)> f;
	std::atomic_bool syncComplete;
	cudaEvent_t asyncSubmit, asyncStart, asyncComplete;
	Task* prev = nullptr;
	Task* next = nullptr;
	TaskTelemetry* telemetry;
	std::function<void(void)> onComplete;
	std::atomic_int* outstandingCounter;

	Task(TaskType type, std::function<void(void)> f);
	~Task();

	void awaitCompletion();
	bool isAsyncComplete();
	void run(cudaStream_t execStream, cudaStream_t telemetryStream);
	void processAsyncCompleteTelemetry();
	void complete();
};

class GreedyRuntime;

class Executor {
private:
	GreedyRuntime* runtime;
	tbb::concurrent_queue<Task*> queue;
	std::vector<std::thread> threads;


public:
	const TaskType type;
	const unsigned maxOutstanding;

	Executor(GreedyRuntime* runtime, TaskType type, const unsigned numThreads, const unsigned maxOutstanding);

	void enqueue(Task* task);
	void join();
	void executorMain(int executorId);
};

class TaskCompletionChecker {
private:
	GreedyRuntime* runtime;
	tbb::concurrent_queue<Task*> queue;
	std::vector<std::thread> threads;

public:
	std::atomic_bool alive;

	TaskCompletionChecker(GreedyRuntime* runtime, const unsigned numThreads);

	void add(Task* task);
	void join();
	void completeTask(Task* task);
	void checkerMain(int checkerId);
};

class GreedyRuntime : public clockwork::alternatives::Runtime {
private:
	std::atomic_bool alive;
	std::atomic_uint coreCount;
	TaskCompletionChecker* checker;
	std::vector<Executor*> executors;
	const unsigned numThreads;
	const unsigned maxOutstanding;

public:
	GreedyRuntime(const unsigned numThreads, const unsigned maxOutstanding);
	~GreedyRuntime();

	unsigned assignCore();

	void enqueue(Task* task);
	void monitorCompletion(Task* task);
	void shutdown(bool awaitShutdown);
	void join();
	bool isAlive() { return alive.load(); }
	virtual clockwork::alternatives::RequestBuilder* newRequest();
};

class RequestBuilder : public clockwork::alternatives::RequestBuilder {
private:
	GreedyRuntime* runtime;
	RequestTelemetry* telemetry = nullptr;
	std::vector<Task*> tasks;
	std::function<void(void)> onComplete = nullptr;
public:
	RequestBuilder(GreedyRuntime *runtime);

	RequestBuilder* setTelemetry(RequestTelemetry* telemetry);
	RequestBuilder* addTask(TaskType type, std::function<void(void)> operation);
	RequestBuilder* setCompletionCallback(std::function<void(void)> onComplete);

	void submit();
};

}
}
}

#endif
