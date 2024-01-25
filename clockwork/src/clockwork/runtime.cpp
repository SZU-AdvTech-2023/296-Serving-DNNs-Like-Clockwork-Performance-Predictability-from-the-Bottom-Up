#include "clockwork/api/worker_api.h"
#include "clockwork/runtime.h"
#include "clockwork/action.h"
#include "clockwork/thread.h"

namespace clockwork {

void BaseExecutor::enqueue(Task* task) {
	if (!queue.enqueue(task, task->eligible())) {
		throw TaskError(actionErrorShuttingDown, "Cannot enqueue task to executor that is shutting down");
	}
}

void BaseExecutor::shutdown() {
	queue.shutdown();
	alive.store(false);
}

void BaseExecutor::join() {
	for (unsigned i = 0; i < threads.size(); i++) {
		threads[i].join();
	}
}

CPUExecutor::CPUExecutor(TaskType type) : BaseExecutor(type) {
	threads.push_back(std::thread(&CPUExecutor::executorMain, this, 0));
	for (auto &thread : threads) threading::initLowPriorityThread(thread);
}

void CPUExecutor::executorMain(unsigned executor_id) {
	std::cout << TaskTypeName(type) << "-" << executor_id << " started" << std::endl;

	while (alive.load()) {
		// Currently, CPUExecutor is only used for LoadModelTask
		LoadModelFromDiskTask* next = dynamic_cast<LoadModelFromDiskTask*>(queue.dequeue());
		
		if (next != nullptr) {
			auto telemetry = next->telemetry;
			telemetry->dequeued = util::hrt();
			next->run();
			telemetry->exec_complete = util::hrt();
		}
	}

	std::vector<Task*> tasks = queue.drain();
	for (Task* task : tasks) {
		task->cancel();
	}
}

GPUExecutorExclusive::GPUExecutorExclusive(TaskType type, unsigned gpu_id):
	BaseExecutor(type), gpu_id(gpu_id) {
	threads.push_back(std::thread(&GPUExecutorExclusive::executorMain, this, 0));
	for (auto &thread : threads) threading::initGPUThread(gpu_id, thread);
}

void GPUExecutorExclusive::executorMain(unsigned executor_id) {
	std::cout << "GPU" << gpu_id << "-" << TaskTypeName(type) << "-" << executor_id << " started" << std::endl;

	int priority = 0;
	if (type==TaskType::PCIe_H2D_Inputs || type==TaskType::PCIe_D2H_Output) {
		priority = -1;
	}
	util::initializeCudaStream(gpu_id, priority);

	cudaStream_t stream = util::Stream();



	while (alive.load()) {
		Task* next = queue.dequeue();

		if (next != nullptr) {
			auto telemetry = next->telemetry;

			telemetry->dequeued = util::hrt();
			next->run(stream);
			telemetry->exec_complete = util::hrt();
		}
	}

	std::vector<Task*> tasks = queue.drain();
	for (Task* task : tasks) {
		task->cancel();
	}
}


AsyncTaskChecker::AsyncTaskChecker() : alive(true) {
	threads.push_back(std::thread(&AsyncTaskChecker::executorMain, this, 0));
	for (auto &thread : threads) threading::initHighPriorityThread(thread);
}

void AsyncTaskChecker::enqueue(AsyncTask* task) {
	queue.push(task);
}

void AsyncTaskChecker::shutdown() {
	alive.store(false);
}

void AsyncTaskChecker::join() {
	for (unsigned i = 0; i < threads.size(); i++) {
		threads[i].join();
	}
}

void AsyncTaskChecker::executorMain(unsigned executor_id) {
	std::cout << "AsyncTaskChecker-" << executor_id << " started" << std::endl;

	std::vector<AsyncTask*> pending_tasks;
	while (alive.load() || pending_tasks.size() > 0) {
		// Check completed tasks
		std::vector<AsyncTask*> still_pending;
		for (AsyncTask* task : pending_tasks) {
			if (task->is_complete()) {
				auto telemetry = task->telemetry;
				telemetry->async_complete = util::hrt();
				task->process_completion();
			} else {
				still_pending.push_back(task);
			}
		}
		pending_tasks = still_pending;

		// Drain any newly queued tasks
		AsyncTask* next;
		while (queue.try_pop(next)) {
			pending_tasks.push_back(next);
		}
		usleep(1);
	}
}

void ClockworkRuntime::shutdown(bool await_completion) {
	/* 
	Stop executors.  They'll finish current tasks, prevent enqueueing
	new tasks, and cancel tasks that haven't been started yet
	*/
	load_model_executor->shutdown();
	for (unsigned gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
		gpu_executors[gpu_id]->shutdown();
		weights_executors[gpu_id]->shutdown();
		inputs_executors[gpu_id]->shutdown();
		outputs_executors[gpu_id]->shutdown();
	}

	if (await_completion) {
		join();
	}
}

void ClockworkRuntime::join() {
	/*
	Wait for executors to be finished
	*/
	load_model_executor->join();
	for (unsigned gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
		gpu_executors[gpu_id]->join();
		weights_executors[gpu_id]->join();
		inputs_executors[gpu_id]->join();
		outputs_executors[gpu_id]->join();
	}

	/*
	Only now do we stop the checker.  Async tasks might still be
	outstanding, and we still want to wait for them to complete
	*/
	for (AsyncTaskChecker* checker : all_checkers) {
		checker->shutdown();
	}
	for (AsyncTaskChecker* checker : all_checkers) {
		checker->join();
	}
}

}
