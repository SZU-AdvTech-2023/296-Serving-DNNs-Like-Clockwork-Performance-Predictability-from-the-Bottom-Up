#ifndef _CLOCKWORK_TELEMETRY_H_
#define _CLOCKWORK_TELEMETRY_H_

#include <pods/pods.h>
#include <pods/binary.h>
#include <pods/buffers.h>
#include <chrono>
#include "clockwork/util.h"

namespace clockwork {

struct TaskTelemetry {
	int action_type, task_type, executor_id, gpu_id, model_id, action_id, batch_size, status;
	clockwork::time_point enqueued, 
		 dequeued, exec_complete, async_complete;
	uint64_t eligible_for_dequeue;
	float async_wait, async_duration;
	TaskTelemetry() : enqueued(util::hrt()){}
};

struct ActionTelemetry {
	int telemetry_type;
	int action_id, action_type;
	int status;
	clockwork::time_point timestamp;
};


struct ExecutorTelemetry {
	int task_type, executor_id;
	clockwork::time_point next_task_begin, slot_available, task_dequeued, task_complete;
	float async_wait, async_duration;
};

struct RequestTelemetry {
	int model_id, request_id;
	clockwork::time_point arrived, submitted, complete;
	std::vector<TaskTelemetry*> tasks;
};

struct SerializedTaskTelemetry {
	int action_type, task_type, executor_id, gpu_id, model_id, action_id, batch_size, status;
	uint64_t created, enqueued, eligible_for_dequeue, dequeued, exec_complete, async_complete;
	uint64_t async_wait, async_duration;

	PODS_SERIALIZABLE(1,
		PODS_MDR(action_type),
		PODS_MDR(task_type),
		PODS_MDR(executor_id),
		PODS_MDR(gpu_id),
		PODS_MDR(model_id),
		PODS_MDR(action_id),
		PODS_MDR(batch_size),
		PODS_MDR(status),
		PODS_MDR(enqueued),
		PODS_MDR(eligible_for_dequeue),
		PODS_MDR(dequeued),
		PODS_MDR(exec_complete),
		PODS_MDR(async_complete),
		PODS_MDR(async_wait),
		PODS_MDR(async_duration)
    )
};

struct SerializedActionTelemetry {
	int telemetry_type;
	int action_id, action_type, status;
	uint64_t timestamp;

	PODS_SERIALIZABLE(1,
		PODS_MDR(telemetry_type),
		PODS_MDR(action_id),
		PODS_MDR(action_type),
		PODS_MDR(status),
		PODS_MDR(timestamp)
    )
};

struct SerializedExecutorTelemetry {
	int task_type, executor_id;
	uint64_t next_task_begin, slot_available, task_dequeued, task_complete;
	uint64_t async_wait, async_duration;

	PODS_SERIALIZABLE(1,
		PODS_MDR(task_type),
		PODS_MDR(executor_id),
		PODS_MDR(next_task_begin),
		PODS_MDR(slot_available),
		PODS_MDR(task_dequeued),
		PODS_MDR(task_complete),
		PODS_MDR(async_wait)
    )
};

struct SerializedRequestTelemetry {
	int model_id, request_id;
	uint64_t arrived, submitted, complete;
	std::vector<SerializedTaskTelemetry> tasks;

	PODS_SERIALIZABLE(1,
		PODS_MDR(model_id),
		PODS_MDR(request_id),
		PODS_MDR(arrived),
		PODS_MDR(submitted),
		PODS_MDR(complete),
		PODS_MDR(tasks)
    )
};

}

#endif
