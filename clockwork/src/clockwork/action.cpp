#include "clockwork/action.h"
#include "clockwork/telemetry.h"
#include <bits/types/FILE.h>
#include <cstdio>
#include <sys/mman.h>
#include <sys/syscall.h>
#include<malloc.h>

namespace clockwork {

void extract_timing_sync(workerapi::Timing* timing, std::shared_ptr<TaskTelemetry> &telemetry) {
	timing->begin = util::nanos(telemetry->dequeued);
	timing->end = util::now();
	timing->duration = timing->end - timing->begin;
}

void extract_timing_async(workerapi::Timing* timing, std::shared_ptr<TaskTelemetry> &telemetry) {
	timing->begin = util::nanos(telemetry->dequeued);
	timing->end = util::nanos(telemetry->async_complete);
	timing->duration = (uint64_t) (telemetry->async_duration * 1000000.0);
}

void set_taskTelemetry(
		std::shared_ptr<TaskTelemetry> telemetry, 
		int action_id, int model_id, int gpu_id, int status, int batch_size, uint64_t earliest,
		int action_type, int task_type) {

	telemetry->action_id = action_id;
	telemetry->model_id = model_id;
	telemetry->gpu_id = gpu_id;
	telemetry->status = status;
	telemetry->batch_size = batch_size;
	telemetry->eligible_for_dequeue = earliest;
	telemetry->action_type = action_type;
	telemetry->task_type = task_type;
}

LoadModelFromDiskAction::LoadModelFromDiskTaskImpl::LoadModelFromDiskTaskImpl(LoadModelFromDiskAction* load_model) : 
		LoadModelFromDiskTask(
			load_model->runtime->manager,
			load_model->action->model_id,
			load_model->action->model_path,
			load_model->action->earliest,
			load_model->action->latest,
			load_model->action->no_of_copies,
			load_model->action->max_batch_size,
			load_model->action->max_exec_duration),
		load_model(load_model) {
}

void LoadModelFromDiskAction::LoadModelFromDiskTaskImpl::run(cudaStream_t stream) {
	try {
		LoadModelFromDiskTask::run(stream);
	} catch (TaskError &error) {
		load_model->handle_error(error);
	}
}

void LoadModelFromDiskAction::LoadModelFromDiskTaskImpl::success(RuntimeModel* rm) {

	auto result = std::make_shared<workerapi::LoadModelFromDiskResult>();

	result->id = load_model->action->id;
	result->action_type = workerapi::loadModelFromDiskAction;
	result->status = actionSuccess;
	result->input_size = rm->model->input_size(1);
	result->output_size = rm->model->output_size(1);
	result->supported_batch_sizes = rm->model->implemented_batch_sizes();
	result->copies_created = load_model->action->no_of_copies;
	result->weights_load_time_nanos = rm->model->transfer_measurement;
	for (auto &p : rm->model->models) {
		result->batch_size_exec_times_nanos.push_back(p.second->exec_measurement);
	}

	// TODO Verify: I assume that GPU-specific weights_caches have identical page_size
	int page_size = load_model->runtime->manager->weights_caches[0]->page_size;
	result->num_weights_pages = rm->model->num_weights_pages(page_size);
	result->weights_size_in_cache = result->num_weights_pages * page_size;

	extract_timing_sync(result.get(), telemetry);

	load_model->success(result);
}

void LoadModelFromDiskAction::LoadModelFromDiskTaskImpl::cancel() {
	TaskError error(actionCancelled, "Action cancelled");
	load_model->handle_error(error);
}

LoadModelFromDiskAction::LoadModelFromDiskAction(ClockworkRuntime* runtime, 
	std::shared_ptr<workerapi::LoadModelFromDisk> action) :
	runtime(runtime), action(action), task(nullptr) {
}

LoadModelFromDiskAction::~LoadModelFromDiskAction() {
	if (task != nullptr) delete task;
	action = nullptr;
}

void LoadModelFromDiskAction::submit() {
	CHECK(task == nullptr);
	task = new LoadModelFromDiskTaskImpl(this);
	runtime->load_model_executor->enqueue(task);
}

void LoadModelFromDiskAction::handle_error(TaskError &error) {
	auto result = std::make_shared<workerapi::ErrorResult>();

	result->id = action->id;
	result->action_type = workerapi::loadModelFromDiskAction;
	result->status = error.status_code;
	result->message = error.message;

	this->error(result);
}




LoadWeightsAction::LoadWeightsTaskImpl::LoadWeightsTaskImpl(LoadWeightsAction* load_weights):
	LoadWeightsTask(load_weights->runtime->manager,
					load_weights->action->model_id,
					load_weights->action->earliest,
					load_weights->action->latest,
					load_weights->action->gpu_id,
					load_weights->runtime->event_pools[load_weights->action->gpu_id]),
	load_weights(load_weights) {
}

void LoadWeightsAction::LoadWeightsTaskImpl::run(cudaStream_t stream) {
	try {
		LoadWeightsTask::run(stream);
		load_weights->runtime->weights_checkers[gpu_id]->enqueue(this);
	} catch (TaskError &error) {
		load_weights->handle_error(error);
	}
}

void LoadWeightsAction::LoadWeightsTaskImpl::process_completion() {
	try {
		LoadWeightsTask::process_completion();
	} catch (TaskError &error) {
		load_weights->handle_error(error);
	}
}

void LoadWeightsAction::LoadWeightsTaskImpl::success(RuntimeModel* rm) {
	auto result = std::make_shared<workerapi::LoadWeightsResult>();

	result->id = load_weights->action->id;
	result->action_type = workerapi::loadWeightsAction;
	result->status = actionSuccess;

	set_taskTelemetry(
			telemetry, 
			load_weights->action->id, load_weights->action->model_id,
			load_weights->action->gpu_id, actionSuccess, 0,
			load_weights->action->earliest,
			workerapi::loadWeightsAction, -1);

	extract_timing_async(result.get(), telemetry);
	
	load_weights->runtime->task_telemetry_logger->log(telemetry);

	load_weights->success(result);
}

void LoadWeightsAction::LoadWeightsTaskImpl::cancel() {
	TaskError error(actionCancelled, "Action cancelled");
	load_weights->handle_error(error);
}

LoadWeightsAction::LoadWeightsAction(ClockworkRuntime* runtime, std::shared_ptr<workerapi::LoadWeights> action) :
	runtime(runtime), action(action), task(nullptr) {
}

LoadWeightsAction::~LoadWeightsAction() {
	if (task != nullptr) delete task;
	action = nullptr;
}

void LoadWeightsAction::submit() {
	task = new LoadWeightsTaskImpl(this);
	runtime->weights_executors[action->gpu_id]->enqueue(task);
}

void LoadWeightsAction::handle_error(TaskError &error) {
	auto result = std::make_shared<workerapi::ErrorResult>();

	result->id = action->id;
	result->action_type = workerapi::loadWeightsAction;
	result->status = error.status_code;
	result->message = error.message;

	set_taskTelemetry(
			task->telemetry, 
			action->id, action->model_id, action->gpu_id,
			error.status_code, 0, action->earliest,
			workerapi::loadWeightsAction, -1);

	runtime->task_telemetry_logger->log(task->telemetry);

	this->error(result);
}





EvictWeightsAction::EvictWeightsTaskImpl::EvictWeightsTaskImpl(EvictWeightsAction* evict_weights) : 
		EvictWeightsTask(
			evict_weights->runtime->manager, 
			evict_weights->action->model_id, 
			evict_weights->action->earliest, 
			evict_weights->action->latest,
			evict_weights->action->gpu_id),
		evict_weights(evict_weights) {
}

void EvictWeightsAction::EvictWeightsTaskImpl::run(cudaStream_t stream) {
	try {
		EvictWeightsTask::run(stream);
	} catch (TaskError &error) {
		evict_weights->handle_error(error);
	}
}

void EvictWeightsAction::EvictWeightsTaskImpl::success(RuntimeModel* rm) {
	auto result = std::make_shared<workerapi::EvictWeightsResult>();

	result->id = evict_weights->action->id;
	result->action_type = workerapi::evictWeightsAction;
	result->status = actionSuccess;

	set_taskTelemetry(
			telemetry, 
			evict_weights->action->id, evict_weights->action->model_id,
			evict_weights->action->gpu_id, actionSuccess, 0,
			evict_weights->action->earliest,
			workerapi::evictWeightsAction, -1);

	extract_timing_async(result.get(), telemetry);
	
	evict_weights->runtime->task_telemetry_logger->log(telemetry);

	evict_weights->success(result);
}

void EvictWeightsAction::EvictWeightsTaskImpl::cancel() {
	TaskError error(actionCancelled, "Action cancelled");
	evict_weights->handle_error(error);
}

EvictWeightsAction::EvictWeightsAction(ClockworkRuntime* runtime, std::shared_ptr<workerapi::EvictWeights> action) :
	runtime(runtime), action(action), task(nullptr) {
}

EvictWeightsAction::~EvictWeightsAction() {
	if (task != nullptr) delete task;
	action = nullptr;
}

void EvictWeightsAction::submit() {
	task = new EvictWeightsTaskImpl(this);
	// Rather than have an entire new executor for this, just for now
	// use the outputs executor because it's never even close to full utilization
	runtime->weights_executors[action->gpu_id]->enqueue(task);
}

void EvictWeightsAction::handle_error(TaskError &error) {
	auto result = std::make_shared<workerapi::ErrorResult>();

	result->id = action->id;
	result->action_type = workerapi::evictWeightsAction;
	result->status = error.status_code;
	result->message = error.message;

	set_taskTelemetry(
			task->telemetry, 
			action->id, action->model_id, action->gpu_id,
			error.status_code, 0, action->earliest,
			workerapi::evictWeightsAction, -1);

	runtime->task_telemetry_logger->log(task->telemetry);	this->error(result);
}




const uint64_t copy_input_lead_in = 5000000; // Copy inputs up to 5 ms before exec
uint64_t InferAction::copy_input_earliest() {
	return copy_input_lead_in > action->earliest ? 0 : (action->earliest - copy_input_lead_in);
}

InferAction::CopyInputTaskImpl::CopyInputTaskImpl(InferAction* infer):
	CopyInputTask(infer->runtime->manager,
				  infer->action->model_id,
				  infer->copy_input_earliest(),
				  infer->action->latest,
				  infer->action->batch_size,
				  infer->action->input_size,
				  infer->action->input,
				  infer->action->input_sizes,
				  infer->action->gpu_id,
				  infer->runtime->event_pools[infer->action->gpu_id]),
	infer(infer) {
}

void InferAction::CopyInputTaskImpl::run(cudaStream_t stream) {
	try {
		CopyInputTask::run(stream);
		// infer->runtime->input_checkers[gpu_id]->enqueue(this);

		// Making this synchronous
		CUDA_CALL(cudaStreamSynchronize(stream));
		CopyInputTask::process_completion();
	} catch (TaskError &error) {
		infer->handle_error(error);
	}
}

void InferAction::CopyInputTaskImpl::process_completion() {
	try {
		CopyInputTask::process_completion();
	} catch (TaskError &error) {
		infer->handle_error(error);
	}
}

void InferAction::CopyInputTaskImpl::success(RuntimeModel* rm, char* io_memory) {
	infer->rm = rm;
	infer->io_memory = io_memory;
	infer->exec = new ExecTaskImpl(infer);
	infer->runtime->gpu_executors[gpu_id]->enqueue(infer->exec);
}

void InferAction::CopyInputTaskImpl::cancel() {
	TaskError error(actionCancelled, "Action cancelled");
	infer->handle_error(error);
}

InferAction::ExecTaskImpl::ExecTaskImpl(InferAction* infer):
	ExecTask(infer->rm,
			 infer->runtime->manager,
			 infer->action->earliest,
			 infer->action->latest,
			 infer->action->batch_size,
			 infer->io_memory,
			 infer->action->gpu_id,
			 infer->runtime->event_pools[infer->action->gpu_id]),
	infer(infer) {
}

void InferAction::ExecTaskImpl::run(cudaStream_t stream) {
	try {
		gpu_clock_before = infer->runtime->gpu_clock->get(gpu_id);
		ExecTask::run(stream);
		infer->runtime->gpu_checkers[gpu_id]->enqueue(this);
	} catch (TaskError &error) {
		infer->handle_error(error);
	}
}

void InferAction::ExecTaskImpl::process_completion() {
	try {
		ExecTask::process_completion();
	} catch (TaskError &error) {
		infer->handle_error(error);
	}	
}

void InferAction::ExecTaskImpl::success() {
	infer->copy_output = new CopyOutputTaskImpl(infer);
	infer->runtime->outputs_executors[gpu_id]->enqueue(infer->copy_output);
}

void InferAction::ExecTaskImpl::cancel() {
	TaskError error(actionCancelled, "Action cancelled");
	infer->handle_error(error);
}

InferAction::CopyOutputTaskImpl::CopyOutputTaskImpl(InferAction* infer):
	CopyOutputTask(infer->rm,
				   infer->runtime->manager,
				   0,
				   18446744073709551615UL,
				   infer->action->batch_size,
				   infer->io_memory,
				   infer->action->gpu_id,
				   infer->runtime->event_pools[infer->action->gpu_id]),
	infer(infer) {
}

void InferAction::CopyOutputTaskImpl::run(cudaStream_t stream) {
	try {
		CopyOutputTask::run(stream);
		// infer->runtime->output_checkers[gpu_id]->enqueue(this);

		// Making this synchronous
		CUDA_CALL(cudaStreamSynchronize(stream));
		telemetry->async_complete = util::hrt();
		CopyOutputTask::process_completion();
	} catch (TaskError &error) {
		infer->handle_error(error);
	}
}

void InferAction::CopyOutputTaskImpl::process_completion() {
	// try {
	// 	CopyOutputTask::process_completion();
	// } catch (TaskError &error) {
	// 	infer->handle_error(error);
	// }
}

void InferAction::CopyOutputTaskImpl::success(char* output) {
	infer->handle_completion(output);
}

void InferAction::CopyOutputTaskImpl::cancel() {
	TaskError error(actionCancelled, "Action cancelled");
	infer->handle_error(error);
}

InferAction::InferAction(ClockworkRuntime* runtime, std::shared_ptr<workerapi::Infer> action) :
		runtime(runtime), action(action), rm(nullptr), io_memory(nullptr), zero_size(action->input_size == 0) {}

InferAction::~InferAction() {
	if (copy_input != nullptr) delete copy_input;
	if (exec != nullptr) delete exec;
	if (copy_output != nullptr) delete copy_output;
	
	if (io_memory != nullptr) {
		runtime->manager->io_pools[action->gpu_id]->free(io_memory);
		io_memory = nullptr;
	}
}

void InferAction::submit() {
	copy_input = new CopyInputTaskImpl(this);
	runtime->inputs_executors[action->gpu_id]->enqueue(copy_input);
}

void InferAction::handle_completion(char* output) {
	if (io_memory != nullptr) {
		runtime->manager->io_pools[action->gpu_id]->free(io_memory);
		io_memory = nullptr;
	}

	auto result = std::make_shared<workerapi::InferResult>();

	result->id = action->id;
	result->action_type = workerapi::inferAction;
	result->status = actionSuccess;

	// set_taskTelemetry(
	// 		copy_input->telemetry, 
	// 		action->id, action->model_id, action->gpu_id,
	// 		actionSuccess, action->batch_size, copy_input_earliest(),
	// 		workerapi::inferAction, copyInputTask);

	// set_taskTelemetry(
	// 		exec->telemetry, 
	// 		action->id, action->model_id, action->gpu_id,
	// 		actionSuccess, action->batch_size, action->earliest,
	// 		workerapi::inferAction, execTask);

	// set_taskTelemetry(
	// 		copy_output->telemetry, 
	// 		action->id, action->model_id, action->gpu_id,
	// 		actionSuccess, action->batch_size, action->earliest,
	// 		workerapi::inferAction, copyOutputTask);

	extract_timing_async(&result->copy_input, copy_input->telemetry);
	extract_timing_async(&result->exec, exec->telemetry);
	extract_timing_async(&result->copy_output, copy_output->telemetry);

	// runtime->task_telemetry_logger->log(copy_input->telemetry);
	// runtime->task_telemetry_logger->log(exec->telemetry);
	// runtime->task_telemetry_logger->log(copy_output->telemetry);

	if (zero_size) {
		result->output_size = 0;
	} else {
		result->output_size = rm->model->output_size(action->batch_size);
	}
	result->output = output;

	result->gpu_id = action->gpu_id;
	result->gpu_clock_before = exec->gpu_clock_before;
	result->gpu_clock = runtime->gpu_clock->get(result->gpu_id);
	
	this->success(result);
}

void InferAction::handle_error(TaskError &error) {
	if (io_memory != nullptr) {
		runtime->manager->io_pools[action->gpu_id]->free(io_memory);
		io_memory = nullptr;
	}

	auto result = std::make_shared<workerapi::ErrorResult>();

	result->id = action->id;
	result->action_type = workerapi::inferAction;
	result->status = error.status_code;
	result->message = error.message;

	if (copy_input != nullptr) {
		set_taskTelemetry(
				copy_input->telemetry, 
				action->id, action->model_id, action->gpu_id,
				error.status_code, action->batch_size, copy_input_earliest(),
				workerapi::inferAction, copyInputTask);
		runtime->task_telemetry_logger->log(copy_input->telemetry);
	}

	if (exec != nullptr) {
		set_taskTelemetry(
				exec->telemetry, 
				action->id, action->model_id, action->gpu_id,
				error.status_code, action->batch_size, action->earliest,
				workerapi::inferAction, execTask);
		runtime->task_telemetry_logger->log(exec->telemetry);
	}

	if (copy_output != nullptr) {
		set_taskTelemetry(
				copy_output->telemetry, 
				action->id, action->model_id, action->gpu_id,
				error.status_code, action->batch_size, action->earliest,
				workerapi::inferAction, copyOutputTask);
		runtime->task_telemetry_logger->log(copy_output->telemetry);
	}


	this->error(result);
}

}
