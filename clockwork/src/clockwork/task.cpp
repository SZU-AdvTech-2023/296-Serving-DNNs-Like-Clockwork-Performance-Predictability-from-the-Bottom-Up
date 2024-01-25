#include "clockwork/task.h"

#include "tbb/concurrent_queue.h"
#include "clockwork/cuda_common.h"
#include "clockwork/api/worker_api.h"
#include "clockwork/action.h"
#include "clockwork/model/batched.h"
#include "lz4.h"

namespace clockwork {

CudaAsyncTask::CudaAsyncTask(unsigned gpu_id, CudaEventPool* event_pool) :
	AsyncTask(gpu_id),
	event_pool(event_pool),
	async_begin_submitted(false),
	async_end_submitted(false),
	async_begin_event(event_pool->get_or_create()),
	async_end_event(event_pool->get_or_create()) {
}

CudaAsyncTask::~CudaAsyncTask() {
	event_pool->release(async_begin_event);
	event_pool->release(async_end_event);
}

void CudaAsyncTask::record_async_begin(cudaStream_t stream) {
	CUDA_CALL(cudaSetDevice(gpu_id));
	cudaError_t status = cudaEventRecord(async_begin_event, stream);
	async_begin_submitted.store(true);
}

void CudaAsyncTask::record_async_end(cudaStream_t stream) {
	CUDA_CALL(cudaSetDevice(gpu_id));
	CUDA_CALL(cudaEventRecord(async_end_event, stream));
	async_end_submitted.store(true);
}

bool CudaAsyncTask::is_complete() {
	CUDA_CALL(cudaSetDevice(gpu_id));

	// Same semantics as cuda event: unused event is complete
	if (!async_begin_submitted.load()) return true;
	if (!async_end_submitted.load()) return false;

	cudaError_t status = cudaEventQuery(async_end_event);
	if (status == cudaErrorNotReady) {
		return false;
	}
	CHECK(status == cudaSuccess || 
		  status == cudaErrorNotReady ||
		  status == cudaErrorCudartUnloading
		 ) << "CUDA: " << cudaGetErrorString(status);
	return true;
}

void CudaAsyncTask::await_completion() {
	CUDA_CALL(cudaSetDevice(gpu_id));

	// Same semantics as cuda event: unused event is complete
	while (!async_begin_submitted.load());
	while (!async_end_submitted.load());

	CUDA_CALL(cudaEventSynchronize(async_end_event));
}

float CudaAsyncTask::async_duration() {
	float async_duration;
	CUDA_CALL(cudaSetDevice(gpu_id));
	CUDA_CALL(cudaEventElapsedTime(&async_duration, async_begin_event, async_end_event));
	return async_duration;
}


LoadModelFromDiskTask::LoadModelFromDiskTask(MemoryManager* manager, 
	int model_id, std::string model_path, uint64_t earliest, uint64_t latest, 
	int no_of_copies, unsigned max_batch_size, uint64_t max_exec_duration) :
		manager(manager), model_id(model_id), model_path(model_path), 
		earliest(earliest), latest(latest), no_of_copies(no_of_copies), 
		max_batch_size(max_batch_size), max_exec_duration(max_exec_duration) {
}

LoadModelFromDiskTask::~LoadModelFromDiskTask() {}

// Task
uint64_t LoadModelFromDiskTask::eligible() {
	return earliest;
}

void LoadModelFromDiskTask::run(cudaStream_t stream) {
	uint64_t now = util::now(); // TODO: use chrono
	if (now < earliest) {
		std::stringstream err;
		err << "LoadModelFromDiskTask ran before it was eligible"
			<< " (now " << util::millis(now)
			<< ", earliest " << util::millis(earliest) << ")";
		throw TaskError(actionErrorRuntimeError, err.str());
	}

	if (now > latest) {
		std::stringstream err;
		err << "LoadModelFromDiskTask could not start in time"
			<< " (now " << util::millis(now)
			<< ", latest " << util::millis(latest) << ")";
		throw TaskError(actionErrorCouldNotStartInTime, err.str());
	}

	std::vector<unsigned> gpu_ids;
	for (unsigned gpu_id = 0; gpu_id < manager->num_gpus; gpu_id++) {
		gpu_ids.push_back(gpu_id);

		for (unsigned i = 0; i < no_of_copies; i++) {
			if (manager->models->contains(model_id+i, gpu_id)) {
				throw TaskError(actionErrorInvalidModelID, "LoadModelFromDiskTask specified ID that already exists");
			}
		}
	}

	try {
		auto duplicates = model::BatchedModel::loadMultipleFromDiskMultiGPU(
			model_path, gpu_ids, no_of_copies, max_batch_size, max_exec_duration);

		for (auto &gpu_id : gpu_ids) {
			auto &models = duplicates[gpu_id];

			for (unsigned i = 0; i < models.size(); i++) {
				models[i]->instantiate_models_on_host();
				models[i]->instantiate_models_on_device();
				bool success = manager->models->put_if_absent(
					this->model_id + i, 
					gpu_id, 
					new RuntimeModel(models[i], gpu_id)
				);
				CHECK(success) << "Loaded models changed while loading from disk";
			}
		}
	} catch (dmlc::Error &error) {
		throw TaskError(actionErrorInvalidModelPath, error.what());
	}

	this->success(manager->models->get(model_id, 0));
}


LoadWeightsTask::LoadWeightsTask(MemoryManager* manager, int model_id,
	uint64_t earliest, uint64_t latest, unsigned gpu_id,
	CudaEventPool* event_pool):
		CudaAsyncTask(gpu_id, event_pool), manager(manager), model_id(model_id),
		earliest(earliest), latest(latest), rm(nullptr), new_weights(nullptr) {
}

LoadWeightsTask::~LoadWeightsTask() {
	new_weights = nullptr;
}

uint64_t LoadWeightsTask::eligible() {
	return earliest;
}

void LoadWeightsTask::run(cudaStream_t stream) {
	uint64_t now = util::now(); // TODO: use chrono
	if (now < earliest) {
		std::stringstream err;
		err << "LoadWeightsTask ran before it was eligible"
			<< " (now " << util::millis(now)
			<< ", earliest " << util::millis(earliest) << ")";
		throw TaskError(loadWeightsTooEarly, err.str());
	}

	if (now > latest) {
		std::stringstream err;
		err << "LoadWeightsTask could not start in time"
			<< " (now " << util::millis(now)
			<< ", latest " << util::millis(latest) << ")";
		throw TaskError(loadWeightsTooLate, err.str());
	}

	rm = manager->models->get(model_id, gpu_id);
	if (rm == nullptr) {
		std::string error_message = "LoadWeightsTask could not find model";
		error_message += " with model ID " + std::to_string(model_id);
		error_message += " and GPU ID " + std::to_string(gpu_id);
		throw TaskError(loadWeightsUnknownModel, error_message);
	}

	rm->lock();

	this->new_version = ++rm->version;
	std::shared_ptr<Allocation> previous_weights = rm->weights;
	rm->weights = nullptr;

	rm->unlock();

	if (previous_weights != nullptr && !previous_weights->evicted) {
		manager->weights_caches[gpu_id]->unlock(previous_weights);
		manager->weights_caches[gpu_id]->free(previous_weights);
	}

	unsigned num_pages = rm->model->num_weights_pages(manager->weights_caches[gpu_id]->page_size);
	this->new_weights = manager->weights_caches[gpu_id]->alloc(num_pages, []{});
	if (this->new_weights == nullptr) {
		throw TaskError(loadWeightsInsufficientCache, "LoadWeightsTask failed to allocate pages from cache");
	}

	this->record_async_begin(stream);
	rm->model->transfer_weights_to_device(new_weights->page_pointers, stream);
	this->record_async_end(stream);

}

void LoadWeightsTask::process_completion() {
	telemetry->async_duration = this->async_duration();

	bool version_unchanged = false;

	rm->lock();

	if (rm->version == this->new_version) {
		rm->version = this->new_version;
		rm->weights = this->new_weights;
		version_unchanged = true;
	}

	rm->unlock();

	if (version_unchanged) {
		success(rm);
	} else {
		throw TaskError(loadWeightsConcurrentModification, "Model weights were modified while being copied");
	}
}

EvictWeightsTask::EvictWeightsTask(MemoryManager* manager, int model_id,
	uint64_t earliest, uint64_t latest, unsigned gpu_id):
		Task(gpu_id),
		manager(manager),
		model_id(model_id),
		earliest(earliest),
		latest(latest) {
}

uint64_t EvictWeightsTask::eligible() {
	return earliest;
}

void EvictWeightsTask::run(cudaStream_t stream) {
	uint64_t now = util::now(); // TODO: use chrono, possibly use the task telemetry
	if (now < earliest) {
		throw TaskError(evictWeightsTooEarly, "EvictWeightsTask ran before it was eligible");
	}

	if (now > latest) {
		throw TaskError(evictWeightsTooLate, "EvictWeightsTask could not start in time");
	}

	rm = manager->models->get(model_id, gpu_id);
	if (rm == nullptr) {
		throw TaskError(evictWeightsUnknownModel, "EvictWeightsTask could not find model with specified id");
	}

	rm->lock();

	rm->version++;
	std::shared_ptr<Allocation> previous_weights = rm->weights;
	rm->weights = nullptr;

	rm->unlock();

	if (previous_weights == nullptr || previous_weights->evicted) {
		throw TaskError(evictWeightsNotInCache, "EvictWeightsTask not processed because no weights exist");
	}

	manager->weights_caches[gpu_id]->unlock(previous_weights);
	manager->weights_caches[gpu_id]->free(previous_weights);

	success(rm);
}


CopyInputTask::CopyInputTask(MemoryManager* manager, int model_id,
	uint64_t earliest, uint64_t latest, unsigned batch_size, size_t input_size,
	char* &input, unsigned gpu_id, CudaEventPool* event_pool):
		CudaAsyncTask(gpu_id, event_pool), manager(manager), model_id(model_id),
		earliest(earliest), latest(latest), batch_size(batch_size),
		input_size(input_size), input(input), rm(nullptr), io_memory(nullptr) {
}


CopyInputTask::CopyInputTask(MemoryManager* manager, int model_id,
	uint64_t earliest, uint64_t latest, unsigned batch_size, size_t input_size,
	char* &input, std::vector<size_t> &compressed_input_sizes, unsigned gpu_id, CudaEventPool* event_pool):
		CudaAsyncTask(gpu_id, event_pool), manager(manager), model_id(model_id),
		earliest(earliest), latest(latest), batch_size(batch_size),
		input_size(input_size), input(input), compressed_input_sizes(compressed_input_sizes),
		rm(nullptr), io_memory(nullptr) {
}

CopyInputTask::~CopyInputTask() {
	io_memory = nullptr;
}

uint64_t CopyInputTask::eligible() {
	return earliest;
}

void CopyInputTask::run(cudaStream_t stream) {
	uint64_t now = util::now(); // TODO: use chrono
	if (now < earliest) {
		std::stringstream err;
		err << "CopyInputTask ran before it was eligible"
			<< " (now " << util::millis(now)
			<< ", earliest " << util::millis(earliest) << ")";
		throw TaskError(copyInputTooEarly, err.str());
	}

	if (now > latest) {
		std::stringstream err;
		err << "CopyInputTask could not start in time"
			<< " (now " << util::millis(now)
			<< ", latest " << util::millis(latest) << ")";
		throw TaskError(copyInputTooLate, err.str());
	}

	rm = manager->models->get(model_id, gpu_id);
	if (rm == nullptr) {
		throw TaskError(copyInputUnknownModel, "CopyInputTask could not find model with specified id");
	}

	if (!rm->model->is_valid_batch_size(batch_size)) {
		std::stringstream err;
		err << "CopyInputTask received unsupported batch size " << batch_size;
		throw TaskError(copyInputInvalidBatchSize, err.str());
	}

	if (input_size == 0) {
		// Used in testing; allow client to send zero-size inputs and generate worker-side
		CHECK(manager->allow_zero_size_inputs) << "Received zero-size input but disallowed by config";

		input_size = rm->model->input_size(batch_size);
		input = manager->host_io_pool->alloc(input_size);
		if (input == nullptr) {
			throw TaskError(copyInputHostAlloc, "Unable to alloc from host_io_pool for infer action input");
		}

		size_t single_input_size = rm->model->input_size(1);
		size_t offset = 0;
		for (int i = 0; i < batch_size; i++) {
			manager->input_generator->generateInput(single_input_size, input+offset);
			offset += single_input_size;
		}

	} else if (compressed_input_sizes.size() > 0) {
		if (compressed_input_sizes.size() > batch_size) {
			throw TaskError(copyInputBadSizes, "More compressed inputs received than batch size");
		}

		size_t single_input_size = rm->model->input_size(1);
		char* decompressed = manager->host_io_pool->alloc(rm->model->input_size(batch_size));
		CHECK(decompressed!=nullptr) << "decompressed was nullptr";

		size_t input_offset = 0;
		size_t output_offset = 0;
		for (auto &next_input_size : compressed_input_sizes) {
			if (input_offset+next_input_size > input_size) {
				throw TaskError(copyInputBadSizes, "Compressed inputs exceeded specified size");
			}

			int decompressed_size = LZ4_decompress_safe(input+input_offset, decompressed+output_offset, next_input_size, single_input_size);
			if (decompressed_size != single_input_size) {
				throw TaskError(copyInputBadDecompress, "Input decompressed to wrong size");
			}

			input_offset += next_input_size;
			output_offset += single_input_size;
		}

		input = decompressed;
		input_size = rm->model->input_size(batch_size);

	} else if (rm->model->input_size(batch_size) != input_size) {
		// Normal behavior requires correctly sized inputs
		std::stringstream err;
		err << "CopyInputTask received incorrectly sized input"
		    << " (expected " << rm->model->input_size(batch_size) 
		    << ", got " << input_size
		    << " (batch_size=" << batch_size << ")";
		throw TaskError(copyInputInvalidInput, err.str());	

	}

	size_t io_memory_size = rm->model->io_memory_size(batch_size);
	this->io_memory = manager->io_pools[gpu_id]->alloc(io_memory_size);

	if (this->io_memory == nullptr) {
		std::stringstream err;
		err << "CopyInputTask failed to allocate memory from io_pool."
		    << " Model = " << rm->model->source
			<< " Batch size " << batch_size << " requires " << io_memory_size
			<< ", but " << manager->io_pools[gpu_id]->remaining()
			<< " of " << manager->io_pools[gpu_id]->size << " remaining"
			<< " (" << manager->io_pools[gpu_id]->numAllocs() << " outstanding allocations)";
		throw TaskError(copyInputIOPoolExhausted, err.str());
	}

	this->record_async_begin(stream);
	rm->model->transfer_input_to_device(batch_size, input, io_memory, stream);
	this->record_async_end(stream);
}

void CopyInputTask::process_completion() {
	telemetry->async_duration = this->async_duration();
	this->success(rm, io_memory);
}



ExecTask::ExecTask(RuntimeModel* rm, MemoryManager* manager, uint64_t earliest,
	uint64_t latest, unsigned batch_size, char* io_memory, unsigned gpu_id,
	CudaEventPool* event_pool):
		CudaAsyncTask(gpu_id, event_pool), rm(rm), manager(manager),
		earliest(earliest), latest(latest), batch_size(batch_size),
		io_memory(io_memory), weights(nullptr), workspace_memory(nullptr) {
}

ExecTask::~ExecTask() {
	weights = nullptr;

	if (workspace_memory != nullptr) {
		manager->workspace_pools[gpu_id]->free(workspace_memory);
		workspace_memory = nullptr;
	}
}

uint64_t ExecTask::eligible() {
	return earliest;
}

void ExecTask::run(cudaStream_t stream) {
	uint64_t now = util::now(); // TODO: use chrono
	if (now < earliest) {
		std::stringstream err;
		err << "ExecTask could not run before it was eligible"
			<< " (now " << util::millis(now)
			<< ", earliest " << util::millis(earliest) << ")";
		throw TaskError(execTooEarly, err.str());
	}

	if (now > latest) {
		std::stringstream err;
		err << "ExecTask could not start in time"
			<< " (now " << util::millis(now)
			<< ", latest " << util::millis(latest) << ")";
		throw TaskError(execTooLate, err.str());
	}

	rm->lock();

	this->weights_version = rm->version;
	this->weights = rm->weights;

	rm->unlock();

	if (weights == nullptr || weights->evicted) {
		throw TaskError(execWeightsMissing, "ExecTask failed due to missing model weights");
	}

	size_t workspace_size = rm->model->workspace_memory_size(batch_size);
	this->workspace_memory = manager->workspace_pools[gpu_id]->alloc(workspace_size);

	if (this->workspace_memory == nullptr) {
		std::stringstream err;
		err << "ExecTask failed to allocate memory from workspace_pool."
		    << " Model = " << rm->model->source
			<< " Batch size " << batch_size << " requires " << workspace_size
			<< ", but " << manager->workspace_pools[gpu_id]->remaining()
			<< " of " << manager->workspace_pools[gpu_id]->size << " remaining"
			<< " (" << manager->workspace_pools[gpu_id]->before() << " before"
			<< " " << manager->workspace_pools[gpu_id]->after() << " after)"
			<< " (" << manager->workspace_pools[gpu_id]->numAllocs() << " outstanding allocations)";
		throw TaskError(execWorkspacePoolExhausted, err.str());
	}

	this->record_async_begin(stream);
	rm->model->call(batch_size, weights->page_pointers, io_memory, workspace_memory, stream);
	this->record_async_end(stream);

	// With one-at-a-time execution, workspace memory allocations are unnecessary
	if (workspace_memory != nullptr) {
		manager->workspace_pools[gpu_id]->free(workspace_memory);
		workspace_memory = nullptr;
	}
}

void ExecTask::process_completion() {
	telemetry->async_duration = this->async_duration();

	rm->lock();

	int current_weights_version = rm->version;

	rm->unlock();

	if (this->weights_version != current_weights_version || weights->evicted) {
		throw TaskError(execConcurrentWeightsModification, "ExecTask failed due to weights version mismatch");
	}

	this->success();
}



CopyOutputTask::CopyOutputTask(RuntimeModel* rm, MemoryManager* manager,
	uint64_t earliest, uint64_t latest, unsigned batch_size, char* io_memory,
	unsigned gpu_id, CudaEventPool* event_pool):
		CudaAsyncTask(gpu_id, event_pool), rm(rm), manager(manager),
		earliest(earliest), latest(latest), batch_size(batch_size),
		io_memory(io_memory), output(nullptr) {
}

CopyOutputTask::~CopyOutputTask() {
}

uint64_t CopyOutputTask::eligible() {
	return earliest;
}

void CopyOutputTask::run(cudaStream_t stream) {
	uint64_t now = util::now(); // TODO: use chrono
	if (now < earliest) {
		throw TaskError(copyOutputTooEarly, "CopyOutputTask ran before it was eligible");
	}

	if (now > latest) {
		throw TaskError(copyOutputTooLate, "CopyOutputTask could not start in time");
	}

	// TODO: this should probably be preallocated; seems silly to fail here
	size_t output_size = rm->model->output_size(batch_size);
	this->output = manager->host_io_pool->alloc(output_size);
	if (this->output == nullptr) {
		throw TaskError(copyOutputHostAlloc, "CopyOutputTask failed to allocate memory from host_io_pool");
	}

	this->record_async_begin(stream);
	rm->model->transfer_output_from_device(batch_size, output, io_memory, stream);
	this->record_async_end(stream);
}

void CopyOutputTask::process_completion() {
	telemetry->async_duration = this->async_duration();
	this->success(output);
}

}
