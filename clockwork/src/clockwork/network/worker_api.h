#ifndef _CLOCKWORK_NETWORK_WORKER_API_H_
#define _CLOCKWORK_NETWORK_WORKER_API_H_

#include "clockwork/api/worker_api.h"
#include "clockwork/network/message.h"

namespace clockwork {
namespace network {

class error_result_tx : public msg_protobuf_tx<RES_ERROR, ErrorResultProto, workerapi::ErrorResult> {
public:
  virtual void set(workerapi::ErrorResult &result) {
  	msg.set_action_id(result.id);
  	msg.set_action_type(result.action_type);
  	msg.set_status(result.status);
  	msg.set_message(result.message);
    msg.set_action_received(result.action_received);
    msg.set_result_sent(result.result_sent);
  }
};

class error_result_rx : public msg_protobuf_rx<RES_ERROR, ErrorResultProto, workerapi::ErrorResult> {
public:
  virtual void get(workerapi::ErrorResult &result) {
  	result.id = msg.action_id();
  	result.action_type = msg.action_type();
  	result.status = msg.status();
  	result.message = msg.message();
    result.action_received = msg.action_received();
    result.result_sent = msg.result_sent();
  }
};

class load_model_from_disk_action_tx : public msg_protobuf_tx<ACT_LOAD_MODEL_FROM_DISK, LoadModelFromDiskActionProto, workerapi::LoadModelFromDisk> {
public:
  virtual void set(workerapi::LoadModelFromDisk &action) {
  	msg.set_action_id(action.id);
  	msg.set_model_id(action.model_id);
  	msg.set_no_of_copies(action.no_of_copies);
  	msg.set_model_path(action.model_path);
  	msg.set_earliest(action.earliest);
  	msg.set_latest(action.latest);
    msg.set_max_batch_size(action.max_batch_size);
    msg.set_max_exec_duration(action.max_exec_duration);
  }
};

class load_model_from_disk_action_rx : public msg_protobuf_rx<ACT_LOAD_MODEL_FROM_DISK, LoadModelFromDiskActionProto, workerapi::LoadModelFromDisk> {
public:
  virtual void get(workerapi::LoadModelFromDisk &action) {
  	action.id = msg.action_id();
  	action.action_type = workerapi::loadModelFromDiskAction;
  	action.model_id = msg.model_id();
  	action.no_of_copies = msg.no_of_copies();
  	action.model_path = msg.model_path();
  	action.earliest = msg.earliest();
  	action.latest = msg.latest();
    action.max_batch_size = msg.max_batch_size();
    action.max_exec_duration = msg.max_exec_duration();
  }
};

class load_model_from_disk_result_tx : public msg_protobuf_tx<RES_LOAD_MODEL_FROM_DISK, LoadModelFromDiskResultProto, workerapi::LoadModelFromDiskResult> {
public:
  virtual void set(workerapi::LoadModelFromDiskResult &result) {
  	msg.set_action_id(result.id);
  	msg.set_input_size(result.input_size);
  	msg.set_output_size(result.output_size);
  	msg.set_copies_created(result.copies_created);
  	for (unsigned batch_size : result.supported_batch_sizes) {
  		msg.add_supported_batch_sizes(batch_size);
  	}
  	msg.set_weights_size_in_cache(result.weights_size_in_cache);
    msg.set_num_weights_pages(result.num_weights_pages);
    msg.set_weights_load_time_nanos(result.weights_load_time_nanos);
    for (uint64_t &t : result.batch_size_exec_times_nanos) {
      msg.add_batch_size_exec_times_nanos(t);
    }
  	msg.mutable_timing()->set_begin(result.begin);
  	msg.mutable_timing()->set_end(result.end);
  	msg.mutable_timing()->set_duration(result.duration);
    msg.set_action_received(result.action_received);
    msg.set_result_sent(result.result_sent);
  }
};

class load_model_from_disk_result_rx : public msg_protobuf_rx<RES_LOAD_MODEL_FROM_DISK, LoadModelFromDiskResultProto, workerapi::LoadModelFromDiskResult> {
public:
  virtual void get(workerapi::LoadModelFromDiskResult &result) {
  	result.id = msg.action_id();
  	result.action_type = workerapi::loadModelFromDiskAction;
  	result.status = actionSuccess;
  	result.begin = msg.timing().begin();
  	result.end = msg.timing().end();
  	result.duration = msg.timing().duration();
  	result.input_size = msg.input_size();
  	result.output_size = msg.output_size();
  	result.copies_created = msg.copies_created();
  	for (unsigned i = 0; i < msg.supported_batch_sizes_size(); i++) {
  		result.supported_batch_sizes.push_back(msg.supported_batch_sizes(i));
  	}
  	result.weights_size_in_cache = msg.weights_size_in_cache();
    result.num_weights_pages = msg.num_weights_pages();
    result.weights_load_time_nanos = msg.weights_load_time_nanos();
    for (unsigned i = 0; i < msg.batch_size_exec_times_nanos_size(); i++) {
      result.batch_size_exec_times_nanos.push_back(msg.batch_size_exec_times_nanos(i));
    }
    result.action_received = msg.action_received();
    result.result_sent = msg.result_sent();
  }
};

class load_weights_action_tx : public msg_protobuf_tx<ACT_LOAD_WEIGHTS, LoadWeightsActionProto, workerapi::LoadWeights> {
public:
  virtual void set(workerapi::LoadWeights &action) {
  	msg.set_action_id(action.id);
  	msg.set_model_id(action.model_id);
  	msg.set_gpu_id(action.gpu_id);
  	msg.set_earliest(action.earliest);
  	msg.set_latest(action.latest);
  	msg.set_expected_duration(action.expected_duration);
  }
};

class load_weights_action_rx : public msg_protobuf_rx<ACT_LOAD_WEIGHTS, LoadWeightsActionProto, workerapi::LoadWeights> {
public:
  virtual void get(workerapi::LoadWeights &action) {
  	action.id = msg.action_id();
  	action.action_type = workerapi::loadWeightsAction;
  	action.model_id = msg.model_id();
  	action.gpu_id = msg.gpu_id();
  	action.earliest = msg.earliest();
  	action.latest = msg.latest();
  	action.expected_duration = msg.expected_duration();
  }
};

class load_weights_result_tx : public msg_protobuf_tx<RES_LOAD_WEIGHTS, LoadWeightsResultProto, workerapi::LoadWeightsResult> {
public:
  virtual void set(workerapi::LoadWeightsResult &result) {
  	msg.set_action_id(result.id);
  	msg.mutable_timing()->set_begin(result.begin);
  	msg.mutable_timing()->set_end(result.end);
  	msg.mutable_timing()->set_duration(result.duration);
    msg.set_action_received(result.action_received);
    msg.set_result_sent(result.result_sent);
  }
};

class load_weights_result_rx : public msg_protobuf_rx<RES_LOAD_WEIGHTS, LoadWeightsResultProto, workerapi::LoadWeightsResult> {
public:
  virtual void get(workerapi::LoadWeightsResult &result) {
  	result.id = msg.action_id();
  	result.action_type = workerapi::loadWeightsAction;
  	result.status = actionSuccess;
  	result.begin = msg.timing().begin();
  	result.end = msg.timing().end();
  	result.duration = msg.timing().duration();
    result.action_received = msg.action_received();
    result.result_sent = msg.result_sent();
  }
};

class evict_weights_action_tx : public msg_protobuf_tx<ACT_EVICT_WEIGHTS, EvictWeightsActionProto, workerapi::EvictWeights> {
public:
  virtual void set(workerapi::EvictWeights &action) {
  	msg.set_action_id(action.id);
  	msg.set_model_id(action.model_id);
  	msg.set_gpu_id(action.gpu_id);
  	msg.set_earliest(action.earliest);
  	msg.set_latest(action.latest);
  }
};

class evict_weights_action_rx : public msg_protobuf_rx<ACT_EVICT_WEIGHTS, EvictWeightsActionProto, workerapi::EvictWeights> {
public:
  virtual void get(workerapi::EvictWeights &action) {
  	action.id = msg.action_id();
  	action.action_type = workerapi::evictWeightsAction;
  	action.model_id = msg.model_id();
  	action.gpu_id = msg.gpu_id();
  	action.earliest = msg.earliest();
  	action.latest = msg.latest();
  }
};

class evict_weights_result_tx : public msg_protobuf_tx<RES_EVICT_WEIGHTS, EvictWeightsResultProto, workerapi::EvictWeightsResult> {
public:
  virtual void set(workerapi::EvictWeightsResult &result) {
  	msg.set_action_id(result.id);
  	msg.mutable_timing()->set_begin(result.begin);
  	msg.mutable_timing()->set_end(result.end);
  	msg.mutable_timing()->set_duration(result.duration);
    msg.set_action_received(result.action_received);
    msg.set_result_sent(result.result_sent);
  }
};

class evict_weights_result_rx : public msg_protobuf_rx<RES_EVICT_WEIGHTS, EvictWeightsResultProto, workerapi::EvictWeightsResult> {
public:
  virtual void get(workerapi::EvictWeightsResult &result) {
  	result.id = msg.action_id();
  	result.action_type = workerapi::evictWeightsAction;
  	result.status = actionSuccess;
  	result.begin = msg.timing().begin();
  	result.end = msg.timing().end();
  	result.duration = msg.timing().duration();
    result.action_received = msg.action_received();
    result.result_sent = msg.result_sent();
  }
};

class infer_action_tx : public msg_protobuf_tx_with_body<ACT_INFER, InferActionProto, workerapi::Infer> {
public:
  virtual void set(workerapi::Infer &action) {
  	msg.set_action_id(action.id);
  	msg.set_model_id(action.model_id);
  	msg.set_gpu_id(action.gpu_id);
  	msg.set_earliest(action.earliest);
  	msg.set_latest(action.latest);
  	msg.set_expected_duration(action.expected_duration);
  	msg.set_batch_size(action.batch_size);
    for (auto &size : action.input_sizes) {
      msg.add_input_sizes(size);
    }
  	body_len_ = action.input_size;
  	body_ = action.input;
  }
};

class infer_action_rx : public msg_protobuf_rx_with_body<ACT_INFER, InferActionProto, workerapi::Infer> {
public:
  virtual void get(workerapi::Infer &action) {
  	action.id = msg.action_id();
  	action.action_type = workerapi::inferAction;
  	action.model_id = msg.model_id();
  	action.gpu_id = msg.gpu_id();
  	action.earliest = msg.earliest();
  	action.latest = msg.latest();
  	action.expected_duration = msg.expected_duration();
  	action.batch_size = msg.batch_size();
  	action.input_size = body_len_;
    for (unsigned i = 0; i < msg.input_sizes_size(); i++) {
      action.input_sizes.push_back(msg.input_sizes(i));
    }
  	action.input = static_cast<char*>(body_);
  }
};

class infer_result_tx : public msg_protobuf_tx_with_body<RES_INFER, InferResultProto, workerapi::InferResult> {
public:
  virtual void set(workerapi::InferResult &result) {
  	msg.set_action_id(result.id);
	  msg.set_gpu_id(result.gpu_id);
    msg.set_gpu_clock_before(result.gpu_clock_before);
    msg.set_gpu_clock(result.gpu_clock);
  	msg.mutable_copy_input_timing()->set_begin(result.copy_input.begin);
  	msg.mutable_copy_input_timing()->set_end(result.copy_input.end);
  	msg.mutable_copy_input_timing()->set_duration(result.copy_input.duration);
  	msg.mutable_exec_timing()->set_begin(result.exec.begin);
  	msg.mutable_exec_timing()->set_end(result.exec.end);
  	msg.mutable_exec_timing()->set_duration(result.exec.duration);
  	msg.mutable_copy_output_timing()->set_begin(result.copy_output.begin);
  	msg.mutable_copy_output_timing()->set_end(result.copy_output.end);
  	msg.mutable_copy_output_timing()->set_duration(result.copy_output.duration);
    msg.set_action_received(result.action_received);
    msg.set_result_sent(result.result_sent);
  	body_len_ = result.output_size;
  	body_ = result.output;
  }
};

class infer_result_rx : public msg_protobuf_rx_with_body<RES_INFER, InferResultProto, workerapi::InferResult> {
public:
  virtual void get(workerapi::InferResult &result) {
  	result.id = msg.action_id();
  	result.action_type = workerapi::inferAction;
  	result.status = actionSuccess;
	  result.gpu_id = msg.gpu_id();
    result.gpu_clock_before = msg.gpu_clock_before();
    result.gpu_clock = msg.gpu_clock();
  	result.copy_input.begin = msg.copy_input_timing().begin();
  	result.copy_input.end = msg.copy_input_timing().end();
  	result.copy_input.duration = msg.copy_input_timing().duration();
  	result.exec.begin = msg.exec_timing().begin();
  	result.exec.end = msg.exec_timing().end();
  	result.exec.duration = msg.exec_timing().duration();
  	result.copy_output.begin = msg.copy_output_timing().begin();
  	result.copy_output.end = msg.copy_output_timing().end();
  	result.copy_output.duration = msg.copy_output_timing().duration();
    result.action_received = msg.action_received();
    result.result_sent = msg.result_sent();
  	result.output_size = body_len_;
  	result.output = static_cast<char*>(body_);
  }
};

class clear_cache_action_tx : public msg_protobuf_tx_with_body<ACT_CLEAR_CACHE, ClearCacheActionProto, workerapi::ClearCache> {
public:
  virtual void set(workerapi::ClearCache &action) {
	msg.set_action_id(action.id);
  }
};

class clear_cache_action_rx : public msg_protobuf_rx_with_body<ACT_CLEAR_CACHE, ClearCacheActionProto, workerapi::ClearCache> {
public:
  virtual void get(workerapi::ClearCache &action) {
	action.id = msg.action_id();
	action.action_type = workerapi::clearCacheAction;
  }
};

class clear_cache_result_tx : public msg_protobuf_tx_with_body<RES_CLEAR_CACHE, ClearCacheResultProto, workerapi::ClearCacheResult>{
public:
  virtual void set(workerapi::ClearCacheResult &result) {
	msg.set_action_id(result.id);
    msg.set_action_received(result.action_received);
    msg.set_result_sent(result.result_sent);
  }
};

class clear_cache_result_rx : public msg_protobuf_rx_with_body<RES_CLEAR_CACHE, ClearCacheResultProto, workerapi::ClearCacheResult>{
public:
  virtual void get(workerapi::ClearCacheResult &result) {
	result.id = msg.action_id();
	result.action_type = workerapi::clearCacheAction;
    result.action_received = msg.action_received();
    result.result_sent = msg.result_sent();
  }
};

class get_worker_state_action_tx : public msg_protobuf_tx_with_body<ACT_GET_WORKER_STATE, GetWorkerStateActionProto, workerapi::GetWorkerState> {
public:
  virtual void set(workerapi::GetWorkerState &action) {
	msg.set_action_id(action.id);
  }
};

class get_worker_state_action_rx : public msg_protobuf_rx_with_body<ACT_GET_WORKER_STATE, GetWorkerStateActionProto, workerapi::GetWorkerState> {
public:
  virtual void get(workerapi::GetWorkerState &action) {
	action.id = msg.action_id();
	action.action_type = workerapi::getWorkerStateAction;
  }
};

class get_worker_state_result_tx : public msg_protobuf_tx_with_body<RES_GET_WORKER_STATE, GetWorkerStateResultProto, workerapi::GetWorkerStateResult>{
public:

  void setgpu(const workerapi::GPUInfo &gpu, GPUInfoProto* proto) {
    proto->set_id(gpu.id);
    proto->set_weights_cache_size(gpu.weights_cache_size);
    proto->set_weights_cache_total_pages(gpu.weights_cache_total_pages);
    for (unsigned model_id : gpu.models) {
      proto->add_models(model_id);
    }
    proto->set_io_pool_size(gpu.io_pool_size);
    proto->set_workspace_pool_size(gpu.workspace_pool_size);    
  }

  void setmodel(const workerapi::ModelInfo &model, ModelInfoProto* proto) {
    proto->set_id(model.id);
    proto->set_source(model.source);
    proto->set_input_size(model.input_size);
    proto->set_output_size(model.output_size);
    for (unsigned batch_size : model.supported_batch_sizes) {
      proto->add_supported_batch_sizes(batch_size);
    }
    proto->set_weights_size(model.weights_size);
    proto->set_num_weights_pages(model.num_weights_pages);
    proto->set_weights_load_time_nanos(model.weights_load_time_nanos);
    for (auto &t : model.batch_size_exec_times_nanos) {
      proto->add_batch_size_exec_times_nanos(t);
    }
  }

  virtual void set(workerapi::GetWorkerStateResult &result) {
  	msg.set_action_id(result.id);
  	auto proto = msg.mutable_worker_memory_info();
  	
    workerapi::WorkerMemoryInfo& worker = result.worker;
    proto->set_page_size(worker.page_size);
    proto->set_host_weights_cache_size(worker.host_weights_cache_size);
    proto->set_host_io_pool_size(worker.host_io_pool_size);

    for (auto &gpu : worker.gpus) {
      GPUInfoProto* gpuproto = proto->add_gpus();
      setgpu(gpu, gpuproto);
    }

    for (auto &model : worker.models) {
      ModelInfoProto* modelproto = proto->add_models();
      setmodel(model, modelproto);
    }
    msg.set_action_received(result.action_received);
    msg.set_result_sent(result.result_sent);
  }
};

class get_worker_state_result_rx : public msg_protobuf_rx_with_body<RES_GET_WORKER_STATE, GetWorkerStateResultProto, workerapi::GetWorkerStateResult>{
public:

  void getgpu(workerapi::GPUInfo &gpu, const GPUInfoProto &proto) {
    gpu.id = proto.id();
    gpu.weights_cache_size = proto.weights_cache_size();
    gpu.weights_cache_total_pages = proto.weights_cache_total_pages();
    for (unsigned i = 0; i < proto.models_size(); i++) {
      gpu.models.push_back(proto.models(i));
    }
    gpu.io_pool_size = proto.io_pool_size();
    gpu.workspace_pool_size = proto.workspace_pool_size();
  }

  void getmodel(workerapi::ModelInfo &model, const ModelInfoProto &proto) {
    model.id = proto.id();
    model.source = proto.source();
    model.input_size = proto.input_size();
    model.output_size = proto.output_size();
    for (unsigned i = 0; i < proto.supported_batch_sizes_size(); i++) {
      model.supported_batch_sizes.push_back(proto.supported_batch_sizes(i));
    }
    model.weights_size = proto.weights_size();
    model.num_weights_pages = proto.num_weights_pages();
    model.weights_load_time_nanos = proto.weights_load_time_nanos();
    for (unsigned i = 0; i < proto.batch_size_exec_times_nanos_size(); i++) {
      model.batch_size_exec_times_nanos.push_back(proto.batch_size_exec_times_nanos(i));
    }
  }

  virtual void get(workerapi::GetWorkerStateResult &result) {
    result.id = msg.action_id();
    result.action_type = workerapi::getWorkerStateAction;
    workerapi::WorkerMemoryInfo& worker = result.worker;

    auto proto = msg.worker_memory_info();
    worker.page_size = proto.page_size();
    worker.host_weights_cache_size = proto.host_weights_cache_size();
    worker.host_io_pool_size = proto.host_io_pool_size();
    for (unsigned i = 0; i < proto.gpus_size(); i++) {
      workerapi::GPUInfo gpu;
      getgpu(gpu, proto.gpus(i));
      worker.gpus.push_back(gpu);
    }
    for (unsigned i = 0; i < proto.models_size(); i++) {
      workerapi::ModelInfo model;
      getmodel(model, proto.models(i));
      worker.models.push_back(model);
    }
    result.action_received = msg.action_received();
    result.result_sent = msg.result_sent();
  }
};

}
}

#endif
