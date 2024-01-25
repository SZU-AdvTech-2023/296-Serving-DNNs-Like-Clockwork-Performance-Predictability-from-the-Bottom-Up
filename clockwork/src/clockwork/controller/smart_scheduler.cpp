#include "clockwork/controller/smart_scheduler.h"

namespace clockwork {

// --- ExecutionProfiler
void SmartScheduler::ExecutionProfiler::set_batch_sizes(
    std::vector<unsigned> &sizes) {
  this->batch_sizes = sizes;
}

void SmartScheduler::ExecutionProfiler::set_estimates(
    std::map<unsigned, uint64_t> latencies) {
  estimates.clear();
  for (auto const &l : latencies) {
    estimates.insert(std::make_pair(l.first, l.second * clk_freq_default));
  }
}

void SmartScheduler::ExecutionProfiler::set_window_size(unsigned size) {
  this->window_size = size;
  for (auto const &s : batch_sizes) {
    sliding_windows[s] = SlidingWindow(size);
  }
}

uint64_t SmartScheduler::ExecutionProfiler::get_latency_estimate(
    unsigned batch_size) {
  return estimates[batch_size] / clk_freq;
}

unsigned SmartScheduler::ExecutionProfiler::get_max_batch_size(uint64_t slack,
                                                               unsigned limit) {
  unsigned current_max = 1;
  for (auto const &s : batch_sizes) {
    if (s > limit) {
      break;
    }
    if (get_latency_estimate(s) > slack) {
      break;
    }
    current_max = s;
  }
  return current_max;
}

void SmartScheduler::ExecutionProfiler::insert(unsigned batch, uint64_t latency,
                                               unsigned freq) {
  clk_freq = freq;
  auto it = sliding_windows.find(batch);
  it->second.insert(latency * freq);
  update_estimate(batch);
}

void SmartScheduler::ExecutionProfiler::update_estimate(unsigned batch) {
  SlidingWindow &sliding_window = sliding_windows[batch];
  if (sliding_window.get_size() == 0) {
    return;
  }
  unsigned rank = sliding_window.get_size() - 1;  // by default, get max value
  if (sliding_window.get_size() >= window_size) {
    rank = ceil(window_size * (percentile / 100.0)) - 1;
  }
  estimates[batch] = sliding_window.get_value(rank);
}

void SmartScheduler::ExecutionProfiler::update_all_estimates() {
  for (auto const &s : batch_sizes) {
    update_estimate(s);
  }
}

// --- WeightsProfiler
SmartScheduler::WeightsProfiler::WeightsProfiler(uint64_t estimate)
    : estimate(estimate) {}
uint64_t SmartScheduler::WeightsProfiler::get_estimate() { return estimate; }

// --- GPU
void SmartScheduler::GPU::update_lru(unsigned model_id) {
  auto pos =
      std::find(lru_loaded_models.begin(), lru_loaded_models.end(), model_id);
  if (pos != std::end(lru_loaded_models)) {
    lru_loaded_models.erase(pos);
  }
  lru_loaded_models.insert(lru_loaded_models.begin(), model_id);
}

void SmartScheduler::GPU::add_model(unsigned model_id) {
  loaded_models.insert(model_id);
}

void SmartScheduler::GPU::evict_model(unsigned model_id) {
  auto pos =
      std::find(lru_loaded_models.begin(), lru_loaded_models.end(), model_id);
  if (pos != std::end(lru_loaded_models)) {
    lru_loaded_models.erase(pos);
    loaded_models.erase(model_id);
  }
}

bool SmartScheduler::GPU::fits_model(unsigned model_num_pages) {
  return available_pages >= model_num_pages;
}

// --- Model
SmartScheduler::Model::Model(unsigned id, unsigned num_pages)
    : id(id), num_pages(num_pages) {}

bool SmartScheduler::Model::available_on(unsigned gpu_id) {
  return (std::find(gpus.begin(), gpus.end(), gpu_id) != gpus.end()) ? true
                                                                     : false;
}

uint64_t SmartScheduler::Model::earliest(unsigned gpu_id) {
  if (!available_on(gpu_id)) {
    CHECK(false)
        << " requested earliest() for a model that is not loaded on the gpu "
        << gpu_id << std::endl;
  }
  return weights_available_at[gpu_id];
}

bool SmartScheduler::Model::is_loaded_on_gpu(unsigned gpu_id) {
  return (std::find(gpus.begin(), gpus.end(), gpu_id) != gpus.end()) ? true
                                                                     : false;
}

void SmartScheduler::Model::add_gpu(unsigned gpu_id) {
  if (is_loaded_on_gpu(gpu_id)) {
    CHECK(false) << "[add_gpu] model is already loaded on gpu" << std::endl;
  }
  gpus.push_back(gpu_id);
}

void SmartScheduler::Model::remove_gpu(unsigned gpu_id) {
  if (!is_loaded_on_gpu(gpu_id)) {
    CHECK(false) << "[remove_gpu] model is not loaded on gpu" << std::endl;
  }
  gpus.erase(std::find(gpus.begin(), gpus.end(), gpu_id));
  weights_available_at.erase(weights_available_at.find(gpu_id));
}

// --- Request
SmartScheduler::Request::Request(uint64_t id, uint64_t user_request_id,
                                 unsigned model_id, uint64_t arrived,
                                 uint64_t deadline)
    : id(id),
      user_request_id(id),
      model_id(model_id),
      arrived(arrived),
      deadline(deadline),
      //   request(request),
      earliest(util::now()),
      start_time(0),
      finish_time(0) {}

unsigned SmartScheduler::Request::get_model_id() {
  // return request.model_id;
  return model_id;
}

// SmartScheduler::Request::~Request() {
//   // delete static_cast<char *>(request.input);
// }

// --- RequestBatch
SmartScheduler::RequestBatch::RequestBatch(uint64_t id, unsigned model_id,
                                           Request request)
    : id(id), batch_size(1), model_id(model_id) {
  earliest = request.earliest;
  deadline = request.deadline;
  start_time = request.start_time;
  finish_time = request.finish_time;
  requests.push_back(request);
}

void SmartScheduler::RequestBatch::add_to_batch(Request request) {
  if (requests.size() == batch_size) {
    CHECK(false) << "[add_to_batch] batch is already at capacity" << std::endl;
  }

  if (requests.size() == 0) {  // it's the first request
    deadline = request.deadline;
    earliest = request.earliest;
  } else {
    if (request.deadline < deadline) {
      deadline = request.deadline;
    }
  }
  requests.push_back(request);
}

unsigned SmartScheduler::RequestBatch::get_model_id() { return model_id; }

// --- SmartScheduler
SmartScheduler::SmartScheduler(uint64_t default_slo, unsigned max_gpus,
                               uint64_t max_exec_time, unsigned max_batch_size,
                               std::string action_telemetry_file)
    : default_slo(default_slo),
      max_gpus(max_gpus),
      max_exec_time(max_exec_time),
      max_batch_size(max_batch_size),
      action_telemetry_file(action_telemetry_file) {}

bool SmartScheduler::is_model_hot_somewhere(unsigned model_id) {
  return (global_cached_models.find(model_id) != global_cached_models.end())
             ? true
             : false;
}

void SmartScheduler::unset_global_cache_state(unsigned model_id) {
  if (global_cached_models.count(model_id) == 0) {
    CHECK(false) << " couldn't find the model in the global cache state "
                 << std::endl;
  }
  global_cached_models.erase(model_id);
}

void SmartScheduler::set_global_cache_state(unsigned model_id) {
  global_cached_models.insert(model_id);
}

void SmartScheduler::add_active_model(uint64_t action_id, unsigned gpu_id,
                                      unsigned model_id) {
  mtx_active_models.lock();
  active_models[action_id] = std::make_pair(gpu_id, model_id);
  mtx_active_models.unlock();
}

void SmartScheduler::remove_active_model(uint64_t action_id) {
  mtx_active_models.lock();
  if (active_models.find(action_id) == active_models.end()) {
    CHECK(false) << "[GPU::remove_active_model] couldn't find the action_id"
                 << std::endl;
  }
  active_models.erase(active_models.find(action_id));
  mtx_active_models.unlock();
}

bool SmartScheduler::is_model_active(unsigned gpu_id, unsigned model_id) {
  bool result = false;
  mtx_active_models.lock();
  for (auto &active_model_item : active_models) {
    unsigned m_id = active_model_item.second.second;
    unsigned g_id = active_model_item.second.first;
    if (model_id == m_id && gpu_id == g_id) {
      result = true;
      break;
    }
  }
  // TODO: change this to std::find_if
  mtx_active_models.unlock();
  return result;
}

void SmartScheduler::init_estimates(unsigned model_id,
                                    BatchedModelState &state) {
  execution_profiler[model_id] = ExecutionProfiler();
  weights_profiler[model_id] = WeightsProfiler(state.weights_transfer_duration);
  execution_profiler[model_id].set_batch_sizes(state.supported_batch_sizes);
  execution_profiler[model_id].set_estimates(state.exec_duration);
  execution_profiler[model_id].set_window_size(latency_profiling_window);
}

void SmartScheduler::set_estimates(unsigned model_id, unsigned batch_size,
                                   uint64_t exec_latency, unsigned freq) {
  mtx_profiler.lock();
  execution_profiler[model_id].insert(batch_size, exec_latency, freq);
  mtx_profiler.unlock();
}

uint64_t SmartScheduler::get_latency_estimate(unsigned model_id,
                                              unsigned batch_size) {
  mtx_profiler.lock();
  uint64_t wcet_estimate =
      execution_profiler[model_id].get_latency_estimate(batch_size);
  mtx_profiler.unlock();
  return wcet_estimate;
}

uint64_t SmartScheduler::get_weights_load_estimate(unsigned model_id) {
  return weights_profiler[model_id].get_estimate();
}

// runs after the very first initialization, we init the system state and
// model stats using the passed info
void SmartScheduler::start(
    std::vector<network::controller::WorkerConnection *> workers,
    ClockworkState &state) {
  this->workers = workers;
  std::string action_telemetry_file = "./controller_action_log.csv";
  logger = ControllerActionTelemetry::log_and_summarize(action_telemetry_file,
                                                        1000000000);
  action_id_seed = 1000;
  global_request_id = 1000;
  global_batch_id_seed = 1000;

  // -- gpu -> worker_idx, gpu_idx
  unsigned gpu_id = 0;
  for (auto &worker : state.workers) {
    for (auto &gpu : worker.gpus) {
      gpus[gpu_id] =
          GPU(gpu_id, worker.id, gpu.id, gpu.weights_cache_total_pages);
      gpu_id++;
      if (max_gpus == gpu_id) {
        break;
      }
    }
    if (max_gpus == gpu_id) {
      break;
    }
  }

  // -- parsing model stats
  for (auto &worker : state.workers) {
    for (auto &model : worker.models) {
      unsigned model_id = model.first;
      auto state = model.second;
      models[model_id] = Model(model_id, state.num_weights_pages);
      init_estimates(model_id, state);
    }
    break;  // assuming all the models are pre-loaded on all the workers
  }
  scheduler_thread = std::thread(&SmartScheduler::do_schedule, this);
}

void SmartScheduler::save_inference_callback(
    uint64_t request_id,
    std::function<void(clientapi::InferenceResponse &)> callback) {
  mtx_inference_callbacks.lock();
  inference_callbacks[request_id] = callback;
  mtx_inference_callbacks.unlock();
}

void SmartScheduler::save_action_callback(
    uint64_t action_id,
    std::function<void(std::shared_ptr<workerapi::Result>)> callback) {
  mtx_action_callbacks.lock();
  action_callbacks[action_id] = callback;
  mtx_action_callbacks.unlock();
}

void SmartScheduler::clientInfer(
    clientapi::InferenceRequest &request,
    std::function<void(clientapi::InferenceResponse &)> callback) {
  mtx_request_queue.lock();
  uint64_t request_id = ++global_request_id;
  uint64_t arrived = util::now();

  uint64_t slo_goal;
  if (request.slo_factor > 0) {
    slo_goal = get_latency_estimate(request.model_id, 1) * request.slo_factor;
  } else {
    slo_goal = default_slo;
  }
  slo_goal = ((slo_goal > 20000000) ? (slo_goal - 5000000)
                              : (slo_goal - 2 * network_transfer_latency)); // aim for a tighter slo
  uint64_t deadline = arrived + slo_goal;

  request_queue.push_back(Request(request_id, request.header.user_request_id,
                                  request.model_id, arrived, deadline));
  mtx_inference_callbacks.lock();
  inference_callbacks[request_id] = callback;
  mtx_inference_callbacks.unlock();
  mtx_request_queue.unlock();

  // Temporary:
  delete static_cast<char *>(request.input);
}

void SmartScheduler::resultFromWorker(
    std::shared_ptr<workerapi::Result> result) {
  mtx_action_callbacks.lock();
  if (action_callbacks.count(result->id) == 0) {
    CHECK(false) << " couldn't find the callback for action " << result->id
                 << std::endl;
  }

  remove_active_model(result->id);

  auto callback = action_callbacks[result->id];
  action_callbacks.erase(action_callbacks.find(result->id));
  mtx_action_callbacks.unlock();
  callback(result);
}

void SmartScheduler::add_request_to_gpu_request_queue(unsigned gpu_id,
                                                      Request request) {
  if (gpu_request_queue.count(gpu_id) == 0) {
    gpu_request_queue[gpu_id] = std::vector<Request>();
  }
  gpu_request_queue[gpu_id].push_back(request);
}

bool SmartScheduler::request_finish_time_compare(Request &lhs, Request &rhs) {
  return lhs.finish_time > rhs.finish_time;
}

bool SmartScheduler::batch_finish_time_compare(RequestBatch &lhs,
                                               RequestBatch &rhs) {
  return lhs.finish_time > rhs.finish_time;
}

bool SmartScheduler::compare_values(const std::pair<unsigned, uint64_t> &a,
                                    const std::pair<unsigned, uint64_t> &b) {
  return (a.second < b.second);
}

bool SmartScheduler::compare_values_decreasing(
    const std::pair<unsigned, unsigned> &a,
    const std::pair<unsigned, unsigned> &b) {
  return (a.second > b.second);
}

void SmartScheduler::get_models_to_preload(
    std::vector<Request> &request_queue,
    std::set<unsigned> &queued_request_models,
    std::map<unsigned, unsigned> &models_incoming_load,
    std::map<unsigned, unsigned> &models_to_load) {
  for (auto &request : request_queue) {
    unsigned model_id = request.get_model_id();
    queued_request_models.insert(model_id);
    if (!is_model_hot_somewhere(model_id)) {  // if the model is not on any gpu
                                              // nor scheduled to be loaded
      models_to_load[model_id] = UINT_MAX;
    }
    // we'd estimate the incoming load in the request queue, no matter the
    // model is already loaded or not
    if (models_incoming_load.find(model_id) == models_incoming_load.end()) {
      models_incoming_load[model_id] = 1;
    } else {
      models_incoming_load[model_id]++;
    }
  }
}

void SmartScheduler::calculate_gpu_load_estimates(
    std::map<unsigned, uint64_t> &estimated_gpu_load,
    std::map<unsigned, unsigned> &models_incoming_load,
    std::map<unsigned, unsigned> &models_to_load) {
  // NOTE: We overlook the pci_idle_at in caclulating the gpu load heuristic
  // init the gpu load estimates
  for (auto &gpu_load_item : gpus) {
    unsigned gpu_id = gpu_load_item.first;
    estimated_gpu_load[gpu_id] = std::max<uint64_t>(
        gpus[gpu_id].gpu_idle_at, util::now() + network_transfer_latency);
  }

  for (auto &model_load_element :
       models_incoming_load) {  // iterate all the models

    unsigned model_id = model_load_element.first;
    unsigned queued_load = model_load_element.second;

    if (is_model_hot_somewhere(model_id)) {  // if the model is already loaded
      for (auto &model_gpu_element :
           models[model_id].gpus) {  // iterate over the gpus
        unsigned gpu_id = model_gpu_element;

        estimated_gpu_load[gpu_id] +=
            (get_latency_estimate(model_id, 1) * queued_load /
             (models[model_id].gpus.size() +
              1));  // distribute the load   // if the model is a not loaded nor
                    // scheduled to be loaded
                    // yet. In other words: adding the estimated future load for
                    // the models which are not already loaded
      }
    } else if (models_to_load.find(model_id) != models_to_load.end() &&
               models_to_load[model_id] != UINT_MAX) {
      unsigned future_gpu_id = models_to_load[model_id];
      estimated_gpu_load[future_gpu_id] +=
          (get_weights_load_estimate(model_id) +
           get_latency_estimate(model_id, 1) * queued_load /
               (models[model_id].gpus.size() + 1));
    }
  }
}

unsigned SmartScheduler::find_least_loaded_gpu(
    std::map<unsigned, uint64_t> &estimated_gpu_load) {
  unsigned target_gpu_id = 0;
  uint64_t gpu_min_load = ULLONG_MAX;

  for (auto &gpu_load : estimated_gpu_load) {
    unsigned gpu_id = gpu_load.first;
    uint64_t estimated_load = gpu_load.second;
    if (estimated_load < gpu_min_load && gpu_id != UINT_MAX) {
      gpu_min_load = estimated_load;
      target_gpu_id = gpu_id;
    }
  }
  return target_gpu_id;
}

bool SmartScheduler::send_model_load_action(unsigned gpu_id,
                                            unsigned model_id) {
  gpus[gpu_id].pci_idle_at = std::max<uint64_t>(
      gpus[gpu_id].pci_idle_at, util::now() + network_transfer_latency);
  auto on_load_weights_complete =
      [this, model_id, gpu_id](std::shared_ptr<workerapi::Result> result) {
        if (auto load_weights_result =
                std::dynamic_pointer_cast<workerapi::LoadWeightsResult>(
                    result)) {
          set_telemetry_load_weights_result(load_weights_result);
        } else if (auto error =
                       std::dynamic_pointer_cast<workerapi::ErrorResult>(
                           result)) {
          //   gpus[gpu_id].evict_model(model_id);
          //   gpus[gpu_id].available_pages += models[model_id].num_pages;
          //   models[model_id].remove_gpu(gpu_id);
          //   if (models[model_id].gpus.size() == 0) {
          //     unset_global_cache_state(model_id);
          //   }
          set_telemetry_error_result(error);
          //   CHECK(false) << "Load weights failed \n";
          DEBUG_PRINT("Load weights failed: " + error->message);
        } else {
          CHECK(false) << "Load weights failed: Internal Controller Error";
        }
      };

  uint64_t scheduling_window_end =
      util::now() + schedule_ahead_length - network_transfer_latency;

  if ((gpus[gpu_id].pci_idle_at > scheduling_window_end) ||
      is_model_active(gpu_id, model_id)) {
    return false;
  }

  auto load_weights_action = std::make_shared<workerapi::LoadWeights>();
  load_weights_action->id = ++action_id_seed;
  load_weights_action->model_id = model_id;
  load_weights_action->gpu_id = gpus[gpu_id].gpu_index;
  load_weights_action->earliest = gpus[gpu_id].pci_idle_at;
  load_weights_action->latest =
      gpus[gpu_id].pci_idle_at + pci_slack;  // 1ms slack
  gpus[gpu_id].pci_idle_at += (get_weights_load_estimate(model_id) + pci_slack);
  models[model_id].weights_available_at[gpu_id] = gpus[gpu_id].pci_idle_at;

  set_telemetry_load_weights(gpus[gpu_id].worker_id, load_weights_action);

  mtx_action_callbacks.lock();
  action_callbacks[load_weights_action->id] = on_load_weights_complete;
  mtx_action_callbacks.unlock();

  add_active_model(load_weights_action->id, gpu_id, model_id);
  gpus[gpu_id].update_lru(model_id);
  gpus[gpu_id].available_pages -= models[model_id].num_pages;
  if (!is_model_hot_somewhere(model_id)) {
    set_global_cache_state(model_id);
  }
  if (!models[model_id].is_loaded_on_gpu(gpu_id)) {
    models[model_id].add_gpu(gpu_id);
  }
  std::vector<std::shared_ptr<workerapi::Action>> actions = {
      load_weights_action};
  workers[gpus[gpu_id].worker_id]->sendActions(actions);

  return true;
}

bool SmartScheduler::send_model_evict_action(unsigned gpu_id,
                                             unsigned victim_model_id) {
  gpus[gpu_id].pci_idle_at = std::max<uint64_t>(
      gpus[gpu_id].pci_idle_at, util::now() + network_transfer_latency);

  auto on_evict_weights_complete = [this, victim_model_id,
                                    gpu_id](std::shared_ptr<workerapi::Result>
                                                result) {
    if (auto evict_weights_result =
            std::dynamic_pointer_cast<workerapi::EvictWeightsResult>(result)) {
      set_telemetry_evict_weights_result(evict_weights_result);
    } else if (auto error =
                   std::dynamic_pointer_cast<workerapi::ErrorResult>(result)) {
      set_telemetry_error_result(error);
      CHECK(false) << "Evict weights failed \n";
      DEBUG_PRINT("Evict weights failed: " + error->message);
    } else {
      CHECK(false) << "Evict weights failed: Internal Controller Error";
    }
  };

  uint64_t scheduling_window_end =
      util::now() + schedule_ahead_length - network_transfer_latency;

  if ((gpus[gpu_id].pci_idle_at > scheduling_window_end) ||
      (is_model_active(gpu_id, victim_model_id))) {
    return false;
  }

  auto evict_weights_action = std::make_shared<workerapi::EvictWeights>();
  evict_weights_action->id = ++action_id_seed;
  evict_weights_action->model_id = victim_model_id;
  evict_weights_action->gpu_id = gpus[gpu_id].gpu_index;
  evict_weights_action->earliest = gpus[gpu_id].pci_idle_at;
  evict_weights_action->latest =
      gpus[gpu_id].pci_idle_at + pci_slack;  // latest to start

  gpus[gpu_id].pci_idle_at += (weights_evict_latency + pci_slack);

  set_telemetry_evict_weights(gpus[gpu_id].worker_id, evict_weights_action);
  mtx_action_callbacks.lock();
  action_callbacks[evict_weights_action->id] = on_evict_weights_complete;
  mtx_action_callbacks.unlock();
  add_active_model(evict_weights_action->id, gpu_id, victim_model_id);
  gpus[gpu_id].evict_model(victim_model_id);
  gpus[gpu_id].available_pages += models[victim_model_id].num_pages;
  models[victim_model_id].remove_gpu(gpu_id);
  if (models[victim_model_id].gpus.size() == 0) {
    unset_global_cache_state(victim_model_id);
  }
  std::vector<std::shared_ptr<workerapi::Action>> actions = {
      evict_weights_action};
  workers[gpus[gpu_id].worker_id]->sendActions(actions);
  return true;
}

void SmartScheduler::drop_request(Request &request) {
  clientapi::InferenceResponse response;
  //   response.header.user_request_id = request.request.header.user_request_id;
  response.header.user_request_id = request.user_request_id;
  response.header.status = clockworkError;
  response.header.message = "dropped before execution";
  response.model_id = request.get_model_id();
  response.output_size = 0;
  response.output = nullptr;
  mtx_inference_callbacks.lock();
  auto callback = inference_callbacks[request.id];
  callback(response);
  inference_callbacks.erase(inference_callbacks.find(request.id));
  mtx_inference_callbacks.unlock();
}

void SmartScheduler::assign_requests_to_gpu_local_queues() {
  std::map<unsigned, unsigned>
      model_multi_gpu_placement_idx;  // model_id ->
                                      // current_assigned_gpu_index
  gpu_request_queue.clear();

  std::vector<uint64_t> drop_list;
  for (auto &request : request_queue) {
    unsigned model_id = request.get_model_id();
    unsigned gpu_id;

    if (!is_model_hot_somewhere(model_id)) {
      if (request.deadline >
          (util::now() + network_transfer_latency + pci_slack +
           get_weights_load_estimate(model_id) +
           get_latency_estimate(
               model_id,
               1))) {  // if the model is not loaded and there's no way we can
                       // execute this request, just drop it here
        drop_list.push_back(request.id);
        drop_request(request);
      }
      continue;
    }

    if (models[model_id].gpus.size() > 0) {
      if (models[model_id].gpus.size() ==
          1) {  // if only one gpu assigned to this model
        gpu_id = models[model_id].gpus[0];  // assign the request to the
                                            // only gpu available
      } else {  // if more than one gpu is assigned to this model
        if (model_multi_gpu_placement_idx.find(model_id) ==
            model_multi_gpu_placement_idx.end()) {
          model_multi_gpu_placement_idx[model_id] =
              0;  // start by the first one
        } else {
          model_multi_gpu_placement_idx[model_id] =
              (model_multi_gpu_placement_idx[model_id] + 1) %
              models[model_id].gpus.size();  // iterate over assigned gpu idxs
        }
        gpu_id = model_multi_gpu_placement_idx[model_id];
      }
    } else {
      CHECK(false) << "there's a problem with model loading\n";
    }
    request.earliest =
        models[model_id]
            .weights_available_at[gpu_id];  // the earliest a request can
                                            // be executed is after the
                                            // weights are available
    add_request_to_gpu_request_queue(gpu_id, request);
  }

  for (auto &dropped_request_id : drop_list) {
    for (unsigned i = 0; request_queue.size(); i++) {
      if (dropped_request_id == request_queue[i].id) {
        request_queue.erase(request_queue.begin() + i);
        break;
      }
    }
  }
}

void SmartScheduler::calculate_model_load_evict_plan(
    std::map<unsigned, unsigned> &models_to_load,
    std::set<unsigned> &queued_request_models,
    std::map<unsigned, std::map<unsigned, std::vector<unsigned>>>
        &model_load_evict_plan) {
  std::map<unsigned, unsigned> tmp_gpu_available_pages;
  std::map<unsigned, std::vector<unsigned>> tmp_lru_loaded_models;

  // initializing temporary values, because we don't want to touch the actual
  // data yet
  for (auto &model_to_load : models_to_load) {
    unsigned gpu_id = model_to_load.second;
    tmp_gpu_available_pages[gpu_id] = gpus[gpu_id].available_pages;
    tmp_lru_loaded_models[gpu_id] = gpus[gpu_id].lru_loaded_models;
  }

  for (auto &model_to_load : models_to_load) {
    unsigned model_id = model_to_load.first;
    unsigned gpu_id = model_to_load.second;

    if (model_load_evict_plan.find(gpu_id) == model_load_evict_plan.end()) {
      model_load_evict_plan[gpu_id] =
          std::map<unsigned, std::vector<unsigned>>();
    }

    model_load_evict_plan[gpu_id][model_id] = std::vector<unsigned>();

    bool all_models_active = false;
    unsigned retry_counter = 0;

    while (tmp_gpu_available_pages[gpu_id] <
           models[model_id].num_pages) {  // evict until there's
                                          // enough space

      unsigned victim_model = tmp_lru_loaded_models[gpu_id].back();
      tmp_lru_loaded_models[gpu_id].pop_back();

      while (
          // (queued_request_models.find(victim_model) !=
          //  queued_request_models.end()) ||
          is_model_active(
              gpu_id, victim_model)) {  // if the model is being requested put
                                        // it back to the lru list at the head

        tmp_lru_loaded_models[gpu_id].insert(
            tmp_lru_loaded_models[gpu_id].begin(), victim_model);
        victim_model = tmp_lru_loaded_models[gpu_id].back();
        tmp_lru_loaded_models[gpu_id].pop_back();

        if (++retry_counter > tmp_lru_loaded_models[gpu_id].size()) {
          all_models_active = true;
          break;
        }
      }
      if (all_models_active) {
        break;
      }
      model_load_evict_plan[gpu_id][model_id].push_back(victim_model);
      tmp_gpu_available_pages[gpu_id] += models[victim_model].num_pages;
    }

    if (all_models_active) {  // if we couldn't evict, don't bother loading the
                              // main model
      model_load_evict_plan[gpu_id].erase(
          model_load_evict_plan[gpu_id].find(model_id));
    } else {
      tmp_lru_loaded_models[gpu_id].insert(
          tmp_lru_loaded_models[gpu_id].begin(), model_id);
      tmp_gpu_available_pages[gpu_id] -= models[model_id].num_pages;
    }
  }
}

// the scheduler thread runs this function every EPOCH
void SmartScheduler::do_schedule() {
  while (true) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(scheduling_epoch));
    if (request_queue.size() == 0) {
      continue;
    }

    // conservative early locking of the global request queue
    mtx_request_queue.lock();
    // STEP:  do a quick pass on the request queue to see if we need to load
    // any models

    std::set<unsigned>
        queued_request_models;  // what models have requests in the queue
    std::map<unsigned, unsigned> models_to_load;  // model_id -> gpu_id
    std::map<unsigned, unsigned>
        models_incoming_load;  // model_id -> incoming_load

    get_models_to_preload(request_queue, queued_request_models,
                          models_incoming_load, models_to_load);

    // prioritizing models with the highest load
    std::vector<std::pair<unsigned, unsigned>> sorted_model_incoming_loads;
    for (auto &item : models_incoming_load) {
      sorted_model_incoming_loads.push_back(item);
    }
    sort(sorted_model_incoming_loads.begin(), sorted_model_incoming_loads.end(),
         compare_values_decreasing);

    // assign the least loaded gpu to each model load candidate
    for (auto &model_load_item : sorted_model_incoming_loads) {
      unsigned model_id = model_load_item.first;
      if (models_to_load.find(model_id) != models_to_load.end()) {
        std::map<unsigned, uint64_t>
            estimated_gpu_load;  // gpu_id -> estimated_idle_at
        calculate_gpu_load_estimates(estimated_gpu_load, models_incoming_load,
                                     models_to_load);
        models_to_load[model_id] = find_least_loaded_gpu(estimated_gpu_load);
      }
    }

    // Only send the actions that can be done within the next schedule ahead
    // window
    std::map<unsigned, std::map<unsigned, std::vector<unsigned>>>
        model_load_evict_plan;  // gpu_id -> to_load_model, [to_evict_models]

    calculate_model_load_evict_plan(models_to_load, queued_request_models,
                                    model_load_evict_plan);

    // -- by now we have the model gpu id -> model evict lists
    // STEP: make evict / load actions
    for (auto &gpu_pci_ops : model_load_evict_plan) {
      unsigned gpu_id = gpu_pci_ops.first;
      bool can_schedule_more = true;

      for (auto &model_pci_ops_item : gpu_pci_ops.second) {
        unsigned model_id = model_pci_ops_item.first;

        for (auto &victim_model_id : model_pci_ops_item.second) {
          if (!can_schedule_more) {
            break;
          }
          can_schedule_more = send_model_evict_action(gpu_id, victim_model_id);
        }
        if (!can_schedule_more) {
          break;
        }
        can_schedule_more = send_model_load_action(gpu_id, model_id);
      }
    }

    // STEP: assigning requests to the gpus
    assign_requests_to_gpu_local_queues();

    // do local scheduling on each gpu queue
    for (auto &local_request_item : gpu_request_queue) {
      if (local_request_item.second.size() ==
          0) {  // if there's no request assigned to a gpu, don't bother
                // calling gpu_local_schedule
        continue;
      }
      gpu_local_batch_schedule(local_request_item.first,
                               local_request_item.second);
    }

    // STEP: check if we need to replicate any model that has high load
    // decide_replication();
    model_drop_count.clear();
    gpu_request_queue.clear();
    mtx_request_queue.unlock();
  }
}

void SmartScheduler::gpu_local_schedule(
    unsigned gpu_id, std::vector<Request> &local_request_queue) {
  std::vector<uint64_t> drop_list;
  std::vector<Request> stash;
  std::set<uint64_t> conflicting_requests;

  gpus[gpu_id].gpu_idle_at = std::max<uint64_t>(
      gpus[gpu_id].gpu_idle_at, util::now() + network_transfer_latency);

  // STEP: early proning --- drop the requests if their deadline is already
  // passed or we can't make it to the deadline or not enough slack to load
  // the model
  unsigned drop_count_tmp = 0;
  for (unsigned i = 0; i < local_request_queue.size(); i++) {
    unsigned model_id = local_request_queue[i].get_model_id();
    uint64_t execution_duration = get_latency_estimate(model_id, 1);
    // we set the earliest, start and finish times while we're iterating over
    // the local_request_queue
    local_request_queue[i].earliest =
        std::max<uint64_t>(gpus[gpu_id].gpu_idle_at,
                           models[model_id].weights_available_at[gpu_id]);
    local_request_queue[i].finish_time = local_request_queue[i].deadline;
    local_request_queue[i].start_time =
        local_request_queue[i].deadline - execution_duration;

    if (local_request_queue[i].earliest + execution_duration >
        local_request_queue[i]
            .deadline) {  // if the infer request cannot be done by any means,
                          // even if sent to the gpu right now
      drop_list.push_back(local_request_queue[i].id);  // add to the drop_list
      drop_count_tmp++;
      if (model_drop_count.find(model_id) ==
          model_drop_count.end()) {  // keep the drop count of each model, so
                                     // we would decide if we want to load
                                     // another instance to alieviate the load
        model_drop_count[model_id] = 1;
      } else {
        model_drop_count[model_id]++;
      }
    }
  }

  // early dropping the requests that cannot be scheduled
  for (auto &request_id : drop_list) {
    // std::cout << "dropping " << request_id << " ...\n";
    for (unsigned i = 0; i < local_request_queue.size(); i++) {
      if (local_request_queue[i].id == request_id) {
        drop_request(local_request_queue[i]);
        local_request_queue.erase(
            local_request_queue.begin() +
            i);  // remove the request from the request queue
        break;
      }
    }
  }

  unsigned stash_size_prev = stash.size();
  unsigned stash_size = stash.size();
  do {
    // STEP: initial placement ---
    sort(local_request_queue.begin(), local_request_queue.end(),
         request_finish_time_compare);  // sort based on finish_time

    if (local_request_queue.size() > 1) {
      for (int i = local_request_queue.size() - 1; i > 0; i--) {
        for (int j = i - 1; j >= 0; j--) {
          //   std::cout << "i: " << i << " j: " << j << "\n";
          if (i != j && !(local_request_queue[j].finish_time <=
                          local_request_queue[i].start_time)) {
            conflicting_requests.insert(local_request_queue[i].id);
            conflicting_requests.insert(local_request_queue[j].id);
          }
        }
      }
    }

    // STEP: take out the conflicting requests
    for (auto request_id : conflicting_requests) {
      for (unsigned i = 0;; i++) {
        if (i >= local_request_queue.size()) {
          break;
        }
        if (local_request_queue[i].id == request_id) {
          stash.push_back(local_request_queue[i]);
          local_request_queue.erase(local_request_queue.begin() + i);
          break;
        }
      }
    }

    // STEP: resolve round

    std::vector<uint64_t> to_remove_from_stash;
    unsigned stash_size = stash.size();

    for (unsigned i = 0; i < stash.size(); i++) {
      // STEP: placing at the tail
      if (local_request_queue.size() == 0) {
        local_request_queue.push_back(stash[i]);
        to_remove_from_stash.push_back(stash[i].id);
        continue;
      } else if ((local_request_queue.size() > 0) &&
                 (stash[i].earliest >=
                  local_request_queue[local_request_queue.size() - 1]
                      .finish_time) &&
                 (stash[i].deadline >=
                  stash[i].start_time +
                      get_latency_estimate(stash[i].get_model_id(),
                                           1))) {  // start time after
        // finish? or local_queue
        // empty? OK, put it at the tail
        // add to the tail
        // mark to remove from the stash
        stash[i].start_time =
            local_request_queue[local_request_queue.size() - 1].finish_time;
        stash[i].finish_time = stash[i].start_time +
                               get_latency_estimate(stash[i].get_model_id(), 1);

        local_request_queue.push_back(stash[i]);
        to_remove_from_stash.push_back(stash[i].id);
        sort(local_request_queue.begin(), local_request_queue.end(),
             request_finish_time_compare);  // sort based on finish_time
        continue;
      }

      // STEP: Place in a hole
      bool placed_in_a_hole = false;
      for (int j = local_request_queue.size() - 1; j > 0; j--) {
        if (local_request_queue[j - 1].finish_time +
                    get_latency_estimate(stash[i].get_model_id(), 1) <=
                stash[i].deadline &&
            (stash[i].earliest <=
             local_request_queue[j].start_time -
                 get_latency_estimate(stash[i].get_model_id(), 1)) &&
            (local_request_queue[j].start_time -
                 local_request_queue[j - 1].finish_time >=
             get_latency_estimate(stash[i].get_model_id(),
                                  1))) {  // if the request fits
                                          // between two
                                          // scheduled requests
          stash[i].finish_time = local_request_queue[j].start_time;
          stash[i].start_time =
              local_request_queue[j].start_time -
              get_latency_estimate(stash[i].get_model_id(), 1);
          local_request_queue.push_back(stash[i]);
          to_remove_from_stash.push_back(stash[i].id);
          sort(local_request_queue.begin(), local_request_queue.end(),
               request_finish_time_compare);
          placed_in_a_hole = true;
          break;
        }
      }
      // STEP: Place at the head
      if (!placed_in_a_hole &&
          local_request_queue[0].start_time >=
              stash[i].earliest +
                  get_latency_estimate(stash[i].get_model_id(), 1)) {
        stash[i].finish_time = local_request_queue[0].start_time;
        stash[i].start_time = local_request_queue[0].start_time -
                              get_latency_estimate(stash[i].get_model_id(), 1);
        local_request_queue.push_back(stash[i]);
        to_remove_from_stash.push_back(stash[i].id);
        sort(local_request_queue.begin(), local_request_queue.end(),
             request_finish_time_compare);
        continue;
      }
    }

    // STEP: remove scheduled requests from the stash
    for (auto request_id : to_remove_from_stash) {
      for (int i = 0;; i++) {
        if (i >= stash.size()) {
          break;
        }
        if (stash[i].id == request_id) {
          stash.erase(stash.begin() + i);
        }
      }
    }

    // STEP: compressing the schedule
    if (local_request_queue.size() > 0) {
      local_request_queue[0].start_time = std::max<uint64_t>(
          local_request_queue[0].earliest,
          gpus[gpu_id].gpu_idle_at);  // shift the first element to the
                                      // earliest time possible
      local_request_queue[0].finish_time =
          local_request_queue[0].start_time +
          get_latency_estimate(local_request_queue[0].get_model_id(), 1);
      for (unsigned i = 1; i < local_request_queue.size(); i++) {
        local_request_queue[i].start_time =
            std::max<uint64_t>(local_request_queue[i].earliest,
                               local_request_queue[i - 1].finish_time);
        local_request_queue[i].finish_time =
            local_request_queue[i].start_time +
            get_latency_estimate(local_request_queue[i].get_model_id(), 1);
      }
    }

    stash_size = stash.size();
  } while (stash_size_prev != stash_size);

  // STEP append all the remaining stashed to the drop_list
  // drop all stashed which can't be started in the current epoch
  for (auto request_item : stash) {
    if (request_item.deadline <= gpus[gpu_id].gpu_idle_at + scheduling_epoch) {
      clientapi::InferenceResponse response;
      response.header.user_request_id = request_item.user_request_id;
      response.header.status = clockworkError;
      response.header.message = "dropped before execution";
      response.output_size = 0;
      response.output = nullptr;

      mtx_inference_callbacks.lock();
      auto callback = inference_callbacks[request_item.id];
      inference_callbacks.erase(inference_callbacks.find(request_item.id));
      mtx_inference_callbacks.unlock();
      callback(response);
      // add to drop_list
      drop_list.push_back(request_item.id);
      // remove the callback
      break;
    } else {
      unsigned model_id = request_item.get_model_id();
      if (model_drop_count.find(model_id) == model_drop_count.end()) {
        model_drop_count[model_id] = 1;
      } else {
        model_drop_count[model_id]++;
      }
    }
  }

  std::vector<std::shared_ptr<workerapi::Action>> actions;

  // STEP: create infer actions
  unsigned index;
  for (index = 0; index < local_request_queue.size(); index++) {
    if (local_request_queue[index].start_time >
        gpus[gpu_id].gpu_idle_at + schedule_ahead_length) {
      break;
    }
    auto infer = std::make_shared<workerapi::Infer>();
    uint64_t request_id = local_request_queue[index].id;
    unsigned user_request_id = local_request_queue[index].user_request_id;
    drop_list.push_back(request_id);
    infer->id = ++action_id_seed;
    infer->model_id = local_request_queue[index].get_model_id();
    infer->gpu_id = gpus[gpu_id].gpu_index;
    infer->batch_size = 1;
    infer->input_size = 0;
    infer->input = nullptr;
    infer->earliest = local_request_queue[index].start_time;
    infer->latest = local_request_queue[index].start_time + infer_slack;

    // update the gpu timings
    gpus[gpu_id].gpu_idle_at = local_request_queue[index].finish_time;

    // update LRU model on the GPU
    unsigned model_id = infer->model_id;
    gpus[gpu_id].update_lru(infer->model_id);

    auto infer_action_complete =
        [this, request_id, model_id,
         user_request_id](std::shared_ptr<workerapi::Result> result) {
          if (auto infer_result =
                  std::dynamic_pointer_cast<workerapi::InferResult>(result)) {
            set_telemetry_infer_result(infer_result);

            set_estimates(model_id, 1, infer_result->exec.duration,
                          infer_result->gpu_clock);

            mtx_inference_callbacks.lock();
            auto callback = inference_callbacks[request_id];
            inference_callbacks.erase(inference_callbacks.find(request_id));
            mtx_inference_callbacks.unlock();

            clientapi::InferenceResponse response;
            response.header.user_request_id = user_request_id;
            response.header.status = clockworkSuccess;
            response.header.message = "";
            response.output_size = 0;
            response.output = nullptr;
            response.model_id = model_id;
            response.batch_size = 1;
            callback(response);

            delete infer_result->output;
            infer_result->output = nullptr;

          } else {
            std::string error_message = "Internal Controller Error";

            if (auto error =
                    std::dynamic_pointer_cast<workerapi::ErrorResult>(result)) {
              error_message = error->message;
              set_telemetry_error_result(error);
            }

            mtx_inference_callbacks.lock();
            auto callback = inference_callbacks[request_id];
            inference_callbacks.erase(inference_callbacks.find(request_id));
            mtx_inference_callbacks.unlock();

            clientapi::InferenceResponse response;
            response.header.user_request_id = user_request_id;
            response.header.status = clockworkError;
            response.header.message = error_message;
            response.output_size = 0;
            response.output = nullptr;
            response.model_id = model_id;
            callback(response);
          }
        };

    add_active_model(infer->id, gpu_id, model_id);
    mtx_action_callbacks.lock();
    action_callbacks[infer->id] = infer_action_complete;
    mtx_action_callbacks.unlock();

    set_telemetry_infer(gpus[gpu_id].worker_id, infer);

    actions.push_back(infer);
  }

  // STEP: send actions to the worker

  if (actions.size() > 0) {
    workers[gpus[gpu_id].worker_id]->sendActions(actions);
    local_request_queue.erase(local_request_queue.begin(),
                              local_request_queue.begin() + index);
  }

  // STEP: remove the scheduled requests from the request queue
  for (auto req_id : drop_list) {
    for (unsigned idx = 0; idx < request_queue.size(); idx++) {
      if (request_queue[idx].id == req_id) {
        request_queue.erase(request_queue.begin() + idx);
      }
    }
  }
}

unsigned SmartScheduler::get_best_batchsize(unsigned model_id,
                                            unsigned queue_size) {
  unsigned best_batch_size = 1;
  for (auto &b : execution_profiler[model_id].batch_sizes) {
    if ((queue_size >= b) && (b <= max_batch_size) && (b > best_batch_size) &&
        (get_latency_estimate(model_id, b) <= max_exec_time)) {
      best_batch_size = b;
    }
  }
  return best_batch_size;
}

void SmartScheduler::gpu_local_batch_schedule(
    unsigned gpu_id, std::vector<Request> &local_request_queue) {
  std::vector<uint64_t> drop_list;
  std::vector<RequestBatch> stash;
  std::set<uint64_t> conflicting_batches;

  std::map<unsigned, std::vector<Request>> model_queues;

  gpus[gpu_id].gpu_idle_at = std::max<uint64_t>(
      gpus[gpu_id].gpu_idle_at, util::now() + network_transfer_latency);

  // STEP: early proning --- drop the requests if their deadline is already
  // passed or we can't make it to the deadline or not enough slack to load
  // the model
  unsigned drop_count_tmp = 0;
  for (unsigned i = 0; i < local_request_queue.size(); i++) {
    unsigned model_id = local_request_queue[i].get_model_id();
    uint64_t execution_duration = get_latency_estimate(model_id, 1);
    // we set the earliest, start and finish times while we're iterating over
    // the local_request_queue
    local_request_queue[i].earliest =
        std::max<uint64_t>(gpus[gpu_id].gpu_idle_at,
                           models[model_id].weights_available_at[gpu_id]);
    local_request_queue[i].finish_time = local_request_queue[i].deadline;
    local_request_queue[i].start_time =
        local_request_queue[i].deadline - execution_duration;

    if (local_request_queue[i].earliest + execution_duration >
        local_request_queue[i]
            .deadline) {  // if the infer request cannot be done by any means,
                          // even if sent to the gpu right now
      drop_list.push_back(local_request_queue[i].id);  // add to the drop_list
      drop_count_tmp++;
      if (model_drop_count.find(model_id) ==
          model_drop_count.end()) {  // keep the drop count of each model, so
                                     // we would decide if we want to load
                                     // another instance to alieviate the load
        model_drop_count[model_id] = 1;
      } else {
        model_drop_count[model_id]++;
      }
    } else {
      if (model_queues.find(model_id) == model_queues.end()) {
        model_queues[model_id] = std::vector<Request>();
      }
      model_queues[model_id].push_back(local_request_queue[i]);
    }
  }

  // early dropping the requests that cannot be scheduled
  for (auto &request_id : drop_list) {
    // std::cout << "dropping " << request_id << " ...\n";
    for (unsigned i = 0; i < local_request_queue.size(); i++) {
      if (local_request_queue[i].id == request_id) {
        drop_request(local_request_queue[i]);
        local_request_queue.erase(
            local_request_queue.begin() +
            i);  // remove the request from the request queue
        break;
      }
    }
  }

  // ----- BATCH 'EM UP

  std::vector<RequestBatch> local_batch_queue;

  for (auto &model_queue : model_queues) {
    unsigned model_id = model_queue.first;
    unsigned processed = 0;
    unsigned model_queue_size = model_queue.second.size();

    while (processed < model_queue_size) {
      unsigned batch_size =
          get_best_batchsize(model_id, model_queue_size - processed);
      RequestBatch new_batch(++global_batch_id_seed, model_id, batch_size);
      for (unsigned i = 0; i < batch_size; i++) {
        new_batch.add_to_batch(model_queue.second[processed]);
        processed++;
      }
      new_batch.finish_time = new_batch.deadline;
      new_batch.start_time =
          new_batch.finish_time - get_latency_estimate(model_id, batch_size);
      local_batch_queue.push_back(new_batch);
    }
  }

  // --- EDF PLACEMENT

  unsigned stash_size_prev = stash.size();
  unsigned stash_size = stash.size();
  do {
    // STEP: initial placement ---
    sort(local_batch_queue.begin(), local_batch_queue.end(),
         batch_finish_time_compare);  // sort based on finish_time

    if (local_batch_queue.size() > 1) {
      for (int i = local_batch_queue.size() - 1; i > 0; i--) {
        for (int j = i - 1; j >= 0; j--) {
          if (i != j && !(local_batch_queue[j].finish_time <=
                          local_batch_queue[i].start_time)) {
            conflicting_batches.insert(local_batch_queue[i].id);
            conflicting_batches.insert(local_batch_queue[j].id);
          }
        }
      }
    }

    // STEP: take out the conflicting requests
    for (auto batch_id : conflicting_batches) {
      for (unsigned i = 0;; i++) {
        if (i >= local_batch_queue.size()) {
          break;
        }
        if (local_batch_queue[i].id == batch_id) {
          stash.push_back(local_batch_queue[i]);
          local_batch_queue.erase(local_batch_queue.begin() + i);
          break;
        }
      }
    }

    // STEP: resolve round

    std::vector<uint64_t> to_remove_from_stash;
    unsigned stash_size = stash.size();

    for (unsigned i = 0; i < stash.size(); i++) {
      // STEP: placing at the tail
      if (local_batch_queue.size() == 0) {
        local_batch_queue.push_back(stash[i]);
        to_remove_from_stash.push_back(stash[i].id);
        continue;
      } else if ((local_batch_queue.size() > 0) &&
                 (stash[i].earliest >=
                  local_batch_queue[local_batch_queue.size() - 1]
                      .finish_time) &&
                 (stash[i].deadline >=
                  stash[i].start_time +
                      get_latency_estimate(
                          stash[i].get_model_id(),
                          stash[i].batch_size))) {  // start time after
        // finish? or local_queue
        // empty? OK, put it at the tail
        // add to the tail
        // mark to remove from the stash
        stash[i].start_time =
            local_batch_queue[local_batch_queue.size() - 1].finish_time;
        stash[i].finish_time =
            stash[i].start_time +
            get_latency_estimate(stash[i].get_model_id(), stash[i].batch_size);

        local_batch_queue.push_back(stash[i]);
        to_remove_from_stash.push_back(stash[i].id);
        sort(local_batch_queue.begin(), local_batch_queue.end(),
             batch_finish_time_compare);  // sort based on finish_time
        continue;
      }

      // STEP: Place in a hole
      bool placed_in_a_hole = false;
      for (int j = local_batch_queue.size() - 1; j > 0; j--) {
        if (local_batch_queue[j - 1].finish_time +
                    get_latency_estimate(stash[i].get_model_id(),
                                         stash[i].batch_size) <=
                stash[i].deadline &&
            (stash[i].earliest <=
             local_batch_queue[j].start_time -
                 get_latency_estimate(stash[i].get_model_id(),
                                      stash[i].batch_size)) &&
            (local_batch_queue[j].start_time -
                 local_batch_queue[j - 1].finish_time >=
             get_latency_estimate(
                 stash[i].get_model_id(),
                 stash[i].batch_size))) {  // if the request fits
                                           // between two
                                           // scheduled requests
          stash[i].finish_time = local_batch_queue[j].start_time;
          stash[i].start_time = local_batch_queue[j].start_time -
                                get_latency_estimate(stash[i].get_model_id(),
                                                     stash[i].batch_size);
          local_batch_queue.push_back(stash[i]);
          to_remove_from_stash.push_back(stash[i].id);
          sort(local_batch_queue.begin(), local_batch_queue.end(),
               batch_finish_time_compare);
          placed_in_a_hole = true;
          break;
        }
      }
      // STEP: Place at the head
      if (!placed_in_a_hole &&
          local_batch_queue[0].start_time >=
              stash[i].earliest + get_latency_estimate(stash[i].get_model_id(),
                                                       stash[i].batch_size)) {
        stash[i].finish_time = local_batch_queue[0].start_time;
        stash[i].start_time =
            local_batch_queue[0].start_time -
            get_latency_estimate(stash[i].get_model_id(), stash[i].batch_size);
        local_batch_queue.push_back(stash[i]);
        to_remove_from_stash.push_back(stash[i].id);
        sort(local_batch_queue.begin(), local_batch_queue.end(),
             batch_finish_time_compare);
        continue;
      }
    }

    // STEP: remove scheduled requests from the stash
    for (auto request_id : to_remove_from_stash) {
      for (int i = 0;; i++) {
        if (i >= stash.size()) {
          break;
        }
        if (stash[i].id == request_id) {
          stash.erase(stash.begin() + i);
        }
      }
    }

    // STEP: compressing the schedule
    if (local_batch_queue.size() > 0) {
      local_batch_queue[0].start_time = std::max<uint64_t>(
          local_batch_queue[0].earliest,
          gpus[gpu_id].gpu_idle_at);  // shift the first element to the
                                      // earliest time possible
      local_batch_queue[0].finish_time =
          local_batch_queue[0].start_time +
          get_latency_estimate(local_batch_queue[0].get_model_id(),
                               local_batch_queue[0].batch_size);
      for (unsigned i = 1; i < local_batch_queue.size(); i++) {
        local_batch_queue[i].start_time =
            std::max<uint64_t>(local_batch_queue[i].earliest,
                               local_batch_queue[i - 1].finish_time);
        local_batch_queue[i].finish_time =
            local_batch_queue[i].start_time +
            get_latency_estimate(local_batch_queue[i].get_model_id(),
                                 local_batch_queue[i].batch_size);
      }
    }

    stash_size = stash.size();
  } while (stash_size_prev != stash_size);

  // STEP append all the remaining stashed to the drop_list
  // drop all stashed which can't be started in the current epoch

  for (auto &batch_item : stash) {
    for (auto &request_item : batch_item.requests) {
      if ((request_item.deadline -
           get_latency_estimate(request_item.get_model_id(), 1)) <=
          gpus[gpu_id].gpu_idle_at + scheduling_epoch) {
        clientapi::InferenceResponse response;
        response.header.user_request_id = request_item.user_request_id;
        response.header.status = clockworkError;
        response.header.message = "dropped before execution";
        response.output_size = 0;
        response.output = nullptr;

        mtx_inference_callbacks.lock();
        auto callback = inference_callbacks[request_item.id];
        inference_callbacks.erase(inference_callbacks.find(request_item.id));
        mtx_inference_callbacks.unlock();
        callback(response);
        // add to drop_list
        drop_list.push_back(request_item.id);

        break;
      } else {
        unsigned model_id = request_item.get_model_id();
        if (model_drop_count.find(model_id) == model_drop_count.end()) {
          model_drop_count[model_id] = 1;
        } else {
          model_drop_count[model_id]++;
        }
      }
    }
  }

  std::vector<std::shared_ptr<workerapi::Action>> actions;

  // STEP: create infer actions
  unsigned index;
  for (index = 0; index < local_batch_queue.size(); index++) {
    if (local_batch_queue[index].start_time >
        gpus[gpu_id].gpu_idle_at + schedule_ahead_length) {
      break;
    }

    std::map<uint64_t, uint64_t>
        client_requests;  // request_id -> user_request_id

    for (auto &request_item : local_batch_queue[index].requests) {
      uint64_t request_id = request_item.id;
      client_requests[request_id] = request_item.user_request_id;
      drop_list.push_back(request_id);
    }

    uint64_t batch_id = local_batch_queue[index].id;
    unsigned batch_size = local_batch_queue[index].batch_size;

    mtx_batch_storage.lock();
    batch_storage[batch_id] = local_batch_queue[index].requests;
    mtx_batch_storage.unlock();

    auto infer = std::make_shared<workerapi::Infer>();
    infer->id = ++action_id_seed;
    infer->model_id = local_batch_queue[index].get_model_id();
    infer->gpu_id = gpus[gpu_id].gpu_index;
    infer->batch_size = batch_size;
    infer->input_size = 0;
    infer->input = nullptr;
    infer->earliest = local_batch_queue[index].start_time;
    infer->latest = local_batch_queue[index].start_time + infer_slack;

    // update the gpu timings
    gpus[gpu_id].gpu_idle_at = local_batch_queue[index].finish_time;

    // update LRU model on the GPU
    unsigned model_id = infer->model_id;
    gpus[gpu_id].update_lru(infer->model_id);

    auto infer_action_complete =
        [this, batch_id, model_id,
         batch_size](std::shared_ptr<workerapi::Result> result) {
          if (auto infer_result =
                  std::dynamic_pointer_cast<workerapi::InferResult>(result)) {
            set_telemetry_infer_result(infer_result);
            set_estimates(model_id, batch_size, infer_result->exec.duration,
                          infer_result->gpu_clock);

            mtx_batch_storage.lock();
            std::vector<Request> requests = batch_storage[batch_id];
            mtx_batch_storage.unlock();

            for (auto &request_item : requests) {
              uint64_t request_id = request_item.id;
              // uint64_t user_request_id =
              // request_item.request.header.user_request_id;
              uint64_t user_request_id = request_item.user_request_id;

              mtx_inference_callbacks.lock();
              auto callback = inference_callbacks[request_id];
              inference_callbacks.erase(inference_callbacks.find(request_id));
              mtx_inference_callbacks.unlock();

              clientapi::InferenceResponse response;
              response.header.user_request_id = user_request_id;
              response.header.status = clockworkSuccess;
              response.header.message = "";
              response.output_size = 0;
              response.output = nullptr;
              response.model_id = model_id;
              response.batch_size = 1;
              callback(response);
            }

            mtx_batch_storage.lock();
            batch_storage.erase(batch_storage.find(batch_id));
            mtx_batch_storage.unlock();

            delete infer_result->output;
            infer_result->output = nullptr;

          } else {
            std::string error_message = "Internal Controller Error";

            if (auto error =
                    std::dynamic_pointer_cast<workerapi::ErrorResult>(result)) {
              error_message = error->message;
              set_telemetry_error_result(error);
            }

            mtx_batch_storage.lock();
            std::vector<Request> requests = batch_storage[batch_id];
            mtx_batch_storage.unlock();

            for (auto &request_item : requests) {
              uint64_t request_id = request_item.id;

              //   uint64_t user_request_id =
              //   request_item.request.header.user_request_id;
              uint64_t user_request_id = request_item.user_request_id;

              mtx_inference_callbacks.lock();
              auto callback = inference_callbacks[request_id];
              inference_callbacks.erase(inference_callbacks.find(request_id));
              mtx_inference_callbacks.unlock();

              clientapi::InferenceResponse response;
              response.header.user_request_id = user_request_id;
              response.header.status = clockworkError;
              response.header.message = error_message;
              response.output_size = 0;
              response.output = nullptr;
              response.model_id = model_id;
              callback(response);
            }

            mtx_batch_storage.lock();
            batch_storage.erase(batch_storage.find(batch_id));
            mtx_batch_storage.unlock();
          }
        };

    add_active_model(infer->id, gpu_id, model_id);
    mtx_action_callbacks.lock();
    action_callbacks[infer->id] = infer_action_complete;
    mtx_action_callbacks.unlock();

    set_telemetry_infer(gpus[gpu_id].worker_id, infer);

    actions.push_back(infer);
  }

  // STEP: send actions to the worker

  if (actions.size() > 0) {
    workers[gpus[gpu_id].worker_id]->sendActions(actions);
    local_batch_queue.erase(local_batch_queue.begin(),
                            local_batch_queue.begin() + index);
  }

  // STEP: remove the scheduled requests from the request queue
  for (auto req_id : drop_list) {
    for (unsigned idx = 0; idx < request_queue.size(); idx++) {
      if (request_queue[idx].id == req_id) {
        request_queue.erase(request_queue.begin() + idx);
      }
    }
  }
  drop_list.clear();
}

void SmartScheduler::decide_replication() {
  std::set<unsigned> queued_models;
  std::map<unsigned, unsigned> model_queued_load;  // model_id -> incoming_load

  // initial incoming load check
  for (auto &request_entry : request_queue) {
    unsigned model_id = request_entry.get_model_id();
    queued_models.insert(model_id);
    if (model_queued_load.find(model_id) == model_queued_load.end()) {
      model_queued_load[model_id] = 1;
    } else {
      model_queued_load[model_id]++;
    }
  }

  std::map<unsigned, unsigned>
      models_to_replicate;  // model_id -> num_new_replicas

  // TODO: this metric of drop counts per scheduling attempt doesn't seem to
  // be as good as expected, because there isn't enough dropped request per
  // model per each schedule to trigger the replication another idea: based on
  // the sum of drop_counts within the last N schedules (= like 10, 10 * 3ms =
  // 30ms, it would take us 30ms to detect a high load on a model?)
  for (auto &model_drop_count_item : model_drop_count) {
    unsigned model_id = model_drop_count_item.first;
    unsigned drop_count = model_drop_count_item.second;
    unsigned total_gpu_count = gpus.size();
    unsigned current_assigned_gpus_count = models[model_id].gpus.size();
    unsigned new_replicas_to_load =
        std::min<unsigned>(total_gpu_count - current_assigned_gpus_count,
                           drop_count / replication_sensitivity);  //
    if (new_replicas_to_load > 0) {
      // deal breaker: if the model is already being loaded on any gpu, don't
      // plan any further replication for it
      bool currently_loading = false;
      for (auto &weights_available_at : models[model_id].weights_available_at) {
        if (weights_available_at.second + wait_time_before_replication >
            util::now()) {  // + 10ms to wait before initiating any new
                            // replication
          currently_loading = true;
          break;
        }
      }
      if (currently_loading) {
        continue;
      }

      models_to_replicate[model_id] = new_replicas_to_load;
    }
  }

  std::map<unsigned, uint64_t>
      estimated_gpu_load;  // gpu_id -> estimated_idle_at

  // init the gpu load estimates
  for (auto &gpu_load_item : gpus) {
    unsigned gpu_id = gpu_load_item.first;
    estimated_gpu_load[gpu_id] = std::max<uint64_t>(
        gpus[gpu_id].gpu_idle_at, util::now() + network_transfer_latency);
  }

  std::map<unsigned, std::map<unsigned, std::vector<unsigned>>>
      replication_plan;  // gpu_id -> [ model_id, [victims] ]

  for (auto &model_to_replicate :
       models_to_replicate) {  // outer loop, iterating over models_to_load to
                               // find a gpu for each

    unsigned model_to_replicate_id = model_to_replicate.first;

    // calculate the estimated gpu loads
    for (auto &model_load_element :
         model_queued_load) {  // iterate all the models

      unsigned model_id = model_load_element.first;
      unsigned queued_load = model_load_element.second;

      for (auto &model_gpu_element : models[model_id].gpus) {
        unsigned gpu_id = model_gpu_element;
        estimated_gpu_load[gpu_id] +=
            (get_latency_estimate(model_id, 1) * queued_load /
             (models[model_id].gpus.size() +
              1));  // distribute the load on all the gpus serving
                    // the target model
      }
    }

    std::vector<std::pair<unsigned, uint64_t>> sorted_gpu_loads;
    for (auto &gpu_load_estimate_item : estimated_gpu_load) {
      sorted_gpu_loads.push_back(std::make_pair(gpu_load_estimate_item.first,
                                                gpu_load_estimate_item.second));
    }
    sort(sorted_gpu_loads.begin(), sorted_gpu_loads.end(), compare_values);

    for (auto &sorted_gpu_load_item : sorted_gpu_loads) {
      unsigned gpu_id = sorted_gpu_load_item.first;
      if (!models[model_to_replicate_id].is_loaded_on_gpu(gpu_id)) {
        // if the model is not loaded on this gpu, add it to the replication
        // plan
        if (replication_plan.find(gpu_id) == replication_plan.end()) {
          replication_plan[gpu_id] =
              std::map<unsigned, std::vector<unsigned>>();
        }
        replication_plan[gpu_id][model_to_replicate_id] =
            std::vector<unsigned>();

        // find the victims

        while (!gpus[gpu_id].fits_model(
            models[model_to_replicate_id].num_pages)) {  // evict until there's
                                                         // enough space
          // evict
          unsigned victim_model = gpus[gpu_id].lru_loaded_models.back();
          gpus[gpu_id].lru_loaded_models.pop_back();
          while (queued_models.find(victim_model) !=
                 queued_models.end()) {  // if the model is being requested put
                                         // it back to the lru list at the head
            gpus[gpu_id].lru_loaded_models.insert(
                gpus[gpu_id].lru_loaded_models.begin(), victim_model);
            victim_model = gpus[gpu_id].lru_loaded_models.back();
            gpus[gpu_id].lru_loaded_models.pop_back();
          }

          replication_plan[gpu_id][model_to_replicate_id].push_back(
              victim_model);

          gpus[gpu_id].available_pages += models[victim_model].num_pages;

          models[victim_model].remove_gpu(gpu_id);
          gpus[gpu_id].evict_model(victim_model);

          if (models[victim_model].gpus.size() ==
              0) {  // if there's no loaded instance in the entire system

            unset_global_cache_state(victim_model);
          }
        }
      }
    }
  }

  // -- by now we have the model gpu id -> model evict lists
  // STEP: make evict / load actions

  for (auto &gpu_pci_ops_item : replication_plan) {
    unsigned gpu_id = gpu_pci_ops_item.first;
    gpus[gpu_id].pci_idle_at = std::max<uint64_t>(
        gpus[gpu_id].pci_idle_at, util::now() + network_transfer_latency);

    for (auto &model_pci_ops_item : gpu_pci_ops_item.second) {
      unsigned model_id = model_pci_ops_item.first;

      for (auto &victim_model_id : model_pci_ops_item.second) {
        send_model_evict_action(gpu_id, victim_model_id);
      }

      send_model_load_action(gpu_id, model_id);
    }
  }
}

// --- Telemetry related functions
void SmartScheduler::set_telemetry_infer(
    unsigned worker_id, std::shared_ptr<workerapi::Infer> &action) {
  ControllerActionTelemetry *telemetry = new ControllerActionTelemetry;
  telemetry->worker_id = worker_id;
  telemetry->set(action);
  mtx_telemetry.lock();
  action_telemetry_map.insert(std::make_pair(action->id, telemetry));
  mtx_telemetry.unlock();
}

void SmartScheduler::set_telemetry_evict_weights(
    unsigned worker_id, std::shared_ptr<workerapi::EvictWeights> &action) {
  ControllerActionTelemetry *telemetry = new ControllerActionTelemetry;
  telemetry->worker_id = worker_id;
  telemetry->set(action);
  mtx_telemetry.lock();
  action_telemetry_map.insert(std::make_pair(action->id, telemetry));
  mtx_telemetry.unlock();
}

void SmartScheduler::set_telemetry_load_weights(
    unsigned worker_id, std::shared_ptr<workerapi::LoadWeights> &action) {
  ControllerActionTelemetry *telemetry = new ControllerActionTelemetry;
  telemetry->worker_id = worker_id;
  telemetry->set(action);
  mtx_telemetry.lock();
  action_telemetry_map.insert(std::make_pair(action->id, telemetry));
  mtx_telemetry.unlock();
}

void SmartScheduler::set_telemetry_infer_result(
    std::shared_ptr<workerapi::InferResult> &result) {
  mtx_telemetry.lock();
  auto it = action_telemetry_map.find(result->id);
  ControllerActionTelemetry *telemetry = it->second;
  action_telemetry_map.erase(it);
  mtx_telemetry.unlock();
  telemetry->set(result);
  logger->log(*telemetry);
  free(telemetry);
}

void SmartScheduler::set_telemetry_load_weights_result(
    std::shared_ptr<workerapi::LoadWeightsResult> &result) {
  mtx_telemetry.lock();
  auto it = action_telemetry_map.find(result->id);
  ControllerActionTelemetry *telemetry = it->second;
  action_telemetry_map.erase(it);
  mtx_telemetry.unlock();
  telemetry->set(result);
  logger->log(*telemetry);
  free(telemetry);
}

void SmartScheduler::set_telemetry_evict_weights_result(
    std::shared_ptr<workerapi::EvictWeightsResult> &result) {
  mtx_telemetry.lock();
  auto it = action_telemetry_map.find(result->id);
  ControllerActionTelemetry *telemetry = it->second;
  action_telemetry_map.erase(it);
  mtx_telemetry.unlock();
  telemetry->set(result);
  logger->log(*telemetry);
  free(telemetry);
}

void SmartScheduler::set_telemetry_error_result(
    std::shared_ptr<workerapi::ErrorResult> &result) {
  mtx_telemetry.lock();
  auto it = action_telemetry_map.find(result->id);
  ControllerActionTelemetry *telemetry = it->second;
  action_telemetry_map.erase(it);
  mtx_telemetry.unlock();
  telemetry->set(result);
  logger->log(*telemetry);
  free(telemetry);
}

}  // namespace clockwork
