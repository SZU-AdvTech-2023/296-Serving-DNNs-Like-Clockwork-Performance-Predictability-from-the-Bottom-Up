#ifndef _CLOCKWORK_SMART_SCHEDULER_H_
#define _CLOCKWORK_SMART_SCHEDULER_H_

#include <string>
#include "clockwork/controller/scheduler.h"
#include "clockwork/telemetry/controller_action_logger.h"
#include "clockwork/sliding_window.h"

namespace clockwork {

class SmartScheduler : public Scheduler {
 public:
  static const uint64_t system_wide_slo = 100000000ULL;
  static const unsigned latency_profiling_window = 100;
  static const uint64_t scheduling_epoch = 1000000ULL;
  static const uint64_t schedule_ahead_length = 5000000ULL;
  static const uint64_t network_transfer_latency = 1000000ULL;
  static const uint64_t pci_slack = 1000000ULL;
  static const uint64_t infer_slack = 500000ULL;
  static const uint64_t weights_evict_latency = 1000000ULL;
  // it means we must be observing at least 50 drops to trigger replication
  static const unsigned replication_sensitivity = 50;
  // there must be at least 100ms between replication attempts
  static const uint64_t wait_time_before_replication = 100000000ULL;
  static const unsigned clk_freq_default = 1380UL;  // MHz
  static const unsigned gpu_cache_pages = 1408UL;
  static const unsigned max_allowed_batch_exec_time = 20000000;

  class ExecutionProfiler;
  class WeightsProfiler;
  class Request;
  class RequestBatch;
  class GPU;

  class ExecutionProfiler {
   public:
    float percentile;
    unsigned window_size;
    unsigned clk_freq;

    std::vector<unsigned> batch_sizes;
    std::map<unsigned, uint64_t> estimates;  // store latency * freq
    std::map<unsigned, SlidingWindow>
        sliding_windows;  // track latency * freq

    void set_batch_sizes(std::vector<unsigned> &sizes);
    void set_estimates(std::map<unsigned, uint64_t> latencies);
    void set_window_size(unsigned size);
    uint64_t get_latency_estimate(unsigned batch_size);
    unsigned get_max_batch_size(uint64_t slack, unsigned limit);
    void insert(unsigned batch, uint64_t latency, unsigned freq);
    void update_estimate(unsigned batch);
    void update_all_estimates();

    ExecutionProfiler()
        : percentile(99), window_size(100), clk_freq(clk_freq_default) {}
  };

  class WeightsProfiler {
   public:
    uint64_t estimate;  // store the estimate in ns
    uint64_t get_estimate();
    WeightsProfiler(){};
    WeightsProfiler(uint64_t estimate);
  };

  class Model {
   public:
    unsigned id;
    std::map<unsigned, uint64_t> weights_available_at;  // gpu_id -> timestamp
    std::vector<unsigned> gpus;
    unsigned num_pages;
    Model() {}
    Model(unsigned id, unsigned num_pages);

    bool available_on(unsigned gpu_id);
    uint64_t earliest(unsigned gpu_id);
    bool is_loaded_on_gpu(unsigned gpu_id);
    void add_gpu(unsigned gpu_id);
    void remove_gpu(unsigned gpu_id);
  };

  class Request {
   public:
    uint64_t id;
    // clientapi::InferenceRequest request;
	uint64_t user_request_id;
	unsigned model_id;
	unsigned batch_size;
    uint64_t arrived;
    uint64_t deadline;
    uint64_t earliest;
    uint64_t start_time;
    uint64_t finish_time;
    // Request() {}
    // Request(uint64_t id, uint64_t arrived, uint64_t deadline,
    //         clientapi::InferenceRequest request);
	Request(uint64_t id, uint64_t user_request_id, unsigned model_id, uint64_t arrived, uint64_t deadline);

    unsigned get_model_id();

    // ~Request();
  };

  class RequestBatch {
   public:
    uint64_t id;
    unsigned model_id;
    unsigned batch_size;
    std::vector<Request> requests;
    uint64_t earliest;
    uint64_t deadline;
    uint64_t start_time;
    uint64_t finish_time;
    RequestBatch(uint64_t id, unsigned model_id, unsigned batch_size)
        : id(id),
          model_id(model_id),
          batch_size(batch_size),
          earliest(0),
          deadline(0),
          start_time(0),
          finish_time(0) {}

    RequestBatch(uint64_t id, unsigned model_id, Request request);
    void add_to_batch(Request request);
	unsigned get_model_id();
  };

  class GPU {
   public:
    unsigned id;
    unsigned worker_id;
    unsigned gpu_index;
    uint64_t gpu_idle_at;
    uint64_t pci_idle_at;
    uint64_t available_pages;
    uint64_t total_pages;
    std::set<unsigned> loaded_models;
    std::vector<unsigned> lru_loaded_models;

    GPU() {
      gpu_idle_at = 0;
      pci_idle_at = 0;
    }
    GPU(unsigned id, unsigned worker_id, unsigned gpu_index,
        unsigned total_pages)
        : id(id),
          worker_id(worker_id),
          gpu_index(gpu_index),
          gpu_idle_at(0),
          pci_idle_at(0),
          total_pages(total_pages),
          available_pages(total_pages) {}

    bool fits_model(unsigned model_num_pages);
    void add_model(unsigned model_id);
    void update_lru(unsigned model_id);
    void evict_model(unsigned model_id);
  };

  // telemetry
  ControllerActionTelemetryLogger *logger = nullptr;
  std::map<unsigned, ControllerActionTelemetry *> action_telemetry_map;
  std::mutex mtx_telemetry;

  // Profilers
  std::map<unsigned, ExecutionProfiler>
      execution_profiler;  // model_id -> execution_profiler
  std::map<unsigned, WeightsProfiler>
      weights_profiler;  // model_id -> weights_profiler
  std::mutex mtx_profiler;

  // slo
  uint64_t default_slo;
  unsigned max_gpus;
  uint64_t max_exec_time;
  unsigned max_batch_size;
  std::string action_telemetry_file;

  // Seeds
  std::atomic_uint64_t action_id_seed;
  std::atomic_uint64_t global_request_id;
  std::atomic_uint64_t global_batch_id_seed;

  // workers and gpus
  std::vector<network::controller::WorkerConnection *> workers;
  std::map<unsigned, GPU> gpus;  // gpu_id -> gpu

  // models
  std::map<unsigned, Model> models;

  // if a model is hot or not
  std::set<unsigned>
      global_cached_models;  // models that are cached, at least on one gpu

  // drop count of each model, at each scheduling phase
  std::map<unsigned, unsigned> model_drop_count;

  // global request_queue
  std::vector<Request> request_queue;
  std::mutex mtx_request_queue;

  std::map<uint64_t, std::vector<Request>> batch_storage;
  std::mutex mtx_batch_storage;

  // local request queue per gpu
  std::map<unsigned, std::vector<Request>> gpu_request_queue;

  // action callbacks
  std::map<uint64_t, std::function<void(std::shared_ptr<workerapi::Result>)>>
      action_callbacks;
  std::mutex mtx_action_callbacks;

  // client inference callbacks
  std::map<uint64_t, std::function<void(clientapi::InferenceResponse &)>>
      inference_callbacks;
  std::mutex mtx_inference_callbacks;

  std::map<uint64_t, char *> infer_input_pointer;

  // active models
  std::map<uint64_t, std::pair<unsigned, unsigned>>
      active_models;  // action -> gpu_id, model_id
  std::mutex mtx_active_models;

  // scheduler thread
  std::thread scheduler_thread;

  SmartScheduler(uint64_t default_slo, unsigned max_gpus, uint64_t max_exec_time, unsigned max_batch_size, std::string action_telemetry_file);

  void add_active_model(uint64_t action_id, unsigned gpu_id, unsigned model_id);
  void remove_active_model(uint64_t action_id);
  bool is_model_active(unsigned gpu_id, unsigned model_id);

  bool is_model_hot_somewhere(unsigned model_id);
  void unset_global_cache_state(unsigned model_id);
  void set_global_cache_state(unsigned model_id);

  unsigned get_best_batchsize(unsigned model_id, unsigned queue_size);

  void get_models_to_load();
  void do_schedule();
  void assign_requests_to_gpu_local_queues();
  void gpu_local_schedule(unsigned gpu_id,
                          std::vector<Request> &local_request_queue);
  void gpu_local_batch_schedule(unsigned gpu_id,
                                std::vector<Request> &local_request_queue);

  void decide_replication();
  void save_inference_callback(
      uint64_t request_id,
      std::function<void(clientapi::InferenceResponse &)> callback);
  void save_action_callback(
      uint64_t request_id,
      std::function<void(std::shared_ptr<workerapi::Result>)> callback);

  void init_estimates(unsigned model_id, BatchedModelState &state);
  void set_estimates(unsigned model_id, unsigned batch_size,
                     uint64_t exec_latency, unsigned freq);
  uint64_t get_latency_estimate(unsigned model_id, unsigned batch_size);
  uint64_t get_weights_load_estimate(unsigned model_id);

  void start(std::vector<network::controller::WorkerConnection *> workers,
             ClockworkState &state);

  bool send_model_load_action(unsigned gpu_id, unsigned model_id);
  bool send_model_evict_action(unsigned gpu_id, unsigned victim_model_id);
  void drop_request(Request &request);

  void clientInfer(
      clientapi::InferenceRequest &request,
      std::function<void(clientapi::InferenceResponse &)> callback);

  void resultFromWorker(std::shared_ptr<workerapi::Result> result);

  void add_request_to_gpu_request_queue(unsigned gpu_id, Request request);
  void get_models_to_preload(std::vector<Request> &request_queue,
                             std::set<unsigned> &queued_request_models,
                             std::map<unsigned, unsigned> &models_incoming_load,
                             std::map<unsigned, unsigned> &models_to_load);
  void calculate_gpu_load_estimates(
      std::map<unsigned, uint64_t> &estimated_gpu_load,
      std::map<unsigned, unsigned> &models_incoming_load,
      std::map<unsigned, unsigned> &models_to_load);

  unsigned find_least_loaded_gpu(
      std::map<unsigned, uint64_t> &estimated_gpu_load);

  void calculate_model_load_evict_plan(
      std::map<unsigned, unsigned> &models_to_load,
      std::set<unsigned> &queued_request_models,
      std::map<unsigned, std::map<unsigned, std::vector<unsigned>>>
          &model_load_evict_plan);

  static bool request_finish_time_compare(Request &lhs, Request &rhs);
  static bool batch_finish_time_compare(RequestBatch &lhs, RequestBatch &rhs);

  static bool compare_values(const std::pair<unsigned, uint64_t> &a,
                             const std::pair<unsigned, uint64_t> &b);
  static bool compare_values_decreasing(const std::pair<unsigned, unsigned> &a,
                                        const std::pair<unsigned, unsigned> &b);

  void set_telemetry_infer(unsigned worker_id,
                           std::shared_ptr<workerapi::Infer> &action);
  void set_telemetry_evict_weights(
      unsigned worker_id, std::shared_ptr<workerapi::EvictWeights> &action);
  void set_telemetry_load_weights(
      unsigned worker_id, std::shared_ptr<workerapi::LoadWeights> &action);

  void set_telemetry_infer_result(
      std::shared_ptr<workerapi::InferResult> &result);
  void set_telemetry_evict_weights_result(
      std::shared_ptr<workerapi::EvictWeightsResult> &result);
  void set_telemetry_load_weights_result(
      std::shared_ptr<workerapi::LoadWeightsResult> &result);
  void set_telemetry_error_result(
      std::shared_ptr<workerapi::ErrorResult> &result);
};  // namespace clockwork

}  // namespace clockwork

#endif
