// Copyright 2020 Max Planck Institute for Software Systems

#ifndef SRC_CLOCKWORK_CONTROLLER_INFER5_INFER5_SCHEDULER_H_
#define SRC_CLOCKWORK_CONTROLLER_INFER5_INFER5_SCHEDULER_H_

#include <atomic>
#include <algorithm>
#include <string>
#include <sstream>
#include <set>
#include "clockwork/controller/scheduler.h"
#include "clockwork/controller/worker_tracker.h"
#include "clockwork/controller/infer5/load_tracker.h"
#include "clockwork/telemetry/controller_action_logger.h"
#include "clockwork/thread.h"
#include "clockwork/api/worker_api.h"
#include "clockwork/sliding_window.h"
#include "tbb/mutex.h"
#include "tbb/queuing_mutex.h"

namespace clockwork {
namespace scheduler {
namespace infer5 {

class Scheduler : public clockwork::Scheduler {
 public:

    static const uint64_t print_interval = 10000000000UL;
    static const bool print_debug = false;
    static const bool print_loads = false;
    static const bool print_scheduler_stats = false;

    // Non-configurable parameters
    static const uint64_t default_clock = 1380; // default gpu clock speed
    static const uint64_t buffer = 5000000UL; // Aim for an SLO this much prior to actual SLO
    static const int estimate_window_size = 10; // Estimate execution time using last 10 measurements
    static const float estimate_percentile; // Percentile to use for estimation; 0.99 (effectively max)
    static const uint64_t lag = 10000000UL; // how much can worker lag behind expected completion time before we stop scheduling
    static const uint64_t future = 1000000UL; // used for setting earliest timestamp; expect 1ms lag getting to worker
    static const uint64_t max_loadweights_slo = 25000000UL;
    static const unsigned network_concurrency = 2; // max number of concurrent network xfers

    // Scheduler parameters configurable by ./controller binary

    const uint64_t default_slo;
    const uint64_t latest_delta; // Actions can run up to 10ms behind schedule before the worker will drop them
    const uint64_t schedule_ahead; // schedule 10ms into the future
    const uint64_t max_allowable_exec_time; // disallow batches with execution times greater than this
    const unsigned max_batch_size;
    const bool generate_inputs; // if clients send 0-size inputs, do we want to generate real ones, or send 0-size?
    const int max_gpus; // max number of gpus to use

    Scheduler(
        uint64_t default_slo, // 100ms
        uint64_t latest_delta, // 10ms
        uint64_t schedule_ahead, // 10ms
        bool generate_inputs, // if clients send no input, should we generate real inputs, or forward the size-0?
        int max_gpus, // max GPUs to use
        uint64_t max_allowable_exec_time, // don't use batch sizes with higher exec time than this
        unsigned max_batch_size, // max allowed batch size
        std::string actions_filename);

    class RequestImpl;
    typedef std::shared_ptr<RequestImpl> Request;

    class Model;
    class RequestImpl {
     public:
        Scheduler* scheduler;
        uint64_t id;
        uint64_t seqno;
        uint64_t slo;
        uint64_t exec_slo;
        uint64_t weights_slo;
        uint64_t deadline;
        Model* model = nullptr;
        clientapi::InferenceRequest request;
        clientapi::InferenceResponse response;

        LoadTracker::Demand demand;

     private:
        std::atomic_bool locked;
        std::atomic_flag response_sent;

        std::function<void(clientapi::InferenceResponse&)> callback;

     public:
        RequestImpl(Scheduler* scheduler,
            clientapi::InferenceRequest request,
            std::function<void(clientapi::InferenceResponse&)> callback);
        ~RequestImpl();

        void set_model(Model* model);
        void set_slo(uint64_t default_slo);
        void set_result(char* output, size_t output_size);
        void set_error(int status, std::string message);

        void lock();

        // Returns true if the result was successful and within the deadline
        void timeout();
        bool complete(uint64_t now, int gpu_id);
        void finalize();

        struct DeadlineComparator {
            bool operator()(const Request &lhs, const Request &rhs) {
                return lhs->deadline > rhs->deadline;
            }
        };
    };

    class InferAction {
    private:
        Scheduler* scheduler;
        bool generated_inputs = false;
     public:
        Model* model;
        ControllerActionTelemetry telemetry;
        std::shared_ptr<workerapi::Infer> action = std::make_shared<workerapi::Infer>();
        std::shared_ptr<workerapi::ErrorResult> error = nullptr;
        std::shared_ptr<workerapi::InferResult> result = nullptr;
        std::vector<Request> requests;
        uint64_t send_by;
        uint64_t report_error_at;

        explicit InferAction(Scheduler* scheduler, Model* model);
        ~InferAction();

        void batch();
        void unbatch();
        void set_expectations(uint64_t exec_start, uint64_t duration, int clock);
        void set_error(std::shared_ptr<workerapi::ErrorResult> &error);
        void set_result(std::shared_ptr<workerapi::InferResult> &result);

        // Returns the fraction of successful requests
        float complete(uint64_t now, int gpu_id);
    };

    class ModelInstance;
    class LoadWeightsAction {
     public:
        Scheduler* scheduler;
        ModelInstance* instance;
        unsigned version;
        ControllerActionTelemetry telemetry;
        Model* model;
        std::shared_ptr<workerapi::LoadWeights> action = std::make_shared<workerapi::LoadWeights>();
        std::shared_ptr<workerapi::ErrorResult> error = nullptr;
        std::shared_ptr<workerapi::LoadWeightsResult> result = nullptr;

        explicit LoadWeightsAction(Scheduler* scheduler, ModelInstance* instance);

        void set_expectations(uint64_t exec_start, uint64_t duration);
        void set_error(std::shared_ptr<workerapi::ErrorResult> &error);
        void set_result(std::shared_ptr<workerapi::LoadWeightsResult> &result);

    };
    class EvictWeightsAction {
     public:
        ModelInstance* instance;
        ControllerActionTelemetry telemetry;
        std::shared_ptr<workerapi::EvictWeights> action = std::make_shared<workerapi::EvictWeights>();
        std::shared_ptr<workerapi::ErrorResult> error = nullptr;
        std::shared_ptr<workerapi::EvictWeightsResult> result = nullptr;

        explicit EvictWeightsAction(ModelInstance* instance);

        void set_expectations();
        void set_error(std::shared_ptr<workerapi::ErrorResult> &error);
        void set_result(std::shared_ptr<workerapi::EvictWeightsResult> &result);

    };

    class GPU;
    class ModelInstance;

    class StrategyImpl {
    public:
        uint64_t priority;
        unsigned batch_size;

        ModelInstance* instance;

        struct Comparator {
            bool operator()(const StrategyImpl &lhs, const StrategyImpl &rhs) const {
                if (lhs.priority == rhs.priority) {
                    if (lhs.batch_size == rhs.batch_size) {
                        return lhs.instance->model->id < rhs.instance->model->id;
                    } else {
                        return lhs.batch_size < rhs.batch_size;
                    }
                } else {
                    return lhs.priority < rhs.priority;
                }
            }
        };

        std::string str() {
            std::stringstream ss;
            ss << "S m=" << instance->model->id << " p=" << priority << "  b=" << batch_size;
            return ss.str();
        }

        StrategyImpl() {}

    };

    class ModelQueue {
     private:
        std::deque<Request> queue;

     public:

        const int batchsize;

        ModelQueue(int batchsize) : batchsize(batchsize) {}

        Request& front() {
            return queue.front();
        }

        void pop() {
            queue.pop_front();
        }

        int size() {
            return queue.size();
        }

        bool has_demand() {
            return size() >= batchsize;
        }

        void push(Request &request) {
            queue.push_back(request);
        }

    };

    class Model {
     public:
        uint64_t seqno_seed = 0;
        unsigned id;
        Scheduler* scheduler;
        unsigned num_weights_pages;
        size_t input_size;
        size_t output_size;
        std::vector<ModelInstance*> instances;
        uint64_t b1_exec;
        std::atomic_int copies_loaded = 0;
        std::atomic_int requests_queued = 0;

        ModelLoadTracker* tracker;
        std::atomic_flag stale;

        void invalidate_tracker() {
            if (!stale.test_and_set()) {
                scheduler->stale.push(this);
            }
        }

        void reset_tracker() {
            stale.clear();
        }

     private:
        tbb::queuing_mutex mutex;

        std::vector<unsigned> supported_batch_sizes;
        std::vector<unsigned> batch_lookup_;
        std::vector<unsigned> index_lookup_;
        unsigned max_batch_size;

        tbb::queuing_mutex estimates_mutex;
        std::vector<uint64_t> estimates;
        std::map<unsigned, SlidingWindow*> estimators;

        tbb::queuing_mutex weights_estimate_mutex;
        SlidingWindow* weights_estimator;
        uint64_t weights_estimate;

        std::atomic_uint64_t request_id_seed_ = 0;

        tbb::concurrent_queue<Request> incoming_requests;
        std::vector<ModelQueue*> queues;


     public:

        Model(Scheduler* scheduler, BatchedModelState &state);

        uint64_t estimate_weights();

        // Enqueues the request to this model, then enqueues InferStrategies to all active ModelInstances
        void enqueue(Request request);

        std::string queues_str() {
            std::stringstream msg;
            bool first = true;
            for (auto &queue : queues) {
                if (!first) msg << " ";
                first = false;
                msg << "q" << queue->batchsize << "=" << queue->size();
            }
            return msg.str();
        }

    private:
        void pull_incoming_requests();

    public:
        std::vector<StrategyImpl> new_strategies(int gpu_id, unsigned gpu_clock, int max_batchsize);

        // Gets actions to execute for this model
        InferAction* try_dequeue(uint64_t gpu_free_at, unsigned gpu_clock, int min_batchsize);

        // GPUs can add new measurements
        void add_measurement(unsigned batch_size, uint64_t duration, unsigned gpu_clock);
        void add_weights_measurement(uint64_t duration);
        uint64_t estimate(unsigned batch_size);

     private:

        // For num_requests requests, what is the maximum batch size we could execute?
        unsigned batch_lookup(unsigned num_requests);
        unsigned index_lookup(unsigned batchsize);

        void check_timeouts(uint64_t free_at);
        uint64_t estimate(unsigned batch_size, int clock);
    };

    class ModelInstance {
     public:
        GPU* gpu = nullptr;
        Model* model = nullptr;
        std::atomic_bool loaded;
        std::atomic_bool loading;
        std::atomic_int version = 0;
        std::vector<StrategyImpl> strategies;
        std::atomic_flag active;
        ModelInstance(GPU* gpu, Model* model): gpu(gpu), model(model), loaded(false), loading(false), active(ATOMIC_FLAG_INIT) {}

        void activate();
        void deactivate();
    };

    class GPU {
     public:
        unsigned id; // a unique id for this (worker_id, gpu_id)
        unsigned worker_id; // the id of the worker
        unsigned gpu_id; // the id of the gpu on the worker
        unsigned pages; // the number of pages on the gpu
        std::vector<ModelInstance*> instances;

        tbb::concurrent_queue<ModelInstance*> activated;

        std::atomic_uint64_t schedule_infer_count = 0;
        std::atomic_uint64_t schedule_infer_active_count = 0;
        std::atomic_uint64_t schedule_infer_inactive_count = 0;
        std::atomic_uint64_t schedule_infer_strategies_empty_count = 0;
        std::atomic_uint64_t schedule_infer_exec_full = 0;
        std::atomic_uint64_t schedule_infer_action_created = 0;
        std::atomic_uint64_t schedule_infer_action_attempted = 0;

        std::string stats() {
            std::stringstream s;
            s << "GPU-" << id << "-INF ";
            s << schedule_infer_active_count.exchange(0) << "/" << schedule_infer_count.exchange(0);
            s << " active, ";
            s << schedule_infer_strategies_empty_count.exchange(0) << " empty, ";
            s << schedule_infer_exec_full.exchange(0) << " execfull, ";
            s << schedule_infer_action_created.exchange(0) << "/";
            s << schedule_infer_action_attempted.exchange(0) << " created.";
            return s.str();
        }

     private:
        tbb::queuing_mutex infer_mutex;
        tbb::queuing_mutex load_mutex;

        network::controller::WorkerConnection* worker;
        Scheduler* scheduler;

        tbb::queuing_mutex exec_mutex;
        WorkerTracker exec;

        tbb::queuing_mutex loadweights_mutex;
        WorkerTracker loadweights;

        std::atomic_int free_pages;
        bool eviction_required = false;
        uint64_t last_print = 0;

        std::set<StrategyImpl, StrategyImpl::Comparator> strategies;

    public:
        GPU(unsigned id,
            Scheduler* scheduler, 
            network::controller::WorkerConnection* worker,
            unsigned worker_id,
            unsigned gpu_id,
            unsigned pages);

        bool schedule_infer();
        bool schedule_load();

    private:
        void send_action(InferAction* action);
        void send_action(LoadWeightsAction* action);
        void send_action(EvictWeightsAction* action);

        void add_model_strategies(ModelInstance* instance, int max_batchsize=INT_MAX);

        std::vector<EvictWeightsAction*> evict_pages(unsigned required_pages);

        void infer_error(InferAction* action, std::shared_ptr<workerapi::ErrorResult> &error);
        void infer_success(InferAction* action, std::shared_ptr<workerapi::InferResult> &result);
        void infer_result(InferAction* action, std::shared_ptr<workerapi::Result> &result);
        void load_error(LoadWeightsAction* action, std::shared_ptr<workerapi::ErrorResult> &error);
        void load_success(LoadWeightsAction* action, std::shared_ptr<workerapi::LoadWeightsResult> &result);
        void load_result(LoadWeightsAction* action, std::shared_ptr<workerapi::Result> &result);
        void evict_error(EvictWeightsAction* action, std::shared_ptr<workerapi::ErrorResult> &result);
        void evict_success(EvictWeightsAction* action, std::shared_ptr<workerapi::EvictWeightsResult> &result);
        void evict_result(EvictWeightsAction* action, std::shared_ptr<workerapi::Result> &result);
    };

    class NetworkExecutor {
     private:

        struct NetworkAction {
            network::controller::WorkerConnection* worker;
            std::shared_ptr<workerapi::Action> action;
            uint64_t start_send_by;
            uint64_t send_error_at;
        };

        tbb::queuing_mutex mutex;
        unsigned idle;
        std::deque<NetworkAction> pending;
        std::function<void(uint64_t, std::shared_ptr<workerapi::Result>)> error_callback;

     public:
        NetworkExecutor(unsigned concurrency, 
            std::function<void(uint64_t, std::shared_ptr<workerapi::Result>)> error_callback);

        void send(network::controller::WorkerConnection* worker, 
                  std::shared_ptr<workerapi::Action> action,
                  uint64_t start_send_by,
                  uint64_t send_error_at);
        void sendComplete();

    private:

        bool next(NetworkAction &toSend);

    };

 public:

    // Track the load on each model, used for LoadWeights/EvictWeights
    LoadTracker* tracker;
    tbb::concurrent_queue<Model*> stale;

    // Non-mutable so thread-safe
    std::vector<GPU*> gpus;
    std::vector<Model*> models;
    tbb::concurrent_queue<GPU*> to_load;
    tbb::concurrent_queue<GPU*> to_infer;
    std::atomic_uint64_t next_load = 0;
    std::atomic_uint64_t next_infer = 0;
    std::atomic_uint64_t request_count = 0;

 private:
    // Threads
    std::string actions_filename;
    ControllerActionTelemetryLogger* printer;
    std::thread network_printer;
    std::thread stats_printer;
    std::vector<std::thread> admission_threads;
    std::vector<std::thread> results_threads;
    std::vector<std::thread> infer_threads;
    std::vector<std::thread> load_threads;
    std::vector<std::thread> tracker_threads;

    // Network executor
    NetworkExecutor* network = nullptr;

    // Messages
    struct TimeoutResult {
        uint64_t timeout_at;
        std::shared_ptr<workerapi::Result> result;
    };

    tbb::concurrent_queue<std::shared_ptr<workerapi::Result>> result_queue;
    tbb::concurrent_queue<TimeoutResult> network_timeout_queue;
    tbb::concurrent_queue<Request> request_queue;

    // Callbacks
    tbb::queuing_mutex callbacks_mutex;
    typedef std::function<void(std::shared_ptr<workerapi::Result>&)> Callback;
    std::unordered_map<uint64_t, Callback> callbacks;

    // Diagnostic
    std::atomic_flag has_logged_inputs_status;

    // Used during experiments if we are generating inputs server-side
    util::InputGenerator* input_generator = nullptr;


 public:

    // Called by GPU threads to register an action
    void add_callback(uint64_t action_id, Callback callback);

    // Called when model loading has completed
    virtual void start(std::vector<network::controller::WorkerConnection*> workers,
                        ClockworkState &state);

    // The actual scheduler interface implementation, invoked by client network thread
    virtual void clientInfer(clientapi::InferenceRequest &request, 
        std::function<void(clientapi::InferenceResponse&)> callback);

    // The actual scheduler interface implementation, invoked by worker network thread
    virtual void resultFromWorker(std::shared_ptr<workerapi::Result> result);

 private:

    // Initialization methods
    void validate_clockwork_state(ClockworkState &state);
    void initialize_models(ClockworkState &state);
    void initialize_gpus(std::vector<network::controller::WorkerConnection*> workers,
                    ClockworkState &state);
    void initialize_model_instances();
    void initialize_network(std::vector<network::controller::WorkerConnection*> workers);
    void print_status();

    // The main thread run methods
    void run_admission_thread();
    void run_tracker_thread();
    void run_results_thread();
    void run_infer_thread(int id);
    void run_load_thread(int id);
    void run_gpu_stats_printer_thread();

    // Logic of the dispatcher thread
    void handle_result(std::shared_ptr<workerapi::Result> &result);
    void handle_request(Request &request);
};

}
}
}

#endif // SRC_CLOCKWORK_CONTROLLER_INFER5_INFER5_SCHEDULER_H_