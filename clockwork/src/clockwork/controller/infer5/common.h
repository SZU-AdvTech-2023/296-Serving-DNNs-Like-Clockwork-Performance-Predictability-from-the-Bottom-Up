// Copyright 2020 Max Planck Institute for Software Systems

#ifndef SRC_CLOCKWORK_CONTROLLER_INFER5_COMMON_H_
#define SRC_CLOCKWORK_CONTROLLER_INFER5_COMMON_H_

#include <functional>
#include "tbb/spin_mutex.h"
#include "clockwork/sliding_window.h"
#include "clockwork/util.h"
#include "clockwork/api/client_api.h"
#include "clockwork/controller/load_tracker.h"

namespace clockwork {
namespace scheduler {

class Estimator {
 public:
    uint64_t current = 100000000UL;
    Estimator(int window_size, float percentile);
    void update(uint64_t measurement);

 private:
    float percentile;
    SlidingWindow window;
    tbb::spin_mutex update_mutex;
};

/* Tracks measurements of a model instance */
class ModelTracker {
 public:
    ModelTracker(std::vector<unsigned> batch_sizes,
                 int window_size=10, float percentile=0.99);

    uint64_t update_load(uint64_t load_time);
    uint64_t update_infer(unsigned batch_size, uint64_t exec_time, uint64_t gpu_clock);

    uint64_t estimate_load();
    uint64_t estimate_infer(unsigned batch_size, uint64_t gpu_clock);
 private:
    Estimator* load_estimator;
    std::vector<Estimator*> infer_estimators;

};

class ModelInstanceInfo {

};

class ModelInfo {
public:
    unsigned id;
    unsigned num_weights_pages;
    size_t input_size;
    size_t output_size;
    std::vector<ModelInstanceInfo*> instances;
    uint64_t b1_exec;
    std::atomic_int copies_loaded = 0;
    std::atomic_int requests_queued = 0;

 private:
    tbb::queuing_mutex mutex;

    std::vector<unsigned> supported_batch_sizes;
    std::vector<unsigned> batch_lookup_;
    unsigned max_batch_size;

    tbb::spin_mutex estimates_mutex;
    std::vector<uint64_t> estimates;
    std::map<unsigned, SlidingWindow*> estimators;

    tbb::spin_mutex weights_estimate_mutex;
    SlidingWindow* weights_estimator;
    uint64_t weights_estimate;

    std::atomic_uint64_t request_id_seed_ = 0;

    tbb::concurrent_queue<Request> incoming;
    std::deque<Request> queue;
};

class RequestImpl;
typedef std::shared_ptr<RequestImpl> Request;
class RequestImpl {
 public:
    uint64_t id;
    uint64_t slo;
    uint64_t exec_slo;
    uint64_t weights_slo;
    uint64_t deadline;
    clientapi::InferenceRequest request;
    clientapi::InferenceResponse response;

    LoadTracker::Demand demand;

 private:
    std::atomic_bool locked;
    std::atomic_flag response_sent;

    std::function<void(clientapi::InferenceResponse&)> callback;

 public:
    RequestImpl(clientapi::InferenceRequest request,
        std::function<void(clientapi::InferenceResponse&)> callback);
    ~RequestImpl();

    // TODO: set response.arrival_count elsewhere
    // void set_model(Model* model) {
    //     this->model = model;
    //     response.arrival_count = model->copies_loaded;
    // }
    void set_slo(uint64_t default_slo);
    void set_result(char* output, size_t output_size);
    void set_error(int status, std::string message);

    void lock();

    // Returns true if the result was successful and within the deadline
    void timeout();
    bool complete(uint64_t now, int gpu_id);

    struct DeadlineComparator {
        bool operator()(const Request &lhs, const Request &rhs) {
            return lhs->deadline > rhs->deadline;
        }
    };
};




}
}
#endif // SRC_CLOCKWORK_CONTROLLER_INFER5_COMMON_H_