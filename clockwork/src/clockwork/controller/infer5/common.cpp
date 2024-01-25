#include <algorithm>
#include "clockwork/controller/infer5/common.h"

namespace clockwork {
namespace scheduler {

Estimator::Estimator(int window_size, float percentile) : window(window_size), percentile(percentile) {}

void Estimator::update(uint64_t measurement) {
    tbb::spin_mutex::scoped_lock lock(update_mutex);

    window.insert(measurement);
    current = window.get_percentile(percentile);
}

ModelTracker::ModelTracker(std::vector<unsigned> batch_sizes, int window_size, float percentile) {
    load_estimator = new Estimator(window_size, percentile);

    for (auto batch_size : batch_sizes) {
        if (infer_estimators.size() < batch_size + 1) {
            infer_estimators.resize(batch_size + 1, nullptr);
        }
        infer_estimators[batch_size] = new Estimator(window_size, percentile);
    }
}

uint64_t ModelTracker::update_load(uint64_t load_time) {
    load_estimator->update(load_time);
}

uint64_t ModelTracker::update_infer(unsigned batch_size, uint64_t exec_time, uint64_t gpu_clock) {
    infer_estimators[batch_size]->update(exec_time * gpu_clock);
}

uint64_t ModelTracker::estimate_load() {
    return load_estimator->current;
}

uint64_t ModelTracker::estimate_infer(unsigned batch_size, uint64_t gpu_clock) {
    return infer_estimators[batch_size]->current;
}

RequestImpl::RequestImpl(
    clientapi::InferenceRequest request,
    std::function<void(clientapi::InferenceResponse&)> callback) : 
        request(request), 
        callback(callback),
        locked(false),
        response_sent(ATOMIC_FLAG_INIT) {
    // Set the response header fields now
    response.header.user_request_id = request.header.user_request_id;
    response.header.message = "";
    response.header.status = 0;
    response.model_id = request.model_id;
    response.batch_size = request.batch_size;
    response.output = nullptr;
    response.output_size = 0;
}

RequestImpl::~RequestImpl() {
    delete static_cast<char*>(request.input);
}

void RequestImpl::lock() {
    locked = true;
}

void RequestImpl::set_slo(uint64_t default_slo) {
    slo = default_slo;
    if (request.slo_factor > 0) {
        slo = model->b1_exec * request.slo_factor;
    }
    response.deadline = request.arrival + slo;

    exec_slo = std::min(slo, slo - Scheduler::buffer);
    exec_slo = std::max(exec_slo, scheduler->schedule_ahead + Scheduler::buffer);
    deadline = request.arrival + exec_slo;

    weights_slo = std::min(max_loadweights_slo, std::min(slo, slo - (model->estimate_weights() + model->b1_exec + Scheduler::buffer + scheduler->schedule_ahead)));
    weights_slo = std::max(weights_slo, scheduler->schedule_ahead + Scheduler::buffer);
}

void RequestImpl::set_result(char* output, size_t output_size) {
    response.header.status = clockworkSuccess;
    response.output = output;
    response.output_size = output_size;
    response.departure_count = model->copies_loaded;
}

void RequestImpl::set_error(int status, std::string message) {
    response.header.status = status;
    response.header.message = message;
}

bool RequestImpl::complete(uint64_t now, int gpu_id) {
    // if (print_debug) std::cout << ("Client <--  " + response.str() + "\n");

    // Here to ensure only one response is sent
    if (response_sent.test_and_set()) return false;

    // Set the departure time (controller.cpp can also do this, 
    // but we want to report departure time back to the action to determine goodput)
    response.departure = now;

    callback(response);

    return response.header.status == clockworkSuccess && response.departure <= response.deadline;
}

void RequestImpl::timeout() {
    if (locked) return;

    // if (print_debug) std::cout << ("Client <--  " + response.str() + "\n");

    // Here to ensure only one response is sent
    if (response_sent.test_and_set()) return;

    if (response.header.status == 0)
        response.header.status = clockworkTimeout;
    response.departure = util::now();
    response.departure_count = model->copies_loaded;

    callback(response);

    {
        tbb::queuing_mutex::scoped_lock lock(model->scheduler->tracker->mutex);
        model->scheduler->tracker->requestCancelled(demand);
    }
}

}
}