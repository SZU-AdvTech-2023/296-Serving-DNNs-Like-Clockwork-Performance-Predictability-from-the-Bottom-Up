#include "clockwork/controller/concurrent_infer_and_load_scheduler.h"
#include <vector>

namespace clockwork {
namespace scheduler {
namespace infer4 {

std::atomic_uint64_t action_id_seed = 0;


Scheduler::Scheduler(uint64_t default_slo, uint64_t latest_delta,
                     uint64_t schedule_ahead, 
                     bool generate_inputs, int max_gpus,
                     uint64_t max_allowable_exec_time, unsigned max_batch_size,
                     std::string actions_filename)
    : default_slo(default_slo),
      latest_delta(latest_delta),
      schedule_ahead(schedule_ahead),
      max_allowable_exec_time(max_allowable_exec_time),
      max_batch_size(max_batch_size),
      generate_inputs(generate_inputs),
      max_gpus(max_gpus),
      actions_filename(actions_filename),
      has_logged_inputs_status(ATOMIC_FLAG_INIT) {
    std::cout << "ConcurrentInferAndLoadScheduler using:" << std::endl;
    std::cout << "\t default_slo=" << default_slo << std::endl;
    std::cout << "\t latest_delta=" << latest_delta << std::endl;
    std::cout << "\t schedule_ahead=" << schedule_ahead << std::endl;
    std::cout << "\t max_allowable_exec_time=" << max_allowable_exec_time << std::endl;
    std::cout << "\t max_batch_size=" << max_batch_size << std::endl;
    std::cout << "\t generate_inputs=" << generate_inputs << std::endl;
    std::cout << "\t max_gpus=" << max_gpus << std::endl;

    if (generate_inputs) {
        input_generator = new util::InputGenerator();
    }
}

Scheduler::RequestImpl::RequestImpl(
    Scheduler* scheduler,
    clientapi::InferenceRequest request,
    std::function<void(clientapi::InferenceResponse&)> callback) : 
        scheduler(scheduler),
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

Scheduler::RequestImpl::~RequestImpl() {
    delete static_cast<char*>(request.input);
}

void Scheduler::RequestImpl::lock() {
    locked = true;
}

void Scheduler::RequestImpl::set_model(Model* model) {
    this->model = model;
    response.arrival_count = model->copies_loaded;
}

void Scheduler::RequestImpl::set_slo(uint64_t default_slo) {
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

void Scheduler::RequestImpl::set_result(char* output, size_t output_size) {
    response.header.status = clockworkSuccess;
    response.output = output;
    response.output_size = output_size;
    response.departure_count = model->copies_loaded;
}

void Scheduler::RequestImpl::set_error(int status, std::string message) {
    response.header.status = status;
    response.header.message = message;
}

bool Scheduler::RequestImpl::complete(uint64_t now, int gpu_id) {
    if (print_debug) std::cout << ("Client <--  " + response.str() + "\n");

    // Here to ensure only one response is sent
    if (response_sent.test_and_set()) return false;

    // Set the departure time (controller.cpp can also do this, 
    // but we want to report departure time back to the action to determine goodput)
    response.departure = now;

    callback(response);

    return response.header.status == clockworkSuccess && response.departure <= response.deadline;
}

void Scheduler::RequestImpl::timeout() {
    if (locked) return;

    if (print_debug) std::cout << ("Client <--  " + response.str() + "\n");

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

void Scheduler::RequestImpl::finalize() {
    timeout();
}

unsigned Scheduler::Model::batch_lookup(unsigned num_requests) {
    return num_requests > max_batch_size ? max_batch_size : batch_lookup_[num_requests];
}

Scheduler::Model::Model(Scheduler* scheduler, BatchedModelState &state)
    : scheduler(scheduler),
      id(state.id), 
      num_weights_pages(state.num_weights_pages),
      input_size(state.input_size),
      output_size(state.output_size) {
    for (auto &batch_size : state.supported_batch_sizes) {

        uint64_t estimate = 100000UL; // Default 0.1ms estimate

        // Lookup real estimate if it was provided
        auto it = state.exec_duration.find(batch_size);
        if (it != state.exec_duration.end()) {
            estimate = it->second;
        }

        // Limit the batch sizes we use
        if (batch_size == 1 || 
            (estimate > 0 
                && estimate <= scheduler->max_allowable_exec_time
                && batch_size <= scheduler->max_batch_size)) {
            if (estimates.size() < batch_size + 1) {
                estimates.resize(batch_size+1, 100000UL * Scheduler::default_clock);
            }
            estimates[batch_size] = estimate * Scheduler::default_clock;
            estimators[batch_size] = new SlidingWindow(Scheduler::estimate_window_size);
            supported_batch_sizes.push_back(batch_size);
        } else {
            std::cout << "Excluding b" << batch_size << " with estimate " << estimate << "model=" << state.model_path << std::endl;
        }
    }

    weights_estimator = new SlidingWindow(Scheduler::estimate_window_size);
    weights_estimate = state.weights_transfer_duration;

    batch_lookup_ = util::make_batch_lookup(supported_batch_sizes);
    max_batch_size = batch_lookup_.size() - 1;

    b1_exec = estimate(1); // Use this for slo_factor
}

void Scheduler::Model::enqueue(Request request) {
    std::vector<uint64_t> batch_size_estimates(supported_batch_sizes.size());
    // Get batch size info
    for (unsigned i = 0; i < supported_batch_sizes.size(); i++) {
        batch_size_estimates[i] = estimate(supported_batch_sizes[i]);
    }

    // Create the request's strategies
    request->id = request_id_seed_++;
    request->strategies.reserve(batch_size_estimates.size());
    for (int i = batch_size_estimates.size()-1; i >= 0; i--) {
        Strategy strategy = std::make_shared<StrategyImpl>();
        strategy->priority = request->deadline - batch_size_estimates[i];
        strategy->deadline = request->deadline;
        strategy->batch_size = supported_batch_sizes[i];
        strategy->request_id = request->id;
        strategy->model = this;
        request->strategies.push_back(strategy);
    }

    // Enqueue the request
    requests_queued++;
    incoming.push(request);

    // Enqueue strategies to all loaded models
    for (auto &instance : instances) {
        if (instance->loaded) {
            instance->gpu->add_strategies(request);
        }
    }
}

std::vector<Scheduler::Request> Scheduler::Model::requests() {
    tbb::queuing_mutex::scoped_lock lock(mutex);

    std::vector<uint64_t> batch_size_estimates(supported_batch_sizes.size());
    for (unsigned i = 0; i < supported_batch_sizes.size(); i++) {
        batch_size_estimates[i] = estimate(supported_batch_sizes[i]);
    }

    Request newrequest;
    while (incoming.try_pop(newrequest)) queue.push_back(newrequest);
    
    return std::vector<Request>(queue.begin(), queue.end());
}

void Scheduler::Model::check_timeouts(uint64_t free_at) {
    while (!queue.empty()) {
        Request request = queue.front();
        if (request->deadline >= free_at) break;

        request->set_error(clockworkControllerCouldNotStartInTime, "");
        request->invalidate_strategies();
        requests_queued--;

        queue.pop_front();
    }
}

Scheduler::InferAction* Scheduler::Model::try_dequeue(
        uint64_t free_at,
        unsigned gpu_clock,
        Strategy &strategy)
{   
    uint64_t exec_time = estimate(strategy->batch_size, gpu_clock);
    uint64_t completion_time = free_at + exec_time;

    if (!strategy->valid || strategy->deadline < completion_time) {
        return nullptr;
    }

    InferAction* action;    
    {
        tbb::queuing_mutex::scoped_lock lock(mutex);

        // Strategy or request may have been invalidated while we were waiting
        if (!strategy->valid || strategy->deadline <= completion_time) {
            return nullptr;
        }

        // Pull any new requests
        Request newrequest;
        while (incoming.try_pop(newrequest)) queue.push_back(newrequest);

        // Drop any timed out requests
        check_timeouts(free_at);

        // Insufficient requests queued
        if (queue.size() < strategy->batch_size) return nullptr;

        // The request that generated this strategy has already completed (e.g. as part of a previous batch)
        // (Now that strategies are explicitly invalidated, this check should be unnecessary)
        // if (queue.front()->id > strategy->request_id) return nullptr;

        // See if the strategy can actually execute given the current GPU clock
        // (This check is done before acquiring the lock)
        // exec_time = estimate(strategy->batch_size, gpu_clock);
        // completion_time = free_at + exec_time;
        // if (completion_time > strategy->deadline) return nullptr;

        // See if this strategy has enough requests to fill its batch
        //   ie. that (batch_size-1) new requests arrived after this request arrived
        // Note that this is not simply queue.size()
        unsigned available_requests = 1 + queue.back()->id - strategy->request_id;
        if (available_requests < strategy->batch_size) {

            // All is not lost; scan the queue in reverse
            available_requests = 0;
            for (auto it = queue.rbegin(); it != queue.rend(); it++) {
                if ((*it)->deadline > completion_time) break;
                available_requests++;
                if (available_requests == strategy->batch_size) {
                    // Have to inherit new deadline
                    strategy->deadline = (*it)->deadline;
                    strategy->request_id = (*it)->id;
                    break;
                }
            }

            // Truly insufficient requests
            if (available_requests < strategy->batch_size) return nullptr;
        }

        // Skip this request if:
        // *  a greater batch size might be achievable by a subsequent request
        // *  there is insufficient time to execute both
        unsigned candidate_batchsize = batch_lookup(available_requests);
        if (strategy->batch_size < candidate_batchsize) {
            uint64_t candidate_exec_time = estimate(candidate_batchsize, gpu_clock);
            uint64_t candidate_completion_time = free_at + candidate_exec_time;

            // We can't bump up to the candidate batch size
            if (candidate_completion_time > strategy->deadline) return nullptr;

            strategy->batch_size = candidate_batchsize;
            exec_time = candidate_exec_time;
            completion_time = candidate_completion_time;
        }

        // Drop any requests that came before this strategy and can't be included
        while (queue.size() > 0 
                && queue.front()->id != strategy->request_id
                && queue.front()->deadline < completion_time) {
            Request request = queue.front();
            request->set_error(clockworkControllerSkipped, "");
            requests_queued--;
            request->invalidate_strategies();
            queue.pop_front();
        }

        // This shouldn't happen
        if (queue.size() < strategy->batch_size) return nullptr;

        action = new InferAction(scheduler, this);
        for (unsigned i = 0; i < strategy->batch_size; i++) {
            auto &request = queue.front();
            request->lock();
            action->requests.push_back(request);
            request->invalidate_strategies();
            requests_queued--;
            queue.pop_front();
        }
    }
    action->set_expectations(free_at, exec_time, gpu_clock);
    action->batch();

    return action;
}


const float Scheduler::estimate_percentile = 0.99;

void Scheduler::Model::add_measurement(unsigned batch_size, uint64_t duration, unsigned gpu_clock) {
    tbb::spin_mutex::scoped_lock lock(estimates_mutex);

    auto it = estimators.find(batch_size);
    CHECK(it != estimators.end()) << "Unsupported batch size " << batch_size;
    auto estimator = it->second;
    estimator->insert(duration * gpu_clock);

    estimates[batch_size] = estimator->get_percentile(Scheduler::estimate_percentile);
}

void Scheduler::Model::add_weights_measurement(uint64_t duration) {
    tbb::spin_mutex::scoped_lock lock(weights_estimate_mutex);

    weights_estimator->insert(duration);
    weights_estimate = weights_estimator->get_percentile(Scheduler::estimate_percentile);
}

uint64_t Scheduler::Model::estimate(unsigned batch_size) {
    return Scheduler::Model::estimate(batch_size, Scheduler::default_clock);
}

uint64_t Scheduler::Model::estimate(unsigned batch_size, int clock) {
    unsigned effective_batch_size = batch_lookup(batch_size);
    return estimates[effective_batch_size] / clock;
}

uint64_t Scheduler::Model::estimate_weights() {
    return weights_estimate;
}

Scheduler::InferAction::InferAction(Scheduler* scheduler, Model* model) : scheduler(scheduler), model(model) {
    action->id = action_id_seed++;
    action->model_id = model->id;
}

Scheduler::InferAction::~InferAction() {
    if (result != nullptr) {
        delete result->output;
        delete action->input;
    }
}


void Scheduler::InferAction::batch() {
    action->batch_size = requests.size();
    action->input_size = 0;

    if (!scheduler->has_logged_inputs_status.test_and_set()) {
        std::stringstream msg;
        if (requests[0]->request.input_size == 0) {
            if (scheduler->generate_inputs) {
                msg << "Network Status:  Client ✘✘✘✘✘ Controller ✔✔✔✔✔ Workers (inputs generated by controller)";
            } else {
                msg << "Network Status:  Client ✘✘✘✘✘ Controller ✘✘✘✘✘ Workers (inputs generated by worker)";                
            }
        } else {
            msg << "Network Status:  Client ✔✔✔✔✔ Controller ✔✔✔✔✔ Workers (inputs generated by client)";   
        }
        msg << std::endl;
        std::cout << msg.str();
    }

    for (auto &req : requests) {
        auto &r = req->request;
        if (r.input_size == 0 && scheduler->generate_inputs) {
            generated_inputs = true;
            char* generated_input;
            scheduler->input_generator->generatePrecompressedInput(model->input_size, &generated_input, &r.input_size);
            r.input = generated_input;
        }
        action->input_size += r.input_size;
    }

    action->input = new char[action->input_size];
    size_t offset = 0;
    for (auto &req : requests) {
        auto &r = req->request;
        std::memcpy(action->input + offset, r.input, r.input_size);
        offset += r.input_size;
        action->input_sizes.push_back(r.input_size);
    }
}

void Scheduler::InferAction::unbatch() {
    size_t single_output_size = result->output_size / requests.size();
    if (generated_inputs) single_output_size = 0;
    size_t offset = 0;
    for (unsigned i = 0; i < requests.size(); i++) {
        char* output = new char[single_output_size];
        std::memcpy(output, result->output + offset, single_output_size);
        offset += single_output_size;

        requests[i]->set_result(output, single_output_size);
    }
}

void Scheduler::InferAction::set_error(std::shared_ptr<workerapi::ErrorResult> &error) {
    this->error = error;
    for (auto &request : requests) {
        request->set_error(error->status, error->message);
    }
}

void Scheduler::InferAction::set_result(std::shared_ptr<workerapi::InferResult> &result) {
    this->result = result;
    this->unbatch();
}

float Scheduler::InferAction::complete(uint64_t now, int gpu_id) {
    std::vector<LoadTracker::Demand*> demands;
    demands.reserve(requests.size());

    float successful_requests = 0;
    float total_requests = 0;
    for (auto &request : requests) {
        if (request->complete(now, gpu_id)) {
            successful_requests += 1;
        }
        demands.push_back(&(request->demand));
        total_requests += 1;
    }

    if (demands.size() > 0) {
        tbb::queuing_mutex::scoped_lock lock(model->scheduler->tracker->mutex);
        model->scheduler->tracker->requestsCompleted(demands, gpu_id);
    }

    return successful_requests / total_requests;
}

void Scheduler::InferAction::set_expectations(uint64_t exec_start, uint64_t duration, int clock) {
    uint64_t now = util::now();
    action->expected_duration = duration;
    action->expected_exec_complete = exec_start + duration;
    action->expected_gpu_clock = clock;
    // action->earliest = exec_start;
    // action->latest = action->earliest + Scheduler::latest_delta;
    // action->earliest = std::max(util::now() + future, exec_start - Scheduler::latest_delta);
    action->earliest = now;
    action->latest = std::max(now + future + scheduler->latest_delta, exec_start + scheduler->latest_delta);
    send_by = std::max(now + future, action->latest - 5000000UL);
    report_error_at = action->latest + duration;
    // action->earliest = util::now() - Scheduler::schedule_ahead;
    // action->latest = action->expected_exec_complete + Scheduler::latest_delta;
}

        // ModelInstance* instance;
        // unsigned version;
        // ControllerActionTelemetry telemetry;
        // std::shared_ptr<workerapi::LoadWeights> action = std::make_shared<workerapi::LoadWeights>();
        // std::shared_ptr<workerapi::ErrorResult> error = nullptr;
        // std::shared_ptr<workerapi::LoadWeightsResult> result = nullptr;

Scheduler::LoadWeightsAction::LoadWeightsAction(
    Scheduler* scheduler, ModelInstance* instance)
      : scheduler(scheduler), instance(instance) {
    action->id = action_id_seed++;
    action->model_id = instance->model->id;
}

void Scheduler::LoadWeightsAction::set_expectations(uint64_t exec_start, uint64_t duration) {
    uint64_t now = util::now();
    action->expected_duration = duration;
    action->expected_exec_complete = exec_start + duration;
    // action->earliest = std::max(now + future, exec_start - scheduler->latest_delta);
    action->earliest = now;
    action->latest = std::max(now + future + scheduler->latest_delta, exec_start + scheduler->latest_delta);
}

void Scheduler::LoadWeightsAction::set_error(std::shared_ptr<workerapi::ErrorResult> &error) {
    this->error = error;
    if (version = instance->version) {
        instance->version++;
        instance->loading = false;
        instance->loaded = false;
    }
}

void Scheduler::LoadWeightsAction::set_result(std::shared_ptr<workerapi::LoadWeightsResult> &result) {
    this->result = result;
    if (version == instance->version) {
        instance->loaded = true;
        instance->loading = false;
    }
}

Scheduler::EvictWeightsAction::EvictWeightsAction(ModelInstance* instance) : instance(instance) {
    action->id = action_id_seed++;
    action->model_id = instance->model->id;
}

void Scheduler::EvictWeightsAction::set_expectations() {
    action->earliest = 0;
    action->latest = UINT64_MAX;
}

void Scheduler::EvictWeightsAction::set_error(std::shared_ptr<workerapi::ErrorResult> &error) {
    this->error = error;
    CHECK(false) << "Error in EvictWeightsAction" << error->str();
}

void Scheduler::EvictWeightsAction::set_result(std::shared_ptr<workerapi::EvictWeightsResult> &result) {
    this->result = result;
}

Scheduler::GPU::GPU(
    unsigned id,
    Scheduler* scheduler, 
    network::controller::WorkerConnection* worker,
    unsigned worker_id,
    unsigned gpu_id,
    unsigned pages) 
    : id(id),
      scheduler(scheduler),
      worker(worker),
      worker_id(worker_id),
      gpu_id(gpu_id),
      pages(pages),
      free_pages(pages),
      exec(Scheduler::default_clock, Scheduler::lag, Scheduler::future), 
      loadweights(Scheduler::default_clock, Scheduler::lag, Scheduler::future) {
}

void Scheduler::GPU::send_action(InferAction* action) {
    auto &infer = action->action;
    infer->gpu_id = gpu_id;
    infer->worker_id = worker_id;

    // Update GPU state
    {
        tbb::spin_mutex::scoped_lock lock(exec_mutex);
        exec.add(infer->id, infer->expected_duration);
    }

    // Save the callback
    auto callback = [this, action](std::shared_ptr<workerapi::Result> result) {
        this->infer_result(action, result);
    };
    scheduler->add_callback(infer->id, callback);

    // Record the telemetry
    action->telemetry.set(infer);
    action->telemetry.requests_queued = action->model->requests_queued;
    action->telemetry.copies_loaded = action->model->copies_loaded;

    // Send the action
    scheduler->network->send(worker, infer, action->send_by, action->report_error_at);

    // Immediately mark the requests as executing for load balancer
    {
        std::vector<LoadTracker::Demand*> demands;
        demands.reserve(action->requests.size());
        for (auto &request : action->requests) {
            demands.push_back(&(request->demand));
        }

        tbb::queuing_mutex::scoped_lock lock(scheduler->tracker->mutex);
        scheduler->tracker->requestsExecuting(demands, id);
    }

    if (print_debug) std::cout << ("Worker <--  " + infer->str() + "\n");
}

void Scheduler::GPU::send_action(LoadWeightsAction* action) {
    auto &load = action->action;
    load->gpu_id = gpu_id;
    load->worker_id = worker_id;

    // Update PCI state
    {
        tbb::spin_mutex::scoped_lock lock(loadweights_mutex);
        loadweights.add(load->id, load->expected_duration);
    }

    // Save the callback
    auto callback = [this, action](std::shared_ptr<workerapi::Result> result) {
        this->load_result(action, result);
    };
    scheduler->add_callback(load->id, callback);

    // Send the action
    scheduler->network->send(worker, load, UINT64_MAX, 0);

    // Record the telemetry
    action->telemetry.set(load);
    action->telemetry.requests_queued = action->instance->model->requests_queued;
    action->telemetry.copies_loaded = action->instance->model->copies_loaded;

    if (print_debug || print_loads) std::cout << ("Worker <--  " + load->str() + "\n");
}

void Scheduler::GPU::send_action(EvictWeightsAction* action) {
    auto &evict = action->action;
    evict->gpu_id = gpu_id;
    evict->worker_id = worker_id;

    // Save the callback
    auto callback = [this, action](std::shared_ptr<workerapi::Result> result) {
        this->evict_result(action, result);
    };
    scheduler->add_callback(evict->id, callback);

    // Send the action
    scheduler->network->send(worker, evict, UINT64_MAX, 0);

    // Record the telemetry
    action->telemetry.set(evict);
    action->telemetry.requests_queued = action->instance->model->requests_queued;
    action->telemetry.copies_loaded = action->instance->model->copies_loaded+1;

    if (print_debug || print_loads) std::cout << ("Worker <--  " + evict->str() + "\n");    
}

std::vector<Scheduler::EvictWeightsAction*> Scheduler::GPU::evict_pages(unsigned required_pages) {
    std::vector<EvictWeightsAction*> ret;
    while (free_pages < required_pages) {
        int model_id = scheduler->tracker->evictModel(id);

        if (model_id == -1) break;

        ModelInstance* instance = instances[model_id];
        instance->version++;
        instance->loading = false;
        instance->loaded = false;
        instance->model->copies_loaded--;

        EvictWeightsAction* evict = new EvictWeightsAction(instance);
        evict->set_expectations();
        ret.push_back(evict);
        
        free_pages += scheduler->models[model_id]->num_weights_pages;
        eviction_required = true; // GPU reached capacity; evictions required in future
    }
    return ret;
}

bool Scheduler::GPU::schedule_load() {
    tbb::queuing_mutex::scoped_lock lock(load_mutex);

    uint64_t now = util::now();
    if (last_load + 100000UL > now) return false;

    if (loads >= Scheduler::max_loads) return false;

    uint64_t available;
    {
        tbb::spin_mutex::scoped_lock lock(loadweights_mutex);
        available = loadweights.available();
    }

    if (available >= now + scheduler->schedule_ahead) return false;

    ModelInstance* instance;
    unsigned size;
    std::vector<EvictWeightsAction*> evict_actions;
    {
        tbb::queuing_mutex::scoped_lock load_lock(scheduler->tracker->load_mutex);
        tbb::queuing_mutex::scoped_lock lock(scheduler->tracker->mutex);

        last_load = util::now();

        int model_id = scheduler->tracker->loadModel(id, eviction_required);
        if (model_id == -1) {
            return false;
        }

        instance = instances[model_id];
        CHECK(instance->loaded == false && instance->loading == false) << "Tracker asked to load model that is already loaded";

        size = scheduler->models[model_id]->num_weights_pages;
        evict_actions = evict_pages(size);

        if (free_pages < size) {
            scheduler->tracker->loadModelComplete(id, model_id, false);
        }
    }

    // Send the evict actions
    for (auto &evict : evict_actions) {
        send_action(evict);
    }

    if (free_pages < size) {
        return false;
    }

    free_pages -= size;

    uint64_t expected_duration = instance->model->estimate_weights();

    LoadWeightsAction* action = new LoadWeightsAction(scheduler, instance);
    action->version = ++instance->version;
    instance->loading = true;
    instance->loaded = false;
    action->set_expectations(available, expected_duration);
    loads++;

    send_action(action);
    return true;
}

bool Scheduler::GPU::schedule_infer() {
    tbb::queuing_mutex::scoped_lock lock(infer_mutex);

    bool active = false;

    // Handle all newly-loaded models, add to strategy queue
    ModelInstance* newly_loaded;
    while (newly_loaded_models.try_pop(newly_loaded)) {
        auto requests = newly_loaded->model->requests();

        for (auto &request : requests) {
            for (auto &strategy : request->strategies) {
                if (strategy->valid) {
                    strategy_queue.push(strategy);
                }
            }
        }
        active = true;
    }

    // Drain all incoming strategies, add to strategy_queue
    Request request;
    while (incoming_strategies.try_pop(request)) {
        for (auto &strategy : request->strategies) {
            if (strategy->valid) {
                strategy_queue.push(strategy);
            }
        }
        active = true;
    }

    // Drop any strategies already processed or with missed deadlines
    while (strategy_queue.size() > 0) {
        auto &strategy = strategy_queue.top();

        if (strategy->valid && strategy->deadline > util::now()) {
            break;
        }

        strategy_queue.pop();
    }

    // if (last_print + 1000000000UL < now) {
    //     last_print = now;
    //     std::stringstream s;
    //     s << "GPU " << id << " strategy queue " << strategy_queue.size() << std::endl;
    //     std::cout << s.str();
    // }

    // Schedule infer actions
    
    while (strategy_queue.size() > 0) {
        uint64_t exec_at;
        int clock;
        {
            tbb::spin_mutex::scoped_lock lock(exec_mutex);
            exec_at = exec.available();
            clock = exec.clock();
        }

        uint64_t schedule_until = util::now() + scheduler->schedule_ahead;
        if (exec_at >= schedule_until) break;

        Strategy strategy = strategy_queue.top();

        // Only valid strategies, for which this GPU has the model loaded
        if (strategy->valid && instances[strategy->model->id]->loaded) {
            InferAction* action = strategy->model->try_dequeue(exec_at, clock, strategy);

            if (action != nullptr) {
                send_action(action);
                active = true;
            }
        }

        strategy_queue.pop();
    }

    return active;
}

void Scheduler::GPU::infer_error(InferAction* action, std::shared_ptr<workerapi::ErrorResult> &error) {
    // std::cout << ("Worker  --> " + error->str() + "\n");

    action->telemetry.set(error);
    
    // Update GPU state tracking
    {
        tbb::spin_mutex::scoped_lock lock(exec_mutex);
        exec.error(error->id, util::now());
    }

    action->set_error(error);
    CHECK(action->complete(util::now(), id) == 0) << "ErrorResult should not result in successful requests";

    action->telemetry.goodput = 0;
    scheduler->printer->log(action->telemetry);

    delete action;
}

void Scheduler::GPU::infer_success(InferAction* action, std::shared_ptr<workerapi::InferResult> &result) {
    action->telemetry.set(result);

    // Update GPU state tracking
    {
        tbb::spin_mutex::scoped_lock lock(exec_mutex);
        exec.success(result->id, result->exec.end);
        exec.update_clock(result->gpu_clock);
    }

    // Update model execution tracking
    action->model->add_measurement(
        action->action->batch_size, 
        result->exec.duration, 
        (result->gpu_clock + result->gpu_clock_before) / 2
    );

    action->set_result(result);
    action->telemetry.goodput = action->complete(util::now(), id);

    scheduler->printer->log(action->telemetry);

    delete action;
}

void Scheduler::GPU::infer_result(InferAction* action, std::shared_ptr<workerapi::Result> &result) {
    if (auto error = std::dynamic_pointer_cast<workerapi::ErrorResult>(result)) {
        infer_error(action, error);

    } else if (auto infer = std::dynamic_pointer_cast<workerapi::InferResult>(result)) {
        infer_success(action, infer);

    } else {
        CHECK(false) << "Unexpected response to Infer action" << result->str();

    }
}

void Scheduler::GPU::load_error(LoadWeightsAction* action, std::shared_ptr<workerapi::ErrorResult> &error){
    action->telemetry.set(error);
    action->telemetry.goodput = 0;
    action->set_error(error);

    // Track model status
    {
        tbb::queuing_mutex::scoped_lock lock(scheduler->tracker->mutex);
        scheduler->tracker->loadModelComplete(id, action->instance->model->id, false);
    }
    free_pages += action->instance->model->num_weights_pages;

    // Update PCI state tracking
    {
        tbb::spin_mutex::scoped_lock lock(loadweights_mutex);
        loadweights.error(error->id, util::now());
    }

    scheduler->printer->log(action->telemetry);

    delete action;
}

void Scheduler::GPU::load_success(LoadWeightsAction* action, std::shared_ptr<workerapi::LoadWeightsResult> &result) {
    action->telemetry.set(result);
    action->set_result(result);

    // Track model status
    {
        tbb::queuing_mutex::scoped_lock lock(scheduler->tracker->mutex);
        scheduler->tracker->loadModelComplete(id, action->instance->model->id, true);
    }

    // Update PCI state tracking
    {
        tbb::spin_mutex::scoped_lock lock(loadweights_mutex);
        loadweights.success(result->id, result->end);
    }

    // Update PCI tracking
    action->instance->model->add_weights_measurement(result->duration);
    action->instance->model->copies_loaded++;

    // Enable new requests
    newly_loaded_models.push(action->instance);

    // TODO: change this?
    action->telemetry.goodput = 1.0;

    scheduler->printer->log(action->telemetry);

    delete action;
}

void Scheduler::GPU::load_result(LoadWeightsAction* action, std::shared_ptr<workerapi::Result> &result) {
    if (!print_debug && print_loads) std::cout << ("Worker  --> " + result->str() + "\n");

    loads--;

    if (auto error = std::dynamic_pointer_cast<workerapi::ErrorResult>(result)) {
        load_error(action, error);

    } else if (auto load = std::dynamic_pointer_cast<workerapi::LoadWeightsResult>(result)) {
        load_success(action, load);

    } else {
        CHECK(false) << "Unexpected response to LoadWeights action" << result->str();

    }    
}

void Scheduler::GPU::evict_error(EvictWeightsAction* action, std::shared_ptr<workerapi::ErrorResult> &error){
    action->telemetry.set(error);

    action->set_error(error);

    scheduler->printer->log(action->telemetry);

    delete action;
}

void Scheduler::GPU::evict_success(EvictWeightsAction* action, std::shared_ptr<workerapi::EvictWeightsResult> &result) {
    action->telemetry.set(result);

    action->set_result(result);

    scheduler->printer->log(action->telemetry);

    delete action;
}

void Scheduler::GPU::evict_result(EvictWeightsAction* action, std::shared_ptr<workerapi::Result> &result) {
    if (!print_debug && print_loads) std::cout << ("Worker  --> " + result->str() + "\n");

    if (auto error = std::dynamic_pointer_cast<workerapi::ErrorResult>(result)) {
        evict_error(action, error);

    } else if (auto evict = std::dynamic_pointer_cast<workerapi::EvictWeightsResult>(result)) {
        evict_success(action, evict);

    } else {
        CHECK(false) << "Unexpected response to LoadWeights action" << result->str();

    }
}

void Scheduler::validate_clockwork_state(ClockworkState &state) {
    unsigned cache_size = state.workers[0].gpus[0].weights_cache_total_pages;
    for (auto &worker : state.workers) {
        for (auto &gpu : worker.gpus) {
            CHECK(gpu.weights_cache_total_pages == cache_size) 
                << "Expect same cache size on all GPUs";
        }
    }

    for (auto &p : state.workers[0].models) {
        unsigned model_id = p.first;
        for (auto &worker : state.workers) {
            CHECK(worker.models.find(model_id) != worker.models.end()) 
                << "Inconsistent models across workers";
        }
    }
}

void Scheduler::print_status() {
    unsigned model_pages = 0;
    for (auto &model : models) {
        model_pages += model->num_weights_pages;
    }

    unsigned gpu_pages = 0;
    for (auto &gpu : gpus) {
        gpu_pages += gpu->pages;
    }


    std::cout << "Total GPU capacity " << gpu_pages << " pages (" << gpu_pages/gpus.size() << " per GPU)." << std::endl
              << "Total model pages " << model_pages << " (" << (100*model_pages / gpu_pages) << "% oversubscription)." << std::endl;
}

void Scheduler::initialize_models(ClockworkState &state) {
    models.resize(state.workers[0].models.size(), nullptr);

    for (auto &p : state.workers[0].models) {
        auto &model = p.second;

        if (model.id >= models.size()) {
            models.resize(model.id+1, nullptr);
        }

        models[model.id] = new Model(this, model);
    }

    std::cout << "Created " << models.size() << " models" << std::endl;
}

void Scheduler::initialize_gpus(std::vector<network::controller::WorkerConnection*> workers,
                ClockworkState &state) 
{
    unsigned total_pages = 0;
    unsigned workers_remaining = state.workers.size();
    unsigned gpus_remaining = max_gpus;
    for (WorkerState &worker : state.workers) {
        int num_gpus = std::min((unsigned) worker.gpus.size(), gpus_remaining / workers_remaining);
        for (unsigned i = 0; i < num_gpus; i++) {
            GPUState &gpustate = worker.gpus[i];
            GPU* gpu = new GPU(
                gpus.size(),
                this,
                workers[worker.id],
                worker.id,
                gpustate.id,
                gpustate.weights_cache_total_pages
            );
            gpus.push_back(gpu);

            total_pages += gpu->pages;
        }
        gpus_remaining -= num_gpus;
        workers_remaining--;
    }
    std::cout << "Created " << gpus.size() << " GPUs on " << state.workers.size() << " Workers" << std::endl;
}

void Scheduler::initialize_model_instances() {
    for (auto &gpu : gpus) {
        gpu->instances.resize(models.size(), nullptr);
    }
    for (auto &model : models) {
        model->instances.resize(gpus.size(), nullptr);
    }

    for (unsigned i = 0; i < gpus.size(); i++) {
        GPU* gpu = gpus[i];
        for (unsigned j = 0; j < models.size(); j++) {
            Model* model = models[j];
            ModelInstance* instance = new ModelInstance(gpu, model);
            model->instances[i] = instance;
            gpu->instances[j] = instance;
        }
    }

    tracker = new LoadTracker(gpus.size(), models.size(), default_slo);
}


void networkPrintThread(std::vector<network::controller::WorkerConnection*> workers) {
    uint64_t last_print = util::now();
    uint64_t print_interval_nanos = 1000000000UL * 10;

    network::connection_stats previous_stats;
    while (true) {
        uint64_t now = util::now();
        if (last_print + print_interval_nanos > now) {
            usleep(100000);
            continue;
        }

        network::connection_stats stats;
        for (auto &worker : workers) {
            stats += worker->stats;
        }
        stats -= previous_stats;
        previous_stats = stats;

        float duration = (now - last_print) / 1000000000.0;
        stats /= duration;

        std::stringstream msg;
        msg << std::fixed << std::setprecision(1);
        msg << "Network->Workers: ";
        msg << (stats.bytes_sent / (1024*1024.0)) << "MB/s ";
        msg << "(" << stats.messages_sent << " msgs) snd, ";
        msg << (stats.bytes_received / (1024*1024.0)) << "MB/s ";
        msg << "(" << stats.messages_received << " msgs) rcv, ";
        msg << std::endl;

        std::cout << msg.str();

        last_print = now;
    }
}

void Scheduler::initialize_network(std::vector<network::controller::WorkerConnection*> workers) {
    auto transmitComplete = [this]() {
        this->network->sendComplete();
    };
    auto transmitError = [this](uint64_t timeout_at, std::shared_ptr<workerapi::Result> result) {
        network_timeout_queue.push({timeout_at, result});
    };

    this->network = new NetworkExecutor(network_concurrency, transmitError);

    for (auto worker : workers) {
        worker->setTransmitCallback(transmitComplete);
    }
}


// Called when model loading has completed
void Scheduler::start(std::vector<network::controller::WorkerConnection*> workers,
                    ClockworkState &state) 
{
    validate_clockwork_state(state);
    initialize_models(state);
    initialize_gpus(workers, state);
    initialize_model_instances();
    initialize_network(workers);

    print_status();

    // Create and start the printer threads
    this->printer = ControllerActionTelemetry::log_and_summarize(actions_filename, print_interval);
    network_printer = std::thread(&networkPrintThread, workers);
    threading::initLoggerThread(network_printer);

    uint64_t num_admission_threads = 2;
    for (int i = 0; i < num_admission_threads; i++) {
        admission_threads.push_back(std::thread(&Scheduler::run_admission_thread, this));
        threading::initHighPriorityThread(admission_threads[i]);
    }

    uint64_t num_results_threads = 2;
    for (int i = 0; i < num_results_threads; i++) {
        results_threads.push_back(std::thread(&Scheduler::run_results_thread, this));
        threading::initHighPriorityThread(results_threads[i]);
    }


    int num_infer_threads = 5;
    for (unsigned i = 0; i < num_infer_threads; i++) {
        infer_threads.push_back(std::thread(&Scheduler::run_infer_thread, this, i));
        threading::initHighPriorityThread(infer_threads[i]);
    }
    for (auto gpu : gpus) {
        to_infer.push(gpu);
    }

    int num_load_threads = 5;
    for (unsigned i = 0; i < num_load_threads; i++) {
        load_threads.push_back(std::thread(&Scheduler::run_load_thread, this, i));
        threading::initHighPriorityThread(load_threads[i]);
    }
    for (auto gpu : gpus) {
        to_load.push(gpu);
    }
}

struct tracker_request {
    int model_id;
    uint64_t size;
    uint64_t start_exec_by;
    uint64_t start_loadweights_by;
};

void Scheduler::handle_requests(std::vector<Request> &requests) {
    std::vector<tracker_request> tracker_requests;
    tracker_requests.reserve(requests.size());

    for (auto &request : requests) {
        int model_id = request->request.model_id;
        Model* model = models[model_id];

        request->set_model(model);
        request->set_slo(default_slo);

        tracker_requests.push_back({
            model_id, model->estimate(1), request->exec_slo, request->weights_slo
        });
    }

    {
        tbb::queuing_mutex::scoped_lock lock(tracker->mutex);

        for (int i = 0; i < requests.size(); i++) {
            auto &t = tracker_requests[i];
            requests[i]->demand = tracker->addRequest(t.model_id, t.size, t.start_exec_by, t.start_loadweights_by);
        }
    }

    for (auto &request : requests) {
        request->model->enqueue(request);
    }
}

void Scheduler::add_callback(uint64_t action_id, Callback callback) {
    auto pair = std::make_pair(action_id, callback);

    tbb::spin_mutex::scoped_lock lock(callbacks_mutex);
    callbacks.insert(pair);
}

void Scheduler::handle_result(std::shared_ptr<workerapi::Result> &result) {
    Callback callback;
    {
        tbb::spin_mutex::scoped_lock lock(callbacks_mutex);

        auto it = callbacks.find(result->id);
        CHECK(it != callbacks.end()) 
            << "Received result for non-existent action " << result->str();

        callback = it->second;
        callbacks.erase(it);
    }

    callback(result);
}

void Scheduler::run_admission_thread() {
    // Process this many requests per iteration
    int max_requests = 50;
    std::vector<Request> requests;
    requests.reserve(max_requests);

    std::priority_queue<Request, std::deque<Request>, RequestImpl::DeadlineComparator> timeout_queue;

    while (true) {
        // Dequeue up to `max_requests`
        Request request;
        while (requests.size() < max_requests && request_queue.try_pop(request)) {
            // Immediately drop requests to invalid models
            unsigned model_id = request->request.model_id;
            if (model_id > models.size() || models[model_id] == nullptr) {
                request->set_error(clockworkError, "Invalid model ID");
                CHECK(!request->complete(util::now(), -1)) << "Erroneous request should not be successful";
                continue;
            }

            requests.push_back(request);
        }

        if (requests.size() > 0) {
            handle_requests(requests);

            for (auto &request : requests) {
                timeout_queue.push(request);
            }

            requests.clear();
        }

        // Drop any timed out requests
        uint64_t now = util::now();
        while (!timeout_queue.empty()) {
            auto &request = timeout_queue.top();

            if (request->deadline > now) break;

            request->finalize();
            request->invalidate_strategies();
            timeout_queue.pop();
        }

        usleep(10);

    }
}

void Scheduler::run_results_thread() {
    bool should_timeout = false;
    TimeoutResult next_timeout;

    while (true) {
        bool active = false;
        std::shared_ptr<workerapi::Result> result;
        if (result_queue.try_pop(result)) {
            handle_result(result);
            active = true;
        }

        if (!should_timeout) {
            should_timeout = network_timeout_queue.try_pop(next_timeout);
        }

        if (should_timeout) {
            if (next_timeout.timeout_at <= util::now()) {
                handle_result(next_timeout.result);
                should_timeout = false;
                active = true;
            }
        }

        if (!active) {
            usleep(10);
        }
    }
}

void Scheduler::run_infer_thread(int id) {
    std::stringstream msg;
    msg << "GPU infer thread [" << id << "] started" << std::endl;
    std::cout << msg.str();

    int inactive = 0;
    int n_gpus = gpus.size();
    int i = 0;
    while (true) {
        uint64_t i = (next_infer++) % n_gpus;
        bool active = gpus[i]->schedule_infer();

        if (i++ % 100 == 0) {
            usleep(10);
        }
    }
}

void Scheduler::run_load_thread(int id) {
    std::stringstream msg;
    msg << "GPU load thread [" << id << "] started" << std::endl;
    std::cout << msg.str();

    int inactive = 0;
    int n_gpus = gpus.size();
    while (true) {
        uint64_t i = (next_load++) % n_gpus;
        bool active = gpus[i]->schedule_load();
        usleep(10);
    }

}

// The actual scheduler interface implementation, invoked by worker network thread
void Scheduler::resultFromWorker(std::shared_ptr<workerapi::Result> result)
{
    if (print_debug) std::cout << ("Worker  --> " + result->str() + "\n");

    result->result_received = util::now();
    result_queue.push(result);
}

// The actual scheduler interface implementation, invoked by client network thread
void Scheduler::clientInfer(clientapi::InferenceRequest &request, 
    std::function<void(clientapi::InferenceResponse&)> callback)
{
    if (print_debug) std::cout << ("Client  --> " + request.str() + "\n");

    request_queue.push(std::make_shared<RequestImpl>(this, request, callback));
}

Scheduler::NetworkExecutor::NetworkExecutor(unsigned concurrency, 
    std::function<void(uint64_t, std::shared_ptr<workerapi::Result>)> error_callback) : 
idle(concurrency), error_callback(error_callback) {}

void Scheduler::NetworkExecutor::send(network::controller::WorkerConnection* worker, 
          std::shared_ptr<workerapi::Action> action,
          uint64_t start_send_by,
          uint64_t send_error_at) {

    NetworkAction toSend;
    {
        tbb::spin_mutex::scoped_lock lock(mutex);

        pending.push_back({worker, action, start_send_by, send_error_at});

        if (idle == 0) return;

        if (!next(toSend)) return;

        idle--;
    }

    toSend.worker->sendAction(toSend.action);            
}

void Scheduler::NetworkExecutor::sendComplete() {
    NetworkAction toSend;
    {
        tbb::spin_mutex::scoped_lock lock(mutex);
        if (!next(toSend)) {
            idle++;
            return;
        }
    }

    toSend.worker->sendAction(toSend.action);
}

bool Scheduler::NetworkExecutor::next(NetworkAction &toSend) {
    uint64_t now = util::now();
    while (pending.size() > 0) {
        toSend = pending.front();
        pending.pop_front();

        if (toSend.start_send_by >= now) {
            return true;
        }

        auto action = toSend.action;
        auto result = std::make_shared<workerapi::ErrorResult>();
        result->id = action->id;
        result->action_type = action->action_type;
        result->status = networkSendTooLate;
        result->action_received = now;
        result->result_sent = now;
        result->result_received = now;
        result->message = "Could not send action to worker in time";

        error_callback(toSend.send_error_at, result);

    }
    return false;
}

}
}
}