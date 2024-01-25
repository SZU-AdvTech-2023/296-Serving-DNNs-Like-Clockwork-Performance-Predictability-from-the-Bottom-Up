#include "clockwork/controller/infer5/load_tracker.h"
#include "clockwork/util.h"
#include "dmlc/logging.h"

namespace clockwork {
namespace scheduler {
namespace infer5 {

ModelLoadTracker* LoadTracker::newModelTracker(int model_id) {
    return new ModelLoadTracker(capacity, model_id, gpus.size());
}
   
ModelLoadTracker::ModelLoadTracker(int64_t capacity, int model_id, int n_gpus) : 
        capacity(capacity),
        model_id(model_id), 
        touched(n_gpus) {
    for (int i = 0; i < touched.size(); i++) {
        touched[i] = false;
    }
}

LoadTracker::Demand ModelLoadTracker::addRequest(int64_t size, uint64_t start_exec_by, uint64_t start_loadweights_by) {
    LoadTracker::Demand demand;
    demand.exec_size = (size * capacity) / start_exec_by;
    demand.loadweights_size = (size * capacity) / start_loadweights_by;
    demand.model_id = model_id;

    this->start_loadweights_by = start_loadweights_by;
    new_exec += demand.exec_size;
    new_loadweights += demand.loadweights_size;

    return demand;
}

void ModelLoadTracker::executing(LoadTracker::Demand &demand, int gpu_id) {
    delta_loadweights += demand.loadweights_size;
    demand.loadweights_size = 0;
    touched[gpu_id] = true;
}

void ModelLoadTracker::completed(LoadTracker::Demand &demand, int gpu_id) {
    delta_exec += demand.exec_size;
    demand.exec_size = 0;
    touched[gpu_id] = true;
}

void ModelLoadTracker::cancelled(LoadTracker::Demand &demand) {
    delta_loadweights += demand.loadweights_size;
    delta_exec += demand.exec_size;
    demand.loadweights_size = 0;
    demand.exec_size = 0;
}

void ModelLoadTracker::loadComplete(int gpu_id, bool success) {
    events.push({gpu_id, success});
}

void LoadTracker::attach(GPU &gpu) {
    for (auto &priority : gpu.detached) {
        CHECK(priority->detached) << "Attaching model already attached";

        // Put back in to priority queues
        if (priority->model->loading[gpu.id]) {
            // Loading on a GPU is neither loadable nor evictable
        } else if (priority->model->gpus[gpu.id]) {
            gpu.cached.insert(priority);
        } else {
            gpu.not_cached.insert(priority);
        }

        priority->detached = false;
    }

    gpu.detached.clear();
}

void LoadTracker::detach(Model &model) {
	// Remove from priority queues
    for (unsigned i = 0; i < n_gpus; i++) {
        auto &gpu = gpus[i];
        auto &priority = model.priorities[i];

        // Only detach once
        if (priority->detached) continue;
        priority->detached = true;
        gpu.detached.push_back(priority);

        if (model.loading[i]) {
            // Loading on a GPU is neither loadable nor evictable
        } else if (model.gpus[i]) {
            auto it = gpu.cached.find(priority);
            CHECK(it != gpu.cached.end()) << "Thought we were cached when we weren't";

            gpu.cached.erase(it);
        } else {
            auto it = gpu.not_cached.find(priority);
            CHECK(it != gpu.not_cached.end()) << "Thought we were not cached when we were";

            gpu.not_cached.erase(it);
        }
    }

}

void LoadTracker::invalidatePriorities(Model &model) {
    if (model.stale) return;
    model.stale = true;
    stale.push_back(&model);
}

void LoadTracker::refreshPriorities() {
    for (auto &model : stale) {
        updatePriority(*model);
        model->stale = false;
    }
    stale.clear();
}

void LoadTracker::updatePriority(Model &model) {
    CHECK(model.stale) << "Updating priority on non-stale model";

    // Calculate each GPU's weight
    double total_weight = 0;
    for (unsigned i = 0; i < n_gpus; i++) {
        if (model.gpus[i]) {
            total_weight += gpus[i].weight;
        }
        CHECK(model.priorities[i]->detached) << "Updating priority on attached model";
    }

    // Load priority is calculated differently to evict priority
    // First, load priority.  Load priority is simply whether we can satisfy outstanding_loadweights
    int64_t load_priority = model.outstanding_loadweights;

    // Subtract served load
    if (total_weight > 0 && load_priority > 0) {
        for (unsigned i = 0; i < n_gpus; i++) {
            if (!model.gpus[i]) continue; // Skip models we are not loaded on

            int64_t required = model.outstanding_loadweights * (gpus[i].weight / total_weight);
            int64_t served = (capacity * required) / gpus[i].outstanding;
            load_priority -= served;
        }
    }

    bool is_empty = model.outstanding_loadweights == 0 && model.outstanding_exec == 0;

    for (unsigned i = 0; i < n_gpus; i++) {
        if (model.gpus[i]) {
            model.priorities[i]->priority = model.last_used[i];
        } else {
            model.priorities[i]->priority = load_priority;
        }
        model.priorities[i]->is_empty = is_empty;
        model.priorities[i]->last_used = model.last_used[i];
    }
}

void LoadTracker::clearLoad(Model &model) {
    for (unsigned i = 0; i < n_gpus; i++) {
        gpus[i].outstanding -= model.allocations[i];
        model.allocations[i] = 0;
    }
}

void LoadTracker::distributeLoad(Model &model) {
    // Update all the counters
    model.outstanding_exec -= model.completed_exec;
    model.completed_exec = 0;
    int64_t loadweights_delta = std::max(model.completed_loadweights, model.timedout_loadweights);
    model.outstanding_loadweights -= loadweights_delta;
    model.completed_loadweights -= loadweights_delta;
    model.timedout_loadweights -= loadweights_delta;

    clearLoad(model);

    if (model.gpu_count == 0) return;

    // For demand tracking we use exec

    double total_weight = 0;
    for (unsigned i = 0; i < n_gpus; i++) {
        if (model.gpus[i]) {
            total_weight += gpus[i].weight;
        }
    }

    for (unsigned i = 0; i < n_gpus; i++) {
        if (model.gpus[i]) {
            auto allocation = model.outstanding_exec * (gpus[i].weight / total_weight);
            model.allocations[i] = allocation;
            gpus[i].outstanding += allocation;
            gpus[i].weight = capacity / ((double) gpus[i].outstanding);
        }
    }
}

void LoadTracker::addGPU(Model &model, GPU &gpu) {
    CHECK(!model.gpus[gpu.id]) << "Adding model to GPU that already has it";
    CHECK(!model.loading[gpu.id]) << "Adding model to GPU that is already loading it";
    CHECK(!gpu.models[model.id]) << "Adding model to GPU that thinks it already has it";

    model.gpus[gpu.id] = true;
    model.loading[gpu.id] = true;
    gpu.models[model.id] = true;
    model.priorities[gpu.id]->preference = model.gpu_count++;
    model.last_used[gpu.id] = seqno_seed++;
}

void LoadTracker::addGPUcomplete(Model &model, GPU &gpu) {
    CHECK(model.gpus[gpu.id]) << "Model load completed on GPU that didn't expect it";
    CHECK(gpu.models[model.id]) << "Model load completed on GPU that didn't expect it";
    CHECK(model.loading[gpu.id]) << "Model load completed on GPU that wasn't loading";

    model.loading[gpu.id] = false;
    model.last_used[gpu.id] = seqno_seed++;
}

void LoadTracker::removeGPU(Model &model, GPU &gpu, bool evicted) {
    CHECK(model.gpus[gpu.id]) << "Removing Model from GPU that doesn't have it";
    CHECK(gpu.models[model.id]) << "Removing Model from GPU that doesn't think it has it";
    if (evicted) {
        CHECK(!model.loading[gpu.id]) << "Evicted loading model";
    } else {
        CHECK(model.loading[gpu.id]) << "Evicted model that is not loading";
    }
    
    model.gpus[gpu.id] = false;
    model.loading[gpu.id] = false;
    gpu.models[model.id] = false;
    model.gpu_count--;
    for (unsigned i = 0; i < n_gpus; i++) {
        auto pref = model.priorities[gpu.id]->preference;
        if (model.priorities[i]->preference > pref) {
            model.priorities[i]->preference--;
        }
        if (model.gpus[i]) {
            model.priorities[i]->last_used = seqno_seed++;
        }
    }
}

void LoadTracker::checkRequests(uint64_t now) {
    while (!requests.empty() && requests.top().time < now) {
        auto &request = requests.top();
        auto &model = models[request.model_id];
        model.timedout_loadweights += request.loadweights_size;

    	detach(model);
        invalidatePriorities(model);
        distributeLoad(model);

        requests.pop();
    }
}

LoadTracker::LoadTracker(int num_gpus, int num_models, uint64_t capacity) : 
n_models(num_models), n_gpus(num_gpus), capacity(capacity) {
    stale.reserve(num_models);
    gpus.resize(num_gpus);
    for (unsigned i = 0; i < num_gpus; i++) {
        gpus[i].id = i;
        gpus[i].models.resize(num_models, false);
    }

    models.resize(num_models);
    for (unsigned i = 0; i < num_models; i++) {
        auto &model = models[i];
        model.id = i;
        model.gpus.resize(num_gpus, false);
        model.loading.resize(num_gpus, false);
        model.allocations.resize(num_gpus, 0);
        model.last_used.resize(num_gpus, 0);
        for (unsigned i = 0; i < num_gpus; i++) {
            model.last_used[i] = seqno_seed++;
        }

        model.priorities.resize(num_gpus);
        for (unsigned j = 0; j < num_gpus; j++) {
            auto priority = new ModelPriority(&model);
            priority->last_used = model.last_used[j];
            model.priorities[j] = priority;

            gpus[j].not_cached.insert(priority);
        }
    }            
}

int LoadTracker::loadModel(int gpu_id, bool requires_eviction) {
    // Complete any pending requests
    checkRequests(util::now());

    auto &gpu = gpus[gpu_id];

    // Update and re-enqueue all models
    refreshPriorities();
    attach(gpu);

    if (gpu.not_cached.size() == 0) return -1;

    auto &priority = *gpu.not_cached.begin();
    if (priority->is_empty) return -1;
    if (priority <= 0) return -1; // all demand satisfied


    Model &model = *(priority->model);

    detach(model);
    invalidatePriorities(model);
    addGPU(model, gpu);
    distributeLoad(model);

    return model.id;
}

int LoadTracker::evictModel(int gpu_id) {
    // Update and re-enqueue all models
    refreshPriorities();
    attach(gpus[gpu_id]);

    auto &gpu = gpus[gpu_id];
    if (gpu.cached.size() == 0) return -1;

    auto &priority = *gpu.cached.rbegin();
    Model &model = *(priority->model);

    detach(model);
    invalidatePriorities(model);
    removeGPU(model, gpus[gpu_id], true);
    distributeLoad(model);

    return model.id;
}

void LoadTracker::process(ModelLoadTracker* tracker) {
    // Detach the model
    Model& model = models[tracker->model_id];
    detach(model);
    invalidatePriorities(model);

    // Remove pending load demand
    uint64_t now = util::now();
    checkRequests(now);

    // Add new requests
    int64_t loadweights = tracker->new_loadweights.exchange(0);
    int64_t exec = tracker->new_exec.exchange(0);
    if (loadweights > 0) {
        LoadTracker::Request request;
        request.model_id = tracker->model_id;
        request.loadweights_size = tracker->new_loadweights;
        request.time = now + tracker->start_loadweights_by;
        requests.push(request);
    }
    model.outstanding_exec += exec;
    model.outstanding_loadweights += loadweights;

    // Process completed requests
    model.completed_exec += tracker->delta_exec.exchange(0);
    model.completed_loadweights += tracker->delta_loadweights.exchange(0);

    // Update last used for models
    for (int i = 0; i < tracker->touched.size(); i++) {
        if (tracker->touched[i]) {
            model.last_used[i] = seqno_seed++;
            tracker->touched[i] = false;
        }
    }

    // Process loadcomplete events
    ModelLoadTracker::LoadEvent event;
    while (tracker->events.try_pop(event)) {
        if (event.success) {
            addGPUcomplete(model, gpus[event.gpu_id]);
        } else {
            removeGPU(model, gpus[event.gpu_id], false);
        }
    }
    
    // Re-distribute model's load
    distributeLoad(model);
}

}
}
}