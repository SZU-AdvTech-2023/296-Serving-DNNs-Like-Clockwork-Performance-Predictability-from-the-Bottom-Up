#ifndef _CLOCKWORK_CONTROLLER_LOAD_TRACKER_H_
#define _CLOCKWORK_CONTROLLER_LOAD_TRACKER_H_

#include <vector>
#include <set>
#include <queue>
#include "tbb/mutex.h"
#include "tbb/queuing_mutex.h"

namespace clockwork {

class LoadTracker {
 public:

    uint64_t last_print;
    uint64_t print_every = 1000000000UL;
    struct Demand {
        int model_id;
        int64_t exec_size;
        int64_t loadweights_size;
    };

 private:
    const int64_t capacity; // For now just use the slo
    struct ModelPriority;
    struct Model {
        int id;
        int gpu_count = 0;
        std::vector<bool> gpus;
        std::vector<bool> loading;

        int64_t outstanding_exec = 0;
        int64_t outstanding_loadweights = 0;

        int64_t completed_exec = 0;
        int64_t completed_loadweights = 0;
        int64_t timedout_loadweights = 0;

        std::vector<uint64_t> allocations;
        std::vector<ModelPriority*> priorities;
        std::vector<uint64_t> last_used;

        bool stale = false;
    };

    struct ModelPriority {
        bool detached = false;
        int64_t priority = 0;
        int preference = 0;
        bool is_empty = true;
        uint64_t last_used = 0;
        Model* model;
        ModelPriority(Model* model) : model(model) {}
    };

    struct CompareModelPriority {
        bool operator() (const ModelPriority* a, const ModelPriority* b) const {
            if (a->is_empty && b->is_empty) {
                return a->last_used > b->last_used;
            } else if (!a->is_empty && !b->is_empty) {
                if (a->priority == b->priority) {
                    return a->last_used > b->last_used;
                } else {
                    return a->priority > b->priority;
                }
            } else {
                return b->is_empty;
            }
        }
    } sort_by_priority;

    struct GPU {
        int id;
        int64_t outstanding = 1000000UL; // always assume 1ms outstanding work
        double weight = 0.01;
        std::vector<bool> models;
        std::set<ModelPriority*, CompareModelPriority> cached;
        std::set<ModelPriority*, CompareModelPriority> not_cached;
        std::vector<ModelPriority*> detached;
    };

    struct Request {
        int model_id;
        int64_t loadweights_size;
        uint64_t time;

        friend bool operator < (const Request& lhs, const Request &rhs) {
            return lhs.time < rhs.time;
        }
        friend bool operator > (const Request& lhs, const Request &rhs) {
            return lhs.time > rhs.time;
        }
    };

    uint64_t seqno_seed = 0;
    std::vector<Model> models;
    std::vector<GPU> gpus;
    std::vector<Model*> stale;
    const unsigned n_models;
    const unsigned n_gpus;

    std::priority_queue<Request, std::vector<Request>, std::greater<Request>> requests;

    void attach(GPU &gpu);
    void detach(Model &model);

    void invalidatePriorities(Model &model);
    void refreshPriorities();

    void updatePriority(Model &model);
    void clearLoad(Model &model);
    void distributeLoad(Model &model);
    void addGPU(Model &model, GPU &gpu);
    void addGPUcomplete(Model &model, GPU &gpu);
    void removeGPU(Model &model, GPU &gpu, bool evicted);
    void checkRequests(uint64_t now);

 public:
    tbb::queuing_mutex load_mutex;
    tbb::queuing_mutex mutex;

    LoadTracker(int num_gpus, int num_models, uint64_t capacity);
    Demand addRequest(int model_id, int64_t size, uint64_t start_exec_by, uint64_t start_loadweights_by);
    void requestExecuting(Demand &demand, int gpu_id);
    void requestsExecuting(std::vector<Demand*> &demands, int gpu_id);
    void requestCompleted(Demand &demand, int gpu_id);
    void requestsCompleted(std::vector<Demand*> &demands, int gpu_id);
    void requestCancelled(Demand &demand);
    void requestsCancelled(std::vector<Demand*> &demands);
    int loadModel(int gpu_id, bool requires_eviction = false);
    void loadModelComplete(int gpu_id, int model_id, bool success);
    int evictModel(int gpu_id);
};

}

#endif