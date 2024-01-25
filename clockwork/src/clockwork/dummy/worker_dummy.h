#ifndef _CLOCKWORK_WORKER_DUMMY_H_
#define _CLOCKWORK_WORKER_DUMMY_H_

#include <atomic>
#include "clockwork/api/worker_api.h"
#include "tbb/concurrent_priority_queue.h"
#include "clockwork/dummy/action_dummy.h"
#include "clockwork/config.h"

/*
This file ties together the worker API (defined in api/worker_api.h) with model actions (defined in action.h)
using a clockwork scheduling framework (defined in runtime.h).
*/

namespace clockwork {

class ExecutorDummy;

struct element {
    uint64_t ready;
    std::function<void(void)> callback;
    std::function<void(void)> defaultfunc;

    friend bool operator < (const element& lhs, const element &rhs) {
        return lhs.ready < rhs.ready;
    }
    friend bool operator > (const element& lhs, const element &rhs) {
        return lhs.ready > rhs.ready;
    }
};

class EngineDummy{

public:
    std::atomic_bool alive;
    std::thread run_thread;
    std::vector<ExecutorDummy*> executors;
    std::vector<element*> infers_to_end;// num_gpus
    std::vector<element*> loads_to_end;// num_gpus


    EngineDummy(unsigned num_gpus);
    void addExecutor(ExecutorDummy* executor);
    void addToEnd(int type, unsigned gpu_id, element* action);
    void startEngine();
    void run();    
    void shutdown(){alive.store(false);}
    void join(){run_thread.join();}

};

class ExecutorDummy{
public:
    int type;
    unsigned gpu_id;
    tbb::concurrent_priority_queue<element, std::greater<element>> actions_to_start;//queue sorted by earlist

    EngineDummy* myEngine;
    MemoryManagerDummy* myManager;
    workerapi::Controller*  myController;

    ExecutorDummy(int Type,unsigned gpuNumber, EngineDummy* engine,  MemoryManagerDummy* manager) : type(Type),gpu_id(gpuNumber),myManager(manager),myEngine(engine){};

    void new_action(std::shared_ptr<workerapi::LoadModelFromDisk> action);
    void new_action(std::shared_ptr<workerapi::LoadWeights> action);
    void new_action(std::shared_ptr<workerapi::EvictWeights> action);
    void new_action(std::shared_ptr<workerapi::Infer> action);

    void setController(workerapi::Controller* Controller){ myController = Controller;};

};


class ClockworkRuntimeDummy {
public:
    unsigned num_gpus;
    MemoryManagerDummy* manager;
    EngineDummy* engine;    // Type 3

    ExecutorDummy* load_model_executor; // Type 0
    std::vector<ExecutorDummy*> gpu_executors;  // Type 3
    std::vector<ExecutorDummy*> weights_executors;  // Type 1
    std::vector<ExecutorDummy*> outputs_executors;  // Type 4

    ClockworkRuntimeDummy() {
        ClockworkWorkerConfig config;
        initialize(config);
    }

    ClockworkRuntimeDummy(ClockworkWorkerConfig &config) {
        initialize(config);
    }

    virtual ~ClockworkRuntimeDummy() {
        delete manager;
        delete engine;
        delete load_model_executor;

        for (unsigned gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
            delete gpu_executors[gpu_id];
            delete weights_executors[gpu_id];
            delete outputs_executors[gpu_id];
        }
    }

    void shutdown(bool await_completion);

    void join();

    void setController(workerapi::Controller* Controller);

protected:


    void initialize(ClockworkWorkerConfig &config) {

        num_gpus = config.num_gpus; 

        manager = new MemoryManagerDummy(config);

        engine = new EngineDummy(num_gpus);

        load_model_executor = new ExecutorDummy( workerapi::loadModelFromDiskAction, 0, engine, manager);
        engine->addExecutor(load_model_executor);

        for (unsigned gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
            gpu_executors.push_back(new ExecutorDummy( workerapi::inferAction, gpu_id, engine,manager));
            weights_executors.push_back(new ExecutorDummy( workerapi::loadWeightsAction, gpu_id, engine,manager));
            outputs_executors.push_back(new ExecutorDummy( workerapi::evictWeightsAction, gpu_id, engine,manager));
            engine->addExecutor(gpu_executors[gpu_id]);
            engine->addExecutor(weights_executors[gpu_id]);
            engine->addExecutor(outputs_executors[gpu_id]);
        }
        engine->startEngine();
    }

};

class ClockworkDummyWorker : public workerapi::Worker {
public:
    ClockworkRuntimeDummy* runtime;// something that keeps records of gpus
    workerapi::Controller* controller;

    //Toy worker 
    ClockworkDummyWorker():runtime(new ClockworkRuntimeDummy()){};
    ClockworkDummyWorker(ClockworkWorkerConfig &config):runtime(new ClockworkRuntimeDummy(config)){};
    ~ClockworkDummyWorker(){
        this->shutdown(false);
        delete runtime;
    };
    void sendActions(std::vector<std::shared_ptr<workerapi::Action>> &actions);
    void shutdown(bool await_completion){
        runtime->shutdown(false);
        if (await_completion) {
            join();
        }
    }
    void join(){runtime->join();}
    void setController(workerapi::Controller* Controller);

private:
    void invalidAction(std::shared_ptr<workerapi::Action> action);
    void loadModel(std::shared_ptr<workerapi::Action> action);
    void loadWeights(std::shared_ptr<workerapi::Action> action);
    void evictWeights(std::shared_ptr<workerapi::Action> action);
    void infer(std::shared_ptr<workerapi::Action> action);
    void clearCache(std::shared_ptr<workerapi::Action> action);
    void getWorkerState(std::shared_ptr<workerapi::Action> action);
};

class LoadModelFromDiskDummy : public LoadModelFromDiskDummyAction{
public:
    EngineDummy* myEngine;
    workerapi::Controller* myController;

    LoadModelFromDiskDummy( MemoryManagerDummy* Manager,EngineDummy* Engine,std::shared_ptr<workerapi::LoadModelFromDisk> LoadModel, workerapi::Controller* Controller);

    void success(std::shared_ptr<workerapi::LoadModelFromDiskResult> result);
    void error(int status_code, std::string message);
};

class LoadWeightsDummy : public LoadWeightsDummyAction{
public:
    EngineDummy* myEngine;
    workerapi::Controller*  myController;

    LoadWeightsDummy(MemoryManagerDummy* Manager, EngineDummy* Engine,std::shared_ptr<workerapi::LoadWeights> LoadWeights, workerapi::Controller* Controller);

    void toComplete();
    void success(std::shared_ptr<workerapi::LoadWeightsResult> result);
    void error(int status_code, std::string message);
};

class EvictWeightsDummy : public EvictWeightsDummyAction{
public:
    EngineDummy* myEngine;
    workerapi::Controller*  myController;

    EvictWeightsDummy(MemoryManagerDummy* Manager,EngineDummy* Engine,std::shared_ptr<workerapi::EvictWeights> EvictWeights, workerapi::Controller* Controller);

    void success(std::shared_ptr<workerapi::EvictWeightsResult> result);
    void error(int status_code, std::string message);
};

class InferDummy : public InferDummyAction{
public:
    EngineDummy* myEngine;
    workerapi::Controller*  myController;

    InferDummy( MemoryManagerDummy* Manager,EngineDummy* Engine,std::shared_ptr<workerapi::Infer> Infer,workerapi::Controller* Controller);

    void toComplete();
    void success(std::shared_ptr<workerapi::InferResult> result);
    void error(int status_code, std::string message);
};


uint64_t adjust_timestamp_dummy(uint64_t timestamp, int64_t clock_delta);
}

#endif