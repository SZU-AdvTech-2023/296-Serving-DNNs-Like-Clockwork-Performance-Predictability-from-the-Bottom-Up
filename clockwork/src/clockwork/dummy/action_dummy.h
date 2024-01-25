#ifndef _CLOCKWORK_ACTION_DUMMY_H_
#define _CLOCKWORK_ACTION_DUMMY_H_

#include <atomic>
#include "clockwork/task.h"
#include "clockwork/api/worker_api.h"
#include "clockwork/dummy/memory_dummy.h"

/*
This file ties together the worker API (defined in api/worker_api.h) with model actions (defined in action.h)
using a clockwork scheduling framework (defined in runtime.h).
*/

namespace clockwork {

class LoadModelFromDiskDummyAction{
public:
    MemoryManagerDummy* myManager;
    std::shared_ptr<workerapi::LoadModelFromDisk> loadmodel;

    uint64_t start = 0;
    uint64_t end = 0;

    LoadModelFromDiskDummyAction(MemoryManagerDummy* Manager,std::shared_ptr<workerapi::LoadModelFromDisk> LoadModel):myManager(Manager),loadmodel(LoadModel){};
    void run();

    virtual void success(std::shared_ptr<workerapi::LoadModelFromDiskResult> result) = 0;
    virtual void error(int status_code, std::string message) = 0;

};

class LoadWeightsDummyAction{
public:
    MemoryManagerDummy* myManager;
    std::shared_ptr<workerapi::LoadWeights> loadweights;

    int version;
    uint64_t start = 0;
    uint64_t end = 0;

    LoadWeightsDummyAction( MemoryManagerDummy* Manager,std::shared_ptr<workerapi::LoadWeights> LoadWeights):myManager(Manager),loadweights(LoadWeights){version = 0;};
    void run();
    void process_completion();

    virtual void toComplete() = 0;
    virtual void success(std::shared_ptr<workerapi::LoadWeightsResult> result) = 0;
    virtual void error(int status_code, std::string message) = 0;
};

class EvictWeightsDummyAction{
public:
    MemoryManagerDummy* myManager;
    std::shared_ptr<workerapi::EvictWeights> evictweights;

    uint64_t start = 0;
    uint64_t end = 0;

    EvictWeightsDummyAction( MemoryManagerDummy* Manager,std::shared_ptr<workerapi::EvictWeights> EvictWeights):myManager(Manager),evictweights(EvictWeights){};
    void run();

    virtual void success(std::shared_ptr<workerapi::EvictWeightsResult> result) = 0;
    virtual void error(int status_code, std::string message) = 0;
};

class InferDummyAction{
public:
    MemoryManagerDummy* myManager;
    std::shared_ptr<workerapi::Infer> infer;

    int version; 
    uint64_t start = 0;
    uint64_t end = 0;

    InferDummyAction( MemoryManagerDummy* Manager,std::shared_ptr<workerapi::Infer> Infer):myManager(Manager),infer(Infer){version = 0;};
    void run();
    void process_completion();

    virtual void toComplete() = 0;
    virtual void success(std::shared_ptr<workerapi::InferResult> result) = 0;
    virtual void error(int status_code, std::string message) = 0;
};

}

#endif