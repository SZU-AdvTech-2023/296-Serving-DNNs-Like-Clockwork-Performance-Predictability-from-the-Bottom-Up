#ifndef _CLOCKWORK_TEST_ACTIONS_DUMMY_H_
#define _CLOCKWORK_TEST_ACTIONS_DUMMY_H_

#include <chrono>
#include <thread>
#include "clockwork/api/worker_api.h"
#include "clockwork/test/actions.h"
#include "clockwork/dummy/action_dummy.h"
#include "clockwork/dummy/worker_dummy.h"


namespace clockwork {

class TestLoadModelFromDiskDummy : public LoadModelFromDiskDummyAction, public TestAction {
public:

    ClockworkRuntimeDummy* myRuntime;

    TestLoadModelFromDiskDummy(ClockworkRuntimeDummy* runtime, std::shared_ptr<workerapi::LoadModelFromDisk> action) : 
        LoadModelFromDiskDummyAction(runtime->manager, action) {myRuntime = runtime;}

    void submit();

    void success(std::shared_ptr<workerapi::LoadModelFromDiskResult> result);
    void error(int status_code, std::string message);

};

class TestLoadWeightsDummy : public LoadWeightsDummyAction, public TestAction {
public:

    ClockworkRuntimeDummy* myRuntime;

    TestLoadWeightsDummy(ClockworkRuntimeDummy* runtime, std::shared_ptr<workerapi::LoadWeights> action) : 
        LoadWeightsDummyAction(runtime->manager, action) {myRuntime = runtime;}

    void submit();
    void toComplete();

    void success(std::shared_ptr<workerapi::LoadWeightsResult> result);
    void error(int status_code, std::string message);
};


class TestEvictWeightsDummy : public EvictWeightsDummyAction, public TestAction {
public:

    ClockworkRuntimeDummy* myRuntime;

    TestEvictWeightsDummy(ClockworkRuntimeDummy* runtime, std::shared_ptr<workerapi::EvictWeights> action) : 
        EvictWeightsDummyAction(runtime->manager, action) {myRuntime = runtime;}

    void submit();

    void success(std::shared_ptr<workerapi::EvictWeightsResult> result);
    void error(int status_code, std::string message);
};


class TestInferDummy : public InferDummyAction, public TestAction {
public:

    ClockworkRuntimeDummy* myRuntime;

	TestInferDummy(ClockworkRuntimeDummy* runtime, std::shared_ptr<workerapi::Infer> action) : 
		InferDummyAction(runtime->manager, action) {myRuntime = runtime;}

    void submit();
    void toComplete();
    
	void success(std::shared_ptr<workerapi::InferResult> result);
    void error(int status_code, std::string message);

};

class ClockworkRuntimeWrapperDummy : public ClockworkRuntimeDummy {
public:
    ~ClockworkRuntimeWrapperDummy() {
        this->shutdown(true);
    }
};

std::shared_ptr<workerapi::Infer> infer_action_dummy(int batch_size, RuntimeModelDummy* model);

}

#endif