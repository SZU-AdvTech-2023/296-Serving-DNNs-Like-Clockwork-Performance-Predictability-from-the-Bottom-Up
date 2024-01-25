#ifndef _CLOCKWORK_TEST_ACTIONS_H_
#define _CLOCKWORK_TEST_ACTIONS_H_

#include <memory>
#include <unistd.h>
#include <libgen.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>
#include "clockwork/task.h"
#include "clockwork/action.h"
#include <catch2/catch.hpp>
#include "clockwork/api/worker_api.h"
#include "clockwork/util.h"
#include "clockwork/test/util.h"
#include "clockwork/model/model.h"
#include "clockwork/worker.h"


using namespace clockwork::model;

namespace clockwork {

class TestAction {
public:
    std::atomic_bool is_success;
    std::atomic_bool is_error;
    int status_code;
    std::string error_message;

    TestAction() : is_success(false), is_error(false) {}

    void setsuccess() {
        is_success = true;
    }

    void seterror(std::shared_ptr<workerapi::ErrorResult> result) {
        is_error = true;
        this->status_code = result->status;
        this->error_message = result->message;
    }

    void await() {
        while ((!is_success) && (!is_error));
    }

    void check_success(bool expect_success, int expected_status_code = 0) {
        if (expect_success) {
            INFO(status_code << ": " << error_message);
            REQUIRE(!is_error);
            REQUIRE(is_success);
        } else {
            REQUIRE(is_error);
            REQUIRE(!is_success);
            INFO(status_code << ": " << error_message);
            REQUIRE(status_code == expected_status_code);
        }
    }
};

class TestLoadModelFromDiskAction : public LoadModelFromDiskAction, public TestAction {
public:
    TestLoadModelFromDiskAction(ClockworkRuntime* runtime, std::shared_ptr<workerapi::LoadModelFromDisk> action) : 
        LoadModelFromDiskAction(runtime, action) {}

    void success(std::shared_ptr<workerapi::LoadModelFromDiskResult> result) {
        setsuccess();
    }

    void error(std::shared_ptr<workerapi::ErrorResult> result) {
        seterror(result);
    }

};

class TestLoadWeightsAction : public LoadWeightsAction, public TestAction {
public:
    TestLoadWeightsAction(ClockworkRuntime* runtime, std::shared_ptr<workerapi::LoadWeights> action) : 
        LoadWeightsAction(runtime, action) {}

    void success(std::shared_ptr<workerapi::LoadWeightsResult> result) {
        setsuccess();
    }

    void error(std::shared_ptr<workerapi::ErrorResult> result) {
        seterror(result);
    }

};


class TestEvictWeightsAction : public EvictWeightsAction, public TestAction {
public:
    TestEvictWeightsAction(ClockworkRuntime* runtime, std::shared_ptr<workerapi::EvictWeights> action) : 
        EvictWeightsAction(runtime, action) {}

    void success(std::shared_ptr<workerapi::EvictWeightsResult> result) {
        setsuccess();
    }

    void error(std::shared_ptr<workerapi::ErrorResult> result) {
        seterror(result);
    }

};


class TestInferAction : public InferAction, public TestAction {
public:
	TestInferAction(ClockworkRuntime* runtime, std::shared_ptr<workerapi::Infer> action) : 
		InferAction(runtime, action) {}

	void success(std::shared_ptr<workerapi::InferResult> result) {
		setsuccess();
	}

	void error(std::shared_ptr<workerapi::ErrorResult> result) {
		seterror(result);
	}

};

class ClockworkRuntimeWrapper : public ClockworkRuntime {
public:
    ~ClockworkRuntimeWrapper() {
        this->shutdown(true);
    }
};



std::shared_ptr<workerapi::LoadModelFromDisk> load_model_from_disk_action();

std::shared_ptr<workerapi::LoadWeights> load_weights_action(int model_id = 0);

std::shared_ptr<workerapi::EvictWeights> evict_weights_action();

std::shared_ptr<workerapi::Infer> infer_action();

std::shared_ptr<workerapi::Infer> infer_action(int batch_size, BatchedModel* model);

std::shared_ptr<workerapi::Infer> infer_action2(ClockworkWorker* worker);

std::shared_ptr<workerapi::GetWorkerState> get_worker_state_action();

}

#endif
