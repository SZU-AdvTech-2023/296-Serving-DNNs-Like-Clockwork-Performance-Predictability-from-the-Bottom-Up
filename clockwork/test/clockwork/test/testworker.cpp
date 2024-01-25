#include <unistd.h>
#include <libgen.h>
#include <fstream>
#include <algorithm>

#include <cuda_runtime.h>
#include "clockwork/api/worker_api.h"
#include "clockwork/test/util.h"
#include "clockwork/model/model.h"
#include "clockwork/worker.h"
#include <catch2/catch.hpp>
#include "clockwork/test/actions.h"
#include "clockwork/test/controller.h"
#include "tbb/concurrent_queue.h"

using namespace clockwork;
using namespace clockwork::model;

TEST_CASE("Test Worker", "[worker]") {
    ClockworkWorker worker;
    TestController controller;
    worker.controller = &controller;

    auto load_model = load_model_from_disk_action();

    std::vector<std::shared_ptr<workerapi::Action>> actions{load_model};

    worker.sendActions(actions);
    controller.expect(actionSuccess);

    worker.shutdown(true);
}

TEST_CASE("Test Infer No Weights", "[worker] [noweights]") {
    ClockworkWorker worker;
    TestController controller;
    worker.controller = &controller;

    auto load_model = load_model_from_disk_action();
    std::vector<std::shared_ptr<workerapi::Action>> actions{load_model};
    worker.sendActions(actions);
    controller.expect(actionSuccess);

    auto infer = infer_action2(&worker);
    actions = {infer};
    worker.sendActions(actions);
    controller.expect(actionErrorModelWeightsNotPresent);

    worker.shutdown(true);
    
}

// // The below test is commented out because it relies on non-deterministic timing
//
// TEST_CASE("Test Infer Weights Not There Yet", "[worker] [noweights]") {
//     ClockworkWorker worker;
//     TestController controller;
//     worker.controller = &controller;

//     auto load_model = load_model_from_disk_action();
//     std::vector<std::shared_ptr<workerapi::Action>> actions{load_model};
//     worker.sendActions(actions);
//     controller.expect(actionSuccess);

//     auto load_weights = load_weights_action();
//     auto infer = infer_action2(&worker);
//     actions = {load_weights, infer};
//     worker.sendActions(actions);
//     controller.expect(actionErrorModelWeightsNotPresent);
//     controller.expect(actionSuccess);

//     worker.shutdown(true);
//     
// }

TEST_CASE("Test Infer Invalid Input", "[worker] [invalid]") {
    ClockworkWorker worker;
    TestController controller;
    worker.controller = &controller;

    auto load_model = load_model_from_disk_action();
    std::vector<std::shared_ptr<workerapi::Action>> actions{load_model};
    worker.sendActions(actions);
    controller.expect(actionSuccess);

    auto infer = infer_action2(&worker);
    infer->input_size = 10;
    infer->input = nullptr;
    actions = {infer};
    worker.sendActions(actions);
    controller.expect(actionErrorInvalidInput);

    worker.shutdown(true);
    
}

TEST_CASE("Test Infer Invalid Input Size", "[worker] [invalid]") {
    ClockworkWorker worker;
    TestController controller;
    worker.controller = &controller;

    auto load_model = load_model_from_disk_action();
    std::vector<std::shared_ptr<workerapi::Action>> actions{load_model};
    worker.sendActions(actions);
    controller.expect(actionSuccess);

    auto infer = infer_action2(&worker);
    infer->input_size = 100;
    infer->input = static_cast<char*>(malloc(100));
    actions = {infer};
    worker.sendActions(actions);
    controller.expect(actionErrorInvalidInput);

    worker.shutdown(true);
    
}

TEST_CASE("Test Worker E2E Simple", "[worker] [e2esimple]") {
    ClockworkWorker worker;
    TestController controller;
    worker.controller = &controller;

    std::vector<std::shared_ptr<workerapi::Action>> actions;

    auto load_model = load_model_from_disk_action();
    actions = {load_model};
    worker.sendActions(actions);
    controller.expect(actionSuccess);

    auto load_weights = load_weights_action();
    actions = {load_weights};
    worker.sendActions(actions);
    controller.expect(actionSuccess);

    auto infer = infer_action2(&worker);
    actions = {infer};
    worker.sendActions(actions);
    controller.expect(actionSuccess);

    auto evict_weights = evict_weights_action();
    actions = {evict_weights};
    worker.sendActions(actions);
    controller.expect(actionSuccess);

    worker.shutdown(true);
    
}

TEST_CASE("Test Worker E2E Timed Success", "[worker]") {
    ClockworkWorker worker;
    TestController controller;
    worker.controller = &controller;

    auto load_model = load_model_from_disk_action();

    auto load_weights = load_weights_action();
    load_weights->earliest = load_model->earliest + 1000000000UL;
    load_weights->latest = load_weights->earliest + 100000000UL;

    auto infer = infer_action2(&worker);
    infer->earliest = load_weights->earliest + 20000000;
    infer->latest = infer->earliest + 100000000UL;

    auto evict_weights = evict_weights_action();
    evict_weights->earliest = infer->earliest + 10000000;
    evict_weights->latest = evict_weights->earliest + 100000000UL;

    std::vector<std::shared_ptr<workerapi::Action>> actions{load_model, load_weights, infer, evict_weights};
    worker.sendActions(actions);

    for (unsigned i = 0; i < 4; i++) {
        controller.expect(actionSuccess);
    }

    worker.shutdown(true);
    
}

TEST_CASE("Test GetWorkerState", "[worker] [e2esimple]") {
	ClockworkWorker worker;
	TestController controller;
	worker.controller = &controller;

	std::vector<std::shared_ptr<workerapi::Action>> actions;

	auto load_model = load_model_from_disk_action();
	actions = {load_model};
	worker.sendActions(actions);
	controller.expect(actionSuccess);

	auto get_worker_state_1 = get_worker_state_action();
	actions = {get_worker_state_1};
	worker.sendActions(actions);
	controller.expect(actionSuccess);

	auto load_weights = load_weights_action();
	actions = {load_weights};
	worker.sendActions(actions);
	controller.expect(actionSuccess);

	auto infer = infer_action2(&worker);
	actions = {infer};
	worker.sendActions(actions);
	controller.expect(actionSuccess);

	auto get_worker_state_2 = get_worker_state_action();
	actions = {get_worker_state_2};
	worker.sendActions(actions);
	controller.expect(actionSuccess);

	auto evict_weights = evict_weights_action();
	actions = {evict_weights};
	worker.sendActions(actions);
	controller.expect(actionSuccess);

	auto get_worker_state_3 = get_worker_state_action();
	actions = {get_worker_state_3};
	worker.sendActions(actions);
	controller.expect(actionSuccess);

	worker.shutdown(true);

}

TEST_CASE("Test Adjust Timestamp", "[adjust]") {
    uint64_t timestamp;
    int64_t delta;
    uint64_t expect;

    timestamp = 0;
    delta = -1;
    expect = 0;
    CHECK(adjust_timestamp(timestamp, delta) == expect);

    timestamp = 1;
    delta = -1;
    expect = 0;
    CHECK(adjust_timestamp(timestamp, delta) == expect);

    timestamp = 2;
    delta = -1;
    expect = 1;
    CHECK(adjust_timestamp(timestamp, delta) == expect);

    timestamp = 100;
    delta = -20;
    expect = 80;
    CHECK(adjust_timestamp(timestamp, delta) == expect);

    timestamp = 0;
    delta = 1;
    expect = 1;
    CHECK(adjust_timestamp(timestamp, delta) == expect);

    timestamp = 1;
    delta = 1;
    expect = 2;
    CHECK(adjust_timestamp(timestamp, delta) == expect);

    timestamp = 2;
    delta = 1;
    expect = 3;
    CHECK(adjust_timestamp(timestamp, delta) == expect);

    timestamp = 100;
    delta = 20;
    expect = 120;
    CHECK(adjust_timestamp(timestamp, delta) == expect);

    timestamp = UINT64_MAX;
    delta = 0;
    expect = UINT64_MAX;
    CHECK(adjust_timestamp(timestamp, delta) == expect);

    timestamp = UINT64_MAX;
    delta = -1;
    expect = UINT64_MAX-1;
    CHECK(adjust_timestamp(timestamp, delta) == expect);

    timestamp = UINT64_MAX;
    delta = -100;
    expect = UINT64_MAX-100;
    CHECK(adjust_timestamp(timestamp, delta) == expect);

    timestamp = UINT64_MAX;
    delta = 1;
    expect = UINT64_MAX;
    CHECK(adjust_timestamp(timestamp, delta) == expect);

    timestamp = UINT64_MAX;
    delta = 10;
    expect = UINT64_MAX;
    CHECK(adjust_timestamp(timestamp, delta) == expect);

    timestamp = UINT64_MAX;
    delta = 100;
    expect = UINT64_MAX;
    CHECK(adjust_timestamp(timestamp, delta) == expect);
}

