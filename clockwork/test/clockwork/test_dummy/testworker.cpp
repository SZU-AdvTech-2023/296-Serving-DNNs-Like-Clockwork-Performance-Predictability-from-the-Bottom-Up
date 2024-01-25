#include <unistd.h>
#include <libgen.h>
#include <fstream>
#include <iostream>
#include <algorithm>

#include "clockwork/api/worker_api.h"
#include "clockwork/test/util.h"
#include "clockwork/modeldef.h"
#include "clockwork/test_dummy/actions.h"
#include "clockwork/dummy/worker_dummy.h"
#include "clockwork/test/controller.h"
#include "tbb/concurrent_queue.h"

TEST_CASE("Test Worker Dummy", "[worker] [dummy]") {
    ClockworkWorkerConfig config("");
    ClockworkDummyWorker worker(config);
    TestController controller;
    worker.setController(&controller);

    auto load_model = load_model_from_disk_action();

    std::vector<std::shared_ptr<workerapi::Action>> actions{load_model};

    worker.sendActions(actions);
    controller.expect(actionSuccess);

    worker.shutdown(true);
}

TEST_CASE("Test Infer No Weights Dummy", "[worker] [noweights] [dummy]") {
    ClockworkWorkerConfig config("");
    ClockworkDummyWorker worker(config);
    TestController controller;
    worker.setController(&controller);

    auto load_model = load_model_from_disk_action();
    std::vector<std::shared_ptr<workerapi::Action>> actions{load_model};
    worker.sendActions(actions);
    controller.expect(actionSuccess);

    auto infer = infer_action();
    actions = {infer};
    worker.sendActions(actions);
    controller.expect(execWeightsMissing);

    worker.shutdown(true);
    
}

TEST_CASE("Test Infer Invalid Input Size Dummy", "[worker] [invalid] [dummy]") {
    ClockworkWorkerConfig config("");
    ClockworkDummyWorker worker(config);
    TestController controller;
    worker.setController(&controller);

    auto load_model = load_model_from_disk_action();
    std::vector<std::shared_ptr<workerapi::Action>> actions{load_model};
    worker.sendActions(actions);
    controller.expect(actionSuccess);

    auto infer = infer_action();
    infer->input_size = 100;
    infer->input = static_cast<char*>(malloc(100));
    actions = {infer};
    worker.sendActions(actions);
    controller.expect(copyInputInvalidInput);

    worker.shutdown(true);
    
}


TEST_CASE("Test Worker E2E Simple Dummy", "[worker] [e2esimple] [dummy]") {
    ClockworkWorkerConfig config("");
    ClockworkDummyWorker worker(config);
    TestController controller;
    worker.setController(&controller);

    std::vector<std::shared_ptr<workerapi::Action>> actions;

    auto load_model = load_model_from_disk_action();
    actions = {load_model};
    worker.sendActions(actions);
    controller.expect(actionSuccess);

    auto load_weights = load_weights_action();
    actions = {load_weights};
    worker.sendActions(actions);
    controller.expect(actionSuccess);

    auto infer = infer_action();
    actions = {infer};
    worker.sendActions(actions);
    controller.expect(actionSuccess);

    auto evict_weights = evict_weights_action();
    actions = {evict_weights};
    worker.sendActions(actions);
    controller.expect(actionSuccess);

    worker.shutdown(true);
    
}

TEST_CASE("Test Worker E2E Timed Success Dummy", "[worker] [dummy]") {
    ClockworkWorkerConfig config("");
    ClockworkDummyWorker worker(config);
    TestController controller;
    worker.setController(&controller);

    auto load_model = load_model_from_disk_action();

    auto load_weights = load_weights_action();
    load_weights->earliest = load_model->earliest + 1000000000UL;
    load_weights->latest = load_weights->earliest + 100000000UL;

    auto infer = infer_action();
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

TEST_CASE("Test GetWorkerState Dummy", "[worker] [e2esimple] [dummy]") {
    ClockworkWorkerConfig config("");
    ClockworkDummyWorker worker(config);
    TestController controller;
    worker.setController(&controller);

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

    auto infer = infer_action();
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
