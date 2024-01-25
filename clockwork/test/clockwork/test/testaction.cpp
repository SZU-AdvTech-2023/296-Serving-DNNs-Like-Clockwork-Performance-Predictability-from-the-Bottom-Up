#include <unistd.h>
#include <libgen.h>
#include <fstream>
#include <algorithm>

#include <cuda_runtime.h>
#include "clockwork/api/worker_api.h"
#include "clockwork/test/util.h"
#include "clockwork/model/model.h"
#include "clockwork/task.h"
#include "clockwork/action.h"
#include <catch2/catch.hpp>
#include "clockwork/test/actions.h"

using namespace clockwork;
using namespace clockwork::model;

std::shared_ptr<ClockworkRuntime> make_runtime() {
    return std::make_shared<ClockworkRuntimeWrapper>();
}

BatchedModel* make_model_for_action() {
    std::string f = clockwork::util::get_example_model();

    Model* model = Model::loadFromDisk(f+".1.so", f+".1.clockwork", f+".clockwork_params", GPU_ID_0);

    std::vector<std::pair<unsigned, Model*>> models = {{1, model}};
    BatchedModel* batched = new BatchedModel(model->weights_size, model->weights_pinned_host_memory, models, GPU_ID_0);

    batched->instantiate_models_on_host();
    batched->instantiate_models_on_device();
    return batched;
}

TEST_CASE("Load Model From Disk Action", "[action] [loadmodel_action]") {
    auto clockwork = make_runtime();

    TestLoadModelFromDiskAction load_model(clockwork.get(), load_model_from_disk_action());

    load_model.submit();
    load_model.await();
    load_model.check_success(true);
}

TEST_CASE("Load Model From Disk Action Multiple", "[action] [loadmodel_action]") {
    auto clockwork = make_runtime();

    for (unsigned i = 0; i < 10; i++) {
        auto action = load_model_from_disk_action();
        action->model_id = i;

        TestLoadModelFromDiskAction load_model(clockwork.get(), action);

        load_model.submit();
        load_model.await();
        load_model.check_success(true);
    }

    
}

TEST_CASE("Load Weights Action", "[action] [loadweights_action]") {
    BatchedModel* model = make_model_for_action();
    auto clockwork = make_runtime();
    clockwork->manager->models->put(0, GPU_ID_0, new RuntimeModel(model, GPU_ID_0));

    TestLoadWeightsAction load_weights(clockwork.get(), load_weights_action());

    load_weights.submit();
    load_weights.await();
    load_weights.check_success(true);    
}

TEST_CASE("Load Weights Action Multiple", "[action] [loadweights_multiple]") {
    BatchedModel* model = make_model_for_action();
    auto clockwork = make_runtime();
    clockwork->manager->models->put(0, GPU_ID_0, new RuntimeModel(model, GPU_ID_0));

    for (unsigned i = 0; i < 5; i++) {
        TestLoadWeightsAction load_weights(clockwork.get(), load_weights_action());

        load_weights.submit();
        load_weights.await();
        load_weights.check_success(true);
    }
}

TEST_CASE("Load Weights Action Invalid Model", "[action]") {
    BatchedModel* model = make_model_for_action();
    auto clockwork = make_runtime();

    TestLoadWeightsAction load_weights(clockwork.get(), load_weights_action());

    load_weights.submit();
    load_weights.await();
    load_weights.check_success(false, actionErrorUnknownModel);
}

TEST_CASE("Load Evict Weights Action", "[action] [evict_action]") {
    BatchedModel* model = make_model_for_action();
    auto clockwork = make_runtime();
    clockwork->manager->models->put(0, GPU_ID_0, new RuntimeModel(model, GPU_ID_0));

    TestLoadWeightsAction load_weights(clockwork.get(), load_weights_action());

    load_weights.submit();
    load_weights.await();
    load_weights.check_success(true);

    TestEvictWeightsAction evict_weights(clockwork.get(), evict_weights_action());

    evict_weights.submit();
    evict_weights.await();
    evict_weights.check_success(true);
}

TEST_CASE("Evict without Weights Action", "[action] [evict_action]") {
    BatchedModel* model = make_model_for_action();
    auto clockwork = make_runtime();
    clockwork->manager->models->put(0, GPU_ID_0, new RuntimeModel(model, GPU_ID_0));

    TestEvictWeightsAction evict_weights(clockwork.get(), evict_weights_action());

    evict_weights.submit();
    evict_weights.await();
    evict_weights.check_success(false, actionErrorModelWeightsNotPresent);
}

TEST_CASE("Infer Action", "[action] [infer_action]") {
    BatchedModel* model = make_model_for_action();
    auto clockwork = make_runtime();
    clockwork->manager->models->put(0, GPU_ID_0, new RuntimeModel(model, GPU_ID_0));

    TestLoadWeightsAction load_weights(clockwork.get(), load_weights_action());

    load_weights.submit();
    load_weights.await();
    load_weights.check_success(true);

    TestInferAction infer(clockwork.get(), infer_action(1, model));

    infer.submit();
    infer.await();
    infer.check_success(true);
}

TEST_CASE("Infer Action Wrong Input Size", "[action] [nomodel]") {
    BatchedModel* model = make_model_for_action();
    auto clockwork = make_runtime();

    auto action = infer_action();
    action->input_size = 10;
    TestInferAction infer(clockwork.get(), action);

    infer.submit();
    infer.await();
    infer.check_success(false, actionErrorUnknownModel);
}

TEST_CASE("Infer Action Nonexistent Model", "[action] [nomodel]") {
    BatchedModel* model = make_model_for_action();
    auto clockwork = make_runtime();

    TestInferAction infer(clockwork.get(), infer_action());

    infer.submit();
    infer.await();
    infer.check_success(false, actionErrorUnknownModel);
}

TEST_CASE("Infer Action Nonexistent Weights", "[action] [noweights]") {
    BatchedModel* model = make_model_for_action();
    auto clockwork = make_runtime();

    TestLoadModelFromDiskAction load_model(clockwork.get(), load_model_from_disk_action());

    load_model.submit();
    load_model.await();
    load_model.check_success(true);

    TestInferAction infer(clockwork.get(), infer_action());

    infer.submit();
    infer.await();
    infer.check_success(false, actionErrorModelWeightsNotPresent);
}

TEST_CASE("Infer Action Multiple", "[action] [infer_action_multiple]") {
    BatchedModel* model = make_model_for_action();
    auto clockwork = make_runtime();
    clockwork->manager->models->put(0, GPU_ID_0, new RuntimeModel(model, GPU_ID_0));

    TestLoadWeightsAction load_weights(clockwork.get(), load_weights_action());

    load_weights.submit();
    load_weights.await();
    load_weights.check_success(true);

    for (unsigned i = 0; i < 2; i++) {
        TestInferAction infer(clockwork.get(), infer_action(1, model));

        infer.submit();
        infer.await();
        infer.check_success(true);
    }
}

TEST_CASE("Make Many Models", "[action] [models]") {
    std::vector<BatchedModel*> models;
    for (unsigned i = 0; i < 30; i++) {
        models.push_back(make_model_for_action());
    }
    for (BatchedModel* model : models) {
        delete model;
    }
}

TEST_CASE("Infer Action Multiple Concurrent", "[action] [infer_action_concurrent]") {
    BatchedModel* model = make_model_for_action();
    auto clockwork = make_runtime();
    clockwork->manager->models->put(0, GPU_ID_0, new RuntimeModel(model, GPU_ID_0));

    TestLoadWeightsAction load_weights(clockwork.get(), load_weights_action());

    load_weights.submit();
    load_weights.await();
    load_weights.check_success(true);

    std::vector<std::shared_ptr<TestInferAction>> infers;

    for (unsigned i = 0; i < 2; i++) {
        infers.push_back(std::make_shared<TestInferAction>(clockwork.get(), infer_action(1, model)));
    }

    for (auto infer : infers) {
        infer->submit();
    }

    for (auto infer : infers) {
        infer->await();
        infer->check_success(true);
    }    
}

TEST_CASE("Infer after Evict Action", "[action] [evict_action]") {
    BatchedModel* model = make_model_for_action();
    auto clockwork = make_runtime();
    clockwork->manager->models->put(0, GPU_ID_0, new RuntimeModel(model, GPU_ID_0));

    TestLoadWeightsAction load_weights(clockwork.get(), load_weights_action());

    load_weights.submit();
    load_weights.await();
    load_weights.check_success(true);
    
    TestInferAction infer(clockwork.get(), infer_action(1, model));

    infer.submit();
    infer.await();
    infer.check_success(true);

    TestEvictWeightsAction evict_weights(clockwork.get(), evict_weights_action());

    evict_weights.submit();
    evict_weights.await();
    evict_weights.check_success(true);

    TestInferAction infer2(clockwork.get(), infer_action(1, model));

    infer2.submit();
    infer2.await();
    infer2.check_success(false, actionErrorModelWeightsNotPresent);   
}

TEST_CASE("Actions E2E", "[action] [e2e]") {
    auto clockwork = make_runtime();

    auto load_model = new TestLoadModelFromDiskAction(clockwork.get(), load_model_from_disk_action());

    load_model->submit();
    load_model->await();
    load_model->check_success(true);

    delete load_model;

    auto load_weights = new TestLoadWeightsAction(clockwork.get(), load_weights_action());

    load_weights->submit();
    load_weights->await();
    load_weights->check_success(true);

    delete load_weights;
    
    BatchedModel* model = clockwork->manager->models->get(0, GPU_ID_0)->model;
    auto infer = new TestInferAction(clockwork.get(), infer_action(1, model));

    infer->submit();
    infer->await();
    infer->check_success(true);

    delete infer;

    auto evict_weights = new TestEvictWeightsAction(clockwork.get(), evict_weights_action());

    evict_weights->submit();
    evict_weights->await();
    evict_weights->check_success(true);

    delete evict_weights;

    auto infer2 = new TestInferAction(clockwork.get(), infer_action(1, model));

    infer2->submit();
    infer2->await();
    infer2->check_success(false, actionErrorModelWeightsNotPresent);

    delete infer2;

    
}

TEST_CASE("Task Cancelled After Shutdown", "[action] [shutdown]") {
    BatchedModel* model = make_model_for_action();
    auto clockwork = make_runtime();
    clockwork->manager->models->put(0, GPU_ID_0, new RuntimeModel(model, GPU_ID_0));

    auto action = load_weights_action();
    action->earliest = util::now() + 1000000000UL; // + 10seconds
    action->latest = action->earliest;

    auto load_weights = new TestLoadWeightsAction(clockwork.get(), action);

    load_weights->submit();

    clockwork = nullptr; // destroys the runtime

    load_weights->await();
    load_weights->check_success(false, actionCancelled);

    delete load_weights;
}

class TestLoadModelFromDiskActionThatDeletesItself : public LoadModelFromDiskAction {
public:
    TestAction &action_status;

    TestLoadModelFromDiskActionThatDeletesItself(
            ClockworkRuntime* runtime, 
            std::shared_ptr<workerapi::LoadModelFromDisk> action, 
            TestAction &action_status) : 
        LoadModelFromDiskAction(runtime, action), action_status(action_status) {}

    void success(std::shared_ptr<workerapi::LoadModelFromDiskResult> result) {
        action_status.setsuccess();
        delete this;
    }

    void error(std::shared_ptr<workerapi::ErrorResult> result) {
        action_status.seterror(result);
        delete this;
    }

};

TEST_CASE("Task Action That Deletes Itself in Callback", "[action] [shutdown]") {
    auto clockwork = make_runtime();

    TestAction action_status;

    auto action = load_model_from_disk_action();
    action->model_path = "";
    auto load_model = new TestLoadModelFromDiskActionThatDeletesItself(
            clockwork.get(), action, action_status);

    load_model->submit();
    action_status.await();
    action_status.check_success(false, actionErrorInvalidModelPath);

    
}
