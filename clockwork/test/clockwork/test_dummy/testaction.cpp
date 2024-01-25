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
#include "tbb/concurrent_queue.h"

using namespace clockwork;

std::shared_ptr<ClockworkRuntimeDummy> make_runtime_dummy() {
    return std::make_shared<ClockworkRuntimeWrapperDummy>();
}


std::vector<ModelDataDummy> loadModelDummy(std::string base_filename) {
    std::vector<ModelDataDummy> modeldata;

    for (unsigned batch_size = 1; ; batch_size *=2) {
        std::stringstream batch_filename_base;
        batch_filename_base << base_filename << "." << batch_size;

        std::string so_filename = batch_filename_base.str() + ".so";
        std::string clockwork_filename = batch_filename_base.str() + ".clockwork";

        if (!clockwork::util::exists(so_filename) || !clockwork::util::exists(clockwork_filename)) {
            break;
        }

        std::string serialized_spec;
        clockwork::util::readFileAsString(clockwork_filename, serialized_spec);

        modeldata.push_back(ModelDataDummy{
            batch_size,
            serialized_spec,
            0,
            0
        });
    }

    return modeldata;
}

RuntimeModelDummy* make_model_for_action(bool batched){
    std::string f;
    if(batched)
        f = clockwork::util::get_example_batched_model();
    else
        f = clockwork::util::get_example_model();

    // Load data for batch sizes and extract performance profile
    std::vector<unsigned> supported_batch_sizes;
    std::vector<uint64_t> batch_size_exec_times_nanos;
    uint64_t weights_load_time_nanos;

    std::vector<ModelDataDummy> modeldata = loadModelDummy(f);
    weights_load_time_nanos = modeldata[GPU_ID_0].weights_measurement;
    for (ModelDataDummy &d : modeldata) {
        supported_batch_sizes.push_back(d.batch_size);
        batch_size_exec_times_nanos.push_back(d.exec_measurement);
    }

    //deserialize the model metadata
    model::PageMappedModelDef* spec = new model::PageMappedModelDef();
    model::PageMappedModelDef::ReadFrom(modeldata[GPU_ID_0].serialized_spec, *spec);

    //Extract model metadata
    unsigned weights_pages_count = spec->weights_pages.size();
    uint64_t weights_size = weights_pages_count*spec->configured_weights_page_size;
    size_t inputs_size = 0;
    size_t outputs_size = 0;
    for (auto &input : spec->inputs) {
        inputs_size += input.size;
    }
    for (auto &output : spec->outputs) {
        outputs_size += output.size;
    }

    workerapi::ModelInfo* modelInfo = new workerapi::ModelInfo();
    modelInfo->id = 0;
    modelInfo->source = f;
    modelInfo->input_size = inputs_size;
    modelInfo->output_size = outputs_size;
    modelInfo->supported_batch_sizes = supported_batch_sizes;
    modelInfo->weights_size = weights_size;
    modelInfo->num_weights_pages = spec->configured_weights_page_size;
    modelInfo->weights_load_time_nanos = weights_load_time_nanos;
    modelInfo->batch_size_exec_times_nanos = batch_size_exec_times_nanos;
    return new RuntimeModelDummy(modelInfo,GPU_ID_0,weights_pages_count);
}

// Load Model 
TEST_CASE("Load Model From Disk Action Dummy", "[action] [loadmodel_action] [loadmodel_single] [dummy]") {
    auto clockwork = make_runtime_dummy();

    TestLoadModelFromDiskDummy load_model(clockwork.get(), load_model_from_disk_action());

    load_model.submit();
    load_model.await();
    load_model.check_success(true);
}

TEST_CASE("Load Model From Disk Action Multiple Dummy", "[action] [loadmodel_action] [loadmodel_mul] [dummy]") {
    auto clockwork = make_runtime_dummy();

    for (unsigned i = 0; i < 5; i++) {
        auto action = load_model_from_disk_action();
        action->model_id = i;

        TestLoadModelFromDiskDummy load_model(clockwork.get(), action);

        load_model.submit();
        load_model.await();
        load_model.check_success(true);
    }
}

TEST_CASE("Load Model From Disk Action Multiple Concurrent Dummy", "[action] [loadmodel_action] [loadmodel_concurrent] [dummy]") {
    auto clockwork = make_runtime_dummy();

    std::vector<std::shared_ptr<TestLoadModelFromDiskDummy>> loadmodels;

    for (unsigned i = 0; i < 5; i++) {
        auto action = load_model_from_disk_action();
        action->model_id = i;
        loadmodels.push_back( std::make_shared<TestLoadModelFromDiskDummy>(clockwork.get(), 
            action) );
    }

    for (auto loadmodel : loadmodels) {
        loadmodel->submit();
    }

    for (auto loadmodel : loadmodels) {
        loadmodel->await();
        loadmodel->check_success(true);
    }
}

TEST_CASE("Load Model Earliest Dummy", "[action] [loadmodel_action] [loadmodel_earliest] [dummy]") {
    auto clockwork = make_runtime_dummy();
    auto action = load_model_from_disk_action();
    uint64_t now = util::now();
    action->earliest = now+1000000000;
    action->latest = now+1000000000;
    TestLoadModelFromDiskDummy load_model(clockwork.get(),action);

    load_model.run();//Check for timestamp should exclude engine deque process
    load_model.await();
    load_model.check_success(false, actionErrorRuntimeError);
}

TEST_CASE("Load Model Latest Dummy", "[action] [loadmodel_action] [loadmodel_latest] [dummy]") {
    auto clockwork = make_runtime_dummy();
    auto action = load_model_from_disk_action();
    uint64_t now = util::now();
    action->earliest = 0;
    action->latest = now - 1000000;
    TestLoadModelFromDiskDummy load_model(clockwork.get(),action);

    load_model.run();//Check for timestamp should exclude engine deque process
    load_model.await();
    load_model.check_success(false, actionErrorCouldNotStartInTime);
}

TEST_CASE("Load Model Already existed Dummy", "[action] [loadmodel_action] [loadmodel_existed] [dummy]") {
    auto clockwork = make_runtime_dummy();

    TestLoadModelFromDiskDummy load_model(clockwork.get(), load_model_from_disk_action());

    load_model.submit();
    load_model.await();
    load_model.check_success(true);

    TestLoadModelFromDiskDummy load_model2(clockwork.get(), load_model_from_disk_action());

    load_model2.submit();
    load_model2.await();
    load_model2.check_success(false, actionErrorInvalidModelID);
}

TEST_CASE("Load Non-Existent Model From Disk Dummy", "[action] [loadmodel_action] [loadmodel_nomodel] [dummy]") {
    auto clockwork = make_runtime_dummy();
    auto action = load_model_from_disk_action();
    action->model_path = "";
    TestLoadModelFromDiskDummy load_model(clockwork.get(),action);

    load_model.submit();
    load_model.await();
    load_model.check_success(false, actionErrorInvalidModelPath);
}

// Load Weights 
TEST_CASE("Load Weights Action Dummy", "[action] [loadweights_action] [loadweights_single] [dummy]") {
    RuntimeModelDummy* model = make_model_for_action(false);
    auto clockwork = make_runtime_dummy();
    clockwork->manager->models->put(0, GPU_ID_0, model);

    TestLoadWeightsDummy load_weights(clockwork.get(), load_weights_action());

    load_weights.submit();
    load_weights.await();
    load_weights.check_success(true);    
}

TEST_CASE("Load Weights Action Multiple Dummy", "[action] [loadweights_action] [loadweights_mul] [dummy]") {
    RuntimeModelDummy* model = make_model_for_action(false);
    auto clockwork = make_runtime_dummy();
    clockwork->manager->models->put(0, GPU_ID_0, model);

    for (unsigned i = 0; i < 5; i++) {
        TestLoadWeightsDummy load_weights(clockwork.get(), load_weights_action());

        load_weights.submit();
        load_weights.await();
        load_weights.check_success(true);
    }
}


TEST_CASE("Load Weights Action Multiple Concurrent Dummy", "[action] [loadweights_action] [loadweights_concurrent] [dummy]") {
    auto clockwork = make_runtime_dummy();
    std::vector<std::shared_ptr<TestLoadModelFromDiskDummy>> loadmodels;

    for (unsigned i = 0; i < 5; i++) {
        auto action = load_model_from_disk_action();
        action->model_id = i;
        loadmodels.push_back( std::make_shared<TestLoadModelFromDiskDummy>(clockwork.get(), 
            action) );
    }

    for (auto loadmodel : loadmodels) {
        loadmodel->submit();
    }

    for (auto loadmodel : loadmodels) {
        loadmodel->await();
        loadmodel->check_success(true);
    }

    std::vector<std::shared_ptr<TestLoadWeightsDummy>> loadweights;

    for (unsigned i = 0; i < loadweights.size(); i++) {
        loadweights.push_back( std::make_shared<TestLoadWeightsDummy>(clockwork.get(), 
            load_weights_action(i)) );
    }

    for (auto loadweight : loadweights) {
        loadweight->submit();
    }

    for (auto loadweight : loadweights) {
        loadweight->await();
        loadweight->check_success(true);
    }
}

TEST_CASE("Load Weights Earliest Dummy", "[action] [loadweights_action] [loadweights_earliest] [dummy]") {
    RuntimeModelDummy* model = make_model_for_action(false);
    auto clockwork = make_runtime_dummy();

    auto action = load_weights_action();
    uint64_t now = util::now();
    action->earliest = now+1000000000;
    action->latest = now+1000000000;
    TestLoadWeightsDummy load_weights(clockwork.get(), action);

    load_weights.run();//Check for timestamp should exclude engine deque process
    load_weights.await();
    load_weights.check_success(false, actionErrorRuntimeError);
}

TEST_CASE("Load Weights Latest Dummy", "[action] [loadweights_action] [loadweights_latest] [dummy]") {
    RuntimeModelDummy* model = make_model_for_action(false);
    auto clockwork = make_runtime_dummy();

    auto action = load_weights_action();
    uint64_t now = util::now();
    action->earliest = 0;
    action->latest = now - 1000000;
    TestLoadWeightsDummy load_weights(clockwork.get(), action);

    load_weights.run();//Check for timestamp should exclude engine deque process
    load_weights.await();
    load_weights.check_success(false, actionErrorCouldNotStartInTime);
}

TEST_CASE("Load Weights Action Invalid Model Dummy", "[action] [loadweights_action]  [loadweights_nomodel] [dummy]") {
    RuntimeModelDummy* model = make_model_for_action(false);
    auto clockwork = make_runtime_dummy();

    TestLoadWeightsDummy load_weights(clockwork.get(), load_weights_action());

    load_weights.submit();
    load_weights.await();
    load_weights.check_success(false, loadWeightsUnknownModel);
}

TEST_CASE("Load Weights Insufficient Cache Dummy", "[action] [loadweights_action] [loadweights_cache] [dummy]") {
    RuntimeModelDummy* model = make_model_for_action(false);
    auto clockwork = make_runtime_dummy();
    clockwork->manager->models->put(0, GPU_ID_0, model);
    clockwork->manager->weights_caches[GPU_ID_0]->n_free_pages = 0;

    TestLoadWeightsDummy load_weights(clockwork.get(), load_weights_action());

    load_weights.submit();
    load_weights.await();
    load_weights.check_success(false, loadWeightsInsufficientCache);
}

TEST_CASE("Load Weights Version Update Dummy", "[action] [loadweights_action] [loadweights_weight_version] [dummy]") {
    RuntimeModelDummy* rm = make_model_for_action(false);
    rm->modelinfo->weights_load_time_nanos = 100000;//Hard code let process_completion() finish later
    auto clockwork = make_runtime_dummy();
    clockwork->manager->models->put(0, GPU_ID_0, rm);

    auto action = load_weights_action();
    TestLoadWeightsDummy load_weights(clockwork.get(), action);
    load_weights.run();//run immediatly so that version update happen in between run() and process_completion()

    rm->lock();
    bool flag = (!load_weights.is_success) && (!load_weights.is_error);
    REQUIRE(flag);
    rm->version++;
    rm->unlock();

    load_weights.await();
    load_weights.check_success(false, loadWeightsConcurrentModification);
}

//TEST_CASE("Double Load Weights Dummy", "[task] [dummy]")  this can't happen ( and be tested or deadlock will happen) since the engine will always try to finish an unfinished action first in the dummy worker


//Evict Weights
TEST_CASE("Load Evict Weights Action Dummy", "[action] [evict_action] [evict_single] [dummy]") {
    RuntimeModelDummy* model = make_model_for_action(false);
    auto clockwork = make_runtime_dummy();
    clockwork->manager->models->put(0, GPU_ID_0, model);

    TestLoadWeightsDummy load_weights(clockwork.get(), load_weights_action());

    load_weights.submit();
    load_weights.await();
    load_weights.check_success(true);

    TestEvictWeightsDummy evict_weights(clockwork.get(), evict_weights_action());

    evict_weights.submit();
    evict_weights.await();
    evict_weights.check_success(true);
}

TEST_CASE("Load Evict Weights Action Multiple Dummy", "[action] [evict_action] [evict_mul] [dummy]") {
    RuntimeModelDummy* model = make_model_for_action(false);
    auto clockwork = make_runtime_dummy();
    clockwork->manager->models->put(0, GPU_ID_0, model);

    for (unsigned i = 0; i < 5; i++) {
        TestLoadWeightsDummy load_weights(clockwork.get(), load_weights_action());

        load_weights.submit();
        load_weights.await();
        load_weights.check_success(true);

        TestEvictWeightsDummy evict_weights(clockwork.get(), evict_weights_action());

        evict_weights.submit();
        evict_weights.await();
        evict_weights.check_success(true);
    }
}


TEST_CASE("Load Evict Weights Action Multiple Concurrent Dummy", "[action] [evict_action] [evict_concurrent] [dummy]") {
    auto clockwork = make_runtime_dummy();
    std::vector<std::shared_ptr<TestLoadModelFromDiskDummy>> loadmodels;

    for (unsigned i = 0; i < 5; i++) {
        auto action = load_model_from_disk_action();
        action->model_id = i;
        loadmodels.push_back( std::make_shared<TestLoadModelFromDiskDummy>(clockwork.get(), 
            action) );
    }

    for (auto loadmodel : loadmodels) {
        loadmodel->submit();
    }

    for (auto loadmodel : loadmodels) {
        loadmodel->await();
        loadmodel->check_success(true);
    }

    std::vector<std::shared_ptr<TestLoadWeightsDummy>> loadweights;
    std::vector<std::shared_ptr<TestEvictWeightsDummy>> evictweights;

    for (unsigned i = 0; i < loadweights.size(); i++) {
        loadweights.push_back( std::make_shared<TestLoadWeightsDummy>(clockwork.get(), 
            load_weights_action(i)) );

        auto action = evict_weights_action();
        action->model_id = i;
        evictweights.push_back( std::make_shared<TestEvictWeightsDummy>(clockwork.get(), action) );
    }

    for (auto loadweight : loadweights) {
        loadweight->submit();
    }

    for (auto loadweight : loadweights) {
        loadweight->await();
        loadweight->check_success(true);
    }

    for (auto evictweight : evictweights) {
        evictweight->submit();
    }

    for (auto evictweight : evictweights) {
        evictweight->await();
        evictweight->check_success(true);
    }
}

//TEST_CASE("Double Evict Dummy", "[task] [dummy]") this can't happen (and be tested or deadlock will happen) since the engine will always try to finish a unfinished action first in the dummy worker

TEST_CASE("Evict Weights Earliest Dummy", "[action] [evict_action] [evict_earliest] [dummy]") {
    RuntimeModelDummy* model = make_model_for_action(false);
    auto clockwork = make_runtime_dummy();

    auto action = evict_weights_action();
    uint64_t now = util::now();
    action->earliest = now+1000000000;
    action->latest = now+1000000000;
    TestEvictWeightsDummy evict_weights(clockwork.get(), action);

    evict_weights.run();//Check for timestamp should exclude engine deque process
    evict_weights.await();
    evict_weights.check_success(false, actionErrorRuntimeError);
}

TEST_CASE("Evict Weights Latest Dummy", "[action] [evict_action] [evict_latest] [dummy]") {
    RuntimeModelDummy* model = make_model_for_action(false);
    auto clockwork = make_runtime_dummy();

    auto action = evict_weights_action();
    uint64_t now = util::now();
    action->earliest = 0;
    action->latest = now - 1000000;
    TestEvictWeightsDummy evict_weights(clockwork.get(), action);

    evict_weights.run();//Check for timestamp should exclude engine deque process
    evict_weights.await();
    evict_weights.check_success(false, actionErrorCouldNotStartInTime);
}

TEST_CASE("Evict Weights Nonexistent Model Dummy", "[action] [evict_action] [evict_nomodel] [dummy]") {
    RuntimeModelDummy* model = make_model_for_action(false);
    auto clockwork = make_runtime_dummy();

    TestEvictWeightsDummy evict_weights(clockwork.get(), evict_weights_action());

    evict_weights.submit();
    evict_weights.await();
    evict_weights.check_success(false, evictWeightsUnknownModel);
}

TEST_CASE("Evict without Weights Action Dummy", "[action] [evict_action] [evict_noweights] [dummy]") {
    RuntimeModelDummy* model = make_model_for_action(false);
    auto clockwork = make_runtime_dummy();
    clockwork->manager->models->put(0, GPU_ID_0, model);

    TestEvictWeightsDummy evict_weights(clockwork.get(), evict_weights_action());

    evict_weights.submit();
    evict_weights.await();
    evict_weights.check_success(false, evictWeightsNotInCache);
}

//Infer Action
TEST_CASE("Infer Action Dummy", "[action] [infer_action] [infer_single] [dummy]") {
    RuntimeModelDummy* model = make_model_for_action(false);
    auto clockwork = make_runtime_dummy();
    clockwork->manager->models->put(0, GPU_ID_0, model);

    TestLoadWeightsDummy load_weights(clockwork.get(), load_weights_action());

    load_weights.submit();
    load_weights.await();
    load_weights.check_success(true);

    TestInferDummy infer(clockwork.get(), infer_action());

    infer.submit();
    infer.await();
    infer.check_success(true);
}

TEST_CASE("Infer Action Batched Dummy", "[action] [infer_action] [infer_batched] [dummy]") {
    RuntimeModelDummy* model = make_model_for_action(true);
    auto clockwork = make_runtime_dummy();
    clockwork->manager->models->put(0, GPU_ID_0, model);

    TestLoadWeightsDummy load_weights(clockwork.get(), load_weights_action());

    load_weights.submit();
    load_weights.await();
    load_weights.check_success(true);

    for (unsigned batch_size = 1; batch_size <= 16; batch_size*=2){
        TestInferDummy infer(clockwork.get(), infer_action_dummy(batch_size, model));

        infer.submit();
        infer.await();
        infer.check_success(true);
    }
}

TEST_CASE("Infer Action Multiple Dummy", "[action] [infer_action] [infer_action_mul] [dummy]") {
    RuntimeModelDummy* model = make_model_for_action(false);
    auto clockwork = make_runtime_dummy();
    clockwork->manager->models->put(0, GPU_ID_0, model);

    TestLoadWeightsDummy load_weights(clockwork.get(), load_weights_action());

    load_weights.submit();
    load_weights.await();
    load_weights.check_success(true);

    for (unsigned i = 0; i < 5; i++) {
        TestInferDummy infer(clockwork.get(), infer_action());

        infer.submit();
        infer.await();
        infer.check_success(true);
    }
}

TEST_CASE("Make Many Models Dummy", "[action] [models] [dummy]") {
    std::vector<RuntimeModelDummy*> models;
    for (unsigned i = 0; i < 30; i++) {
        models.push_back(make_model_for_action(false));
    }
    for (RuntimeModelDummy* model : models) {
        delete model;
    }
}//Q what is it?

TEST_CASE("Infer Action Multiple Concurrent Dummy", "[action] [infer_action] [infer_concurrent] [dummy]") {
    RuntimeModelDummy* model = make_model_for_action(false);
    auto clockwork = make_runtime_dummy();
    clockwork->manager->models->put(0, GPU_ID_0, model);

    TestLoadWeightsDummy load_weights(clockwork.get(), load_weights_action());

    load_weights.submit();
    load_weights.await();
    load_weights.check_success(true);

    std::vector<std::shared_ptr<TestInferDummy>> infers;

    for (unsigned i = 0; i < 16; i++) {
        infers.push_back(std::make_shared<TestInferDummy>(clockwork.get(), infer_action()));
    }

    for (auto infer : infers) {
        infer->submit();
    }

    for (auto infer : infers) {
        infer->await();
        infer->check_success(true);
    }    
}

TEST_CASE("Infer Multiple GPUs Dummy", "[action] [infer_action] [infer_mul_gpus] [dummy]") {    
    auto clockwork = make_runtime_dummy();
    unsigned num_gpus = clockwork->num_gpus;

    TestLoadModelFromDiskDummy load_model(clockwork.get(), load_model_from_disk_action());
    load_model.submit();
    load_model.await();
    load_model.check_success(true);

    std::vector<std::shared_ptr<TestLoadWeightsDummy>> loadweights;

    for(unsigned i = 0; i < num_gpus; i++){
        auto action = load_weights_action();
        action->gpu_id = i;
        loadweights.push_back(std::make_shared<TestLoadWeightsDummy>(clockwork.get(),action));
    }

    for(auto loadweight : loadweights){
        loadweight->submit();
    }

    for(auto loadweight : loadweights){
        loadweight->await();
        loadweight->check_success(true);
    }

    std::vector<std::shared_ptr<TestInferDummy>> infers;

    for (unsigned i = 0; i < num_gpus; i++) {
        auto action = infer_action();
        action->gpu_id = i;
        infers.push_back(std::make_shared<TestInferDummy>(clockwork.get(),action ));
    }

    for (auto infer : infers) {
        infer->submit();
    }

    for (auto infer : infers) {
        infer->await();
        infer->check_success(true);
    }  
}

TEST_CASE("Infer Earliest Dummy", "[action] [infer_action] [infer_earliest] [dummy]") {
    RuntimeModelDummy* model = make_model_for_action(false);
    auto clockwork = make_runtime_dummy();

    auto action = infer_action();
    uint64_t now = util::now();
    action->earliest = now+1000000000;
    action->latest = now+1000000000;
    TestInferDummy infer(clockwork.get(), action);

    infer.run();//Check for timestamp should exclude engine deque process
    infer.await();
    infer.check_success(false, actionErrorRuntimeError);
}

TEST_CASE("Infer Latest Dummy", "[action] [infer_action] [infer_latest] [dummy]") {
    RuntimeModelDummy* model = make_model_for_action(false);
    auto clockwork = make_runtime_dummy();

    auto action = infer_action();
    uint64_t now = util::now();
    action->earliest = 0;
    action->latest = now - 1000000;
    TestInferDummy infer(clockwork.get(), action);

    infer.run();//Check for timestamp should exclude engine deque process
    infer.await();
    infer.check_success(false, actionErrorCouldNotStartInTime);
}

TEST_CASE("Infer Action Wrong Input Size Dummy", "[action] [infer_action] [infer_wrong_iosize] [dummy]") {
    RuntimeModelDummy* model = make_model_for_action(false);
    auto clockwork = make_runtime_dummy();
    clockwork->manager->models->put(0, GPU_ID_0, model);

    TestLoadWeightsDummy load_weights(clockwork.get(), load_weights_action());
    load_weights.submit();
    load_weights.await();
    load_weights.check_success(true);

    auto action = infer_action();
    action->input_size = 10;
    TestInferDummy infer(clockwork.get(), action);

    infer.submit();
    infer.await();
    infer.check_success(false, copyInputInvalidInput);
}

TEST_CASE("Infer Action Nonexistent Model Dummy", "[action] [infer_action] [infer_nomodel] [dummy]") {
    auto clockwork = make_runtime_dummy();

    TestInferDummy infer(clockwork.get(), infer_action());

    infer.submit();
    infer.await();
    infer.check_success(false, copyInputUnknownModel);
}

TEST_CASE("Infer Action Nonexistent Weights Dummy", "[action] [infer_action] [infer_noweights] [dummy]") {
    RuntimeModelDummy* model = make_model_for_action(false);
    auto clockwork = make_runtime_dummy();
    clockwork->manager->models->put(0, GPU_ID_0, model);

    TestInferDummy infer(clockwork.get(), infer_action());

    infer.submit();
    infer.await();
    infer.check_success(false, execWeightsMissing);
}

TEST_CASE("Infer after Evict Action Dummy", "[action] [infer_action] [infer_after_evict] [dummy]") {
    RuntimeModelDummy* model = make_model_for_action(false);
    auto clockwork = make_runtime_dummy();
    clockwork->manager->models->put(0, GPU_ID_0, model);

    TestLoadWeightsDummy load_weights(clockwork.get(), load_weights_action());

    load_weights.submit();
    load_weights.await();
    load_weights.check_success(true);

    TestInferDummy infer(clockwork.get(), infer_action());

    infer.submit();
    infer.await();
    infer.check_success(true);

    TestEvictWeightsDummy evict_weights(clockwork.get(), evict_weights_action());

    evict_weights.submit();
    evict_weights.await();
    evict_weights.check_success(true);

    TestInferDummy infer2(clockwork.get(), infer_action());

    infer2.submit();
    infer2.await();
    infer2.check_success(false, execWeightsMissing);   
}

TEST_CASE("Actions E2E Dummy", "[action] [e2e] [dummy]") {
    auto clockwork = make_runtime_dummy();

    auto load_model = new TestLoadModelFromDiskDummy(clockwork.get(), load_model_from_disk_action());

    load_model->submit();
    load_model->await();
    load_model->check_success(true);

    delete load_model;

    auto load_weights = new TestLoadWeightsDummy(clockwork.get(), load_weights_action());

    load_weights->submit();
    load_weights->await();
    load_weights->check_success(true);

    delete load_weights;
    
    RuntimeModelDummy* model = clockwork->manager->models->get(0, GPU_ID_0);
    auto infer = new TestInferDummy(clockwork.get(), infer_action());

    infer->submit();
    infer->await();
    infer->check_success(true);

    delete infer;

    auto evict_weights = new TestEvictWeightsDummy(clockwork.get(), evict_weights_action());

    evict_weights->submit();
    evict_weights->await();
    evict_weights->check_success(true);

    delete evict_weights;

    auto infer2 = new TestInferDummy(clockwork.get(), infer_action());

    infer2->submit();
    infer2->await();
    infer2->check_success(false, execWeightsMissing);

    delete infer2;  
}

TEST_CASE("Task Cancelled After Shutdown Dummy", "[action] [shutdown] [dummy]") {
    RuntimeModelDummy* model = make_model_for_action(false);
    auto clockwork = make_runtime_dummy();
    clockwork->manager->models->put(0, GPU_ID_0, model);

    auto action = load_weights_action();
    auto load_weights = new TestLoadWeightsDummy(clockwork.get(), action);   

    load_weights->submit();

    clockwork = nullptr; // destroys the runtime

    load_weights->await();
    load_weights->check_success(false, actionCancelled);

    delete load_weights;
}

class TestLoadModelFromDiskActionThatDeletesItselfDummy : public LoadModelFromDiskDummyAction {
public:
    TestAction &action_status;

    ClockworkRuntimeDummy* myRuntime;

    TestLoadModelFromDiskActionThatDeletesItselfDummy(ClockworkRuntimeDummy* runtime, std::shared_ptr<workerapi::LoadModelFromDisk> action, 
            TestAction &action_status) : 
        LoadModelFromDiskDummyAction(runtime->manager, action), action_status(action_status) {myRuntime = runtime;}

    void submit(){
    myRuntime->load_model_executor->actions_to_start.push(element{loadmodel->earliest, [this]() {this->run();} ,[this]() {this->error(actionCancelled, "Action cancelled");} });
    }

    void success(std::shared_ptr<workerapi::LoadModelFromDiskResult> result) {
        action_status.setsuccess();
        delete this;
    }

    void error(int status_code, std::string message) {
        auto result = std::make_shared<workerapi::ErrorResult>();
        result->action_type = workerapi::loadModelFromDiskAction;
        result->status = status_code; 
        action_status.seterror(result);
        delete this;
    }

};

TEST_CASE("Task Action That Deletes Itself in Callback Dummy", "[action] [shutdown] [dummy]") {
    auto clockwork = make_runtime_dummy();

    TestAction action_status;

    auto action = load_model_from_disk_action();
    action->model_path = "";
    auto load_model = new TestLoadModelFromDiskActionThatDeletesItselfDummy(
            clockwork.get(), action, action_status);

    load_model->submit();
    action_status.await();
    action_status.check_success(false, actionErrorInvalidModelPath);

    
}//Q what is it