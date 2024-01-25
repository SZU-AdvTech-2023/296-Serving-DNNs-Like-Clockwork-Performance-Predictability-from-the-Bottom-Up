#include "clockwork/test_dummy/actions.h"

namespace clockwork {

std::shared_ptr<workerapi::Infer> infer_action_dummy(int batch_size, RuntimeModelDummy* model) {
    auto action = infer_action();
    action->batch_size = batch_size;
    action->input_size = model->input_size(batch_size);
    return action;
}

void TestLoadModelFromDiskDummy::submit(){
    myRuntime->load_model_executor->actions_to_start.push(element{loadmodel->earliest, [this]() {this->run();} ,[this]() {this->error(actionCancelled, "Action cancelled");} });
}

void TestLoadModelFromDiskDummy::success(std::shared_ptr<workerapi::LoadModelFromDiskResult> result) {
    setsuccess();
}

void TestLoadModelFromDiskDummy::error(int status_code, std::string message){
    auto result = std::make_shared<workerapi::ErrorResult>();
    result->action_type = workerapi::loadModelFromDiskAction;
    result->status = status_code; 
    seterror(result);
}

void TestLoadWeightsDummy::submit(){
    myRuntime->weights_executors[0]->actions_to_start.push(element{loadweights->earliest, [this]() {this->run();} ,[this]() {this->error(actionCancelled, "Action cancelled");} });
}

void TestLoadWeightsDummy::toComplete(){
    element* action = new element();
    action->ready = end;
    action->callback = [this]() {this->process_completion();};
    action->defaultfunc = [this]() {this->error(actionCancelled, "Action cancelled");};
    myRuntime->engine->addToEnd(workerapi::loadWeightsAction, loadweights->gpu_id,action);
}

void TestLoadWeightsDummy::success(std::shared_ptr<workerapi::LoadWeightsResult> result) {
    setsuccess();
}

void TestLoadWeightsDummy::error(int status_code, std::string message) {
    auto result = std::make_shared<workerapi::ErrorResult>();
    result->action_type = workerapi::loadWeightsAction;
    result->status = status_code;
    seterror(result);
}

void TestEvictWeightsDummy::submit(){
    myRuntime->outputs_executors[0]->actions_to_start.push(element{evictweights->earliest, [this]() {this->run();} ,[this]() {this->error(actionCancelled, "Action cancelled");} });
}

void TestEvictWeightsDummy::success(std::shared_ptr<workerapi::EvictWeightsResult> result) {
    setsuccess();
}

void TestEvictWeightsDummy::error(int status_code, std::string message) {
    auto result = std::make_shared<workerapi::ErrorResult>();
    result->action_type = workerapi::evictWeightsAction;
    result->status = status_code;
    seterror(result);
}

void TestInferDummy::submit(){
    myRuntime->gpu_executors[0]->actions_to_start.push(element{infer->earliest, [this]() {this->run();} ,[this]() {this->error(actionCancelled, "Action cancelled");} });
}

void TestInferDummy::toComplete(){
    element* action = new element();
    action->ready = end;
    action->callback = [this]() {this->process_completion();};
    action->defaultfunc = [this]() {this->error(actionCancelled, "Action cancelled");};
    myRuntime->engine->addToEnd(workerapi::inferAction,infer->gpu_id, action);
}

void TestInferDummy::success(std::shared_ptr<workerapi::InferResult> result) {
    setsuccess();
}

void TestInferDummy::error(int status_code, std::string message) {
    auto result = std::make_shared<workerapi::ErrorResult>();
    result->action_type = workerapi::inferAction;
    result->status = status_code;
    seterror(result);
}

}
