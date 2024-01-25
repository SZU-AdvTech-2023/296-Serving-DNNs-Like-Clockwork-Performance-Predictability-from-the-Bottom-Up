#include "clockwork/dummy/worker_dummy.h"

namespace clockwork {

EngineDummy::EngineDummy(unsigned num_gpus){
    for (unsigned gpu_id = 0; gpu_id < num_gpus; gpu_id++){
        infers_to_end.push_back(nullptr);
        loads_to_end.push_back(nullptr);
    }
}

void EngineDummy::addExecutor(ExecutorDummy* executor){executors.push_back(executor);}

void EngineDummy::addToEnd(int type, unsigned gpu_id, element* action){
    if(type == workerapi::loadWeightsAction)
        loads_to_end[gpu_id] = action;
    else if (type == workerapi::inferAction)
        infers_to_end[gpu_id] = action;
}

void EngineDummy::startEngine(){
    alive.store(true);
    run_thread = std::thread(&EngineDummy::run, this);
}

void EngineDummy::run() {
    while(alive.load()){
        uint64_t timestamp = util::now();
        element next;
        for(ExecutorDummy* executor: executors){
            if(!alive.load()) break;
            if(executor->type == workerapi::loadWeightsAction){
                if(loads_to_end[executor->gpu_id] != nullptr){
                    if(loads_to_end[executor->gpu_id]->ready <= timestamp){
                        loads_to_end[executor->gpu_id]->callback();
                        delete loads_to_end[executor->gpu_id];
                        loads_to_end[executor->gpu_id] = nullptr;
                    }
                }else if(executor->actions_to_start.try_pop(next)){
                    if(next.ready <= timestamp)
                        next.callback();//loads_to_end[executor->gpu_id] = &element{end_at,loadweights_on_complete} if on_start succeed
                    else
                        executor->actions_to_start.push(next); 
                }
            }else if(executor->type == workerapi::inferAction){
                if(infers_to_end[executor->gpu_id] != nullptr){
                    if(infers_to_end[executor->gpu_id]->ready <= timestamp){
                        infers_to_end[executor->gpu_id]->callback();
                        delete infers_to_end[executor->gpu_id];
                        infers_to_end[executor->gpu_id] = nullptr;
                    }
                }else if(executor->actions_to_start.try_pop(next)){
                    if(next.ready <= timestamp)
                        next.callback();
                    else
                        executor->actions_to_start.push(next);
                }
            }else if(executor->actions_to_start.try_pop(next)){
                if(next.ready <= timestamp)
                        next.callback();
                else
                    executor->actions_to_start.push(next);
            } 
        }
    }

    element next;
    for(ExecutorDummy* executor: executors){
        if(executor->type == workerapi::loadWeightsAction){
            if(loads_to_end[executor->gpu_id] != nullptr){

                loads_to_end[executor->gpu_id]->defaultfunc();
                delete loads_to_end[executor->gpu_id];
                loads_to_end[executor->gpu_id] = nullptr;

            }else if(executor->actions_to_start.try_pop(next)){
                next.defaultfunc();
            }
        }else if(executor->type == workerapi::inferAction){
            if(infers_to_end[executor->gpu_id] != nullptr){

                infers_to_end[executor->gpu_id]->defaultfunc();
                delete infers_to_end[executor->gpu_id];
                infers_to_end[executor->gpu_id] = nullptr;

            }else if(executor->actions_to_start.try_pop(next)){
                next.defaultfunc();
            }
        }else if(executor->actions_to_start.try_pop(next)){
            next.defaultfunc();
        } 
    }

}

void ExecutorDummy::new_action(std::shared_ptr<workerapi::LoadModelFromDisk> action){
    LoadModelFromDiskDummy* loadmodel = new LoadModelFromDiskDummy(myManager,myEngine,action,myController);
    actions_to_start.push(element{loadmodel->loadmodel->earliest, [loadmodel]() {loadmodel->run();} ,[loadmodel]() {loadmodel->error(actionCancelled, "Action cancelled");} });
};

void ExecutorDummy::new_action(std::shared_ptr<workerapi::LoadWeights> action){
    LoadWeightsDummy* loadweights = new LoadWeightsDummy(myManager,myEngine,action, myController);
    actions_to_start.push(element{loadweights->loadweights->earliest,[loadweights]() {loadweights->run();}, [loadweights]() {loadweights->error(actionCancelled, "Action cancelled");} });
};

void ExecutorDummy::new_action(std::shared_ptr<workerapi::EvictWeights> action){
    EvictWeightsDummy* evictweights = new EvictWeightsDummy(myManager,myEngine,action, myController);
    actions_to_start.push(element{evictweights->evictweights->earliest, [evictweights]() {evictweights->run();}, [evictweights]() {evictweights->error(actionCancelled, "Action cancelled");} });
};

void ExecutorDummy::new_action(std::shared_ptr<workerapi::Infer> action){
    InferDummy* infer = new InferDummy(myManager,myEngine,action, myController);
    actions_to_start.push(element{infer->infer->earliest, [infer]() {infer->run();}, [infer]() {infer->error(actionCancelled, "Action cancelled");} });
};

void ClockworkRuntimeDummy::setController(workerapi::Controller* Controller){
    for (unsigned gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
            gpu_executors[gpu_id]->setController(Controller);
            weights_executors[gpu_id]->setController(Controller);
            outputs_executors[gpu_id]->setController(Controller);
    }
    load_model_executor->setController(Controller);
}

void ClockworkRuntimeDummy::shutdown(bool await_completion) {
    /* 
    Stop engine.  It'll finish current tasks, prevent enqueueing
    new tasks, and cancel tasks that haven't been started yet
    */
    engine->shutdown();
    if (await_completion) {
        join();
    }
}

void ClockworkRuntimeDummy::join() {
    //Wait for engine to be finished
    engine->join();
}

void ClockworkDummyWorker::sendActions(std::vector<std::shared_ptr<workerapi::Action>> &actions) {
    for (std::shared_ptr<workerapi::Action> action : actions) {
        switch (action->action_type) {
            case workerapi::loadModelFromDiskAction: loadModel(action); break;
            case workerapi::loadWeightsAction: loadWeights(action); break;
            case workerapi::inferAction: infer(action); break;
            case workerapi::evictWeightsAction: evictWeights(action); break;
            case workerapi::clearCacheAction: clearCache(action); break;
            case workerapi::getWorkerStateAction: getWorkerState(action); break;
            default: invalidAction(action); break;
        }
    }
}
void ClockworkDummyWorker::setController(workerapi::Controller* Controller){
    runtime->setController(Controller);
    controller = Controller;
}
void ClockworkDummyWorker::invalidAction(std::shared_ptr<workerapi::Action> action) {
    auto result = std::make_shared<workerapi::ErrorResult>();

    result->id = action->id;
    result->action_type = action->action_type;
    result->status = actionErrorRuntimeError;
    result->message = "Invalid Action";

    controller->sendResult(result);
}

// Need to be careful of timestamp = 0 and timestamp = UINT64_MAX which occur often
// and clock_delta can be positive or negative
uint64_t adjust_timestamp_dummy(uint64_t timestamp, int64_t clock_delta) {
    if (clock_delta >= 0) return std::max(timestamp, timestamp + clock_delta);
    else return std::min(timestamp, timestamp + clock_delta);
}

void ClockworkDummyWorker::loadModel(std::shared_ptr<workerapi::Action> action) {
    auto load_model = std::static_pointer_cast<workerapi::LoadModelFromDisk>(action);
    if (load_model != nullptr) {
        // It is a hack to do this here, but easiest / safest place to do it for now
        load_model->earliest = adjust_timestamp_dummy(load_model->earliest, load_model->clock_delta);
        load_model->latest = adjust_timestamp_dummy(load_model->latest, load_model->clock_delta);

        runtime->load_model_executor->new_action(load_model);
    } else {
        invalidAction(action);
    }
}

void ClockworkDummyWorker::loadWeights(std::shared_ptr<workerapi::Action> action) {
    auto load_weights = std::static_pointer_cast<workerapi::LoadWeights>(action);
    if (load_weights != nullptr) {
        // It is a hack to do this here, but easiest / safest place to do it for now
        load_weights->earliest = adjust_timestamp_dummy(load_weights->earliest, load_weights->clock_delta);
        load_weights->latest = adjust_timestamp_dummy(load_weights->latest, load_weights->clock_delta);

        runtime->weights_executors[load_weights->gpu_id]->new_action(load_weights);      
    } else {
        invalidAction(action);
    }
}

void ClockworkDummyWorker::evictWeights(std::shared_ptr<workerapi::Action> action) {
    auto evict_weights = std::static_pointer_cast<workerapi::EvictWeights>(action);
    if (evict_weights != nullptr) {
        // It is a hack to do this here, but easiest / safest place to do it for now
        evict_weights->earliest = adjust_timestamp_dummy(evict_weights->earliest, evict_weights->clock_delta);
        evict_weights->latest = adjust_timestamp_dummy(evict_weights->latest, evict_weights->clock_delta);

        runtime->weights_executors[evict_weights->gpu_id]->new_action(evict_weights);
        
    } else {
        invalidAction(action);
    }
}

void ClockworkDummyWorker::infer(std::shared_ptr<workerapi::Action> action) {
    auto infer = std::static_pointer_cast<workerapi::Infer>(action);
    if (infer != nullptr) {
        // It is a hack to do this here, but easiest / safest place to do it for now
        infer->earliest = adjust_timestamp_dummy(infer->earliest, infer->clock_delta);
        infer->latest = adjust_timestamp_dummy(infer->latest, infer->clock_delta);

        runtime->gpu_executors[infer->gpu_id]->new_action(infer);
    } else {
        invalidAction(action);
    }
}

void ClockworkDummyWorker::clearCache(std::shared_ptr<workerapi::Action> action) {
    auto clear_cache = std::static_pointer_cast<workerapi::ClearCache>(action);
    if (clear_cache != nullptr) {
        runtime->manager->models->clearWeights();
        for (unsigned i = 0; i < runtime->num_gpus; i++) {
            runtime->manager->weights_caches[i]->clear();
        }
        auto result = std::make_shared<workerapi::ClearCacheResult>();
        result->id = action->id;
        result->action_type = workerapi::clearCacheAction;
        result->status = actionSuccess; 
        controller->sendResult(result);
    } else {
        invalidAction(action);
    }
}

void ClockworkDummyWorker::getWorkerState(std::shared_ptr<workerapi::Action> action) {
    auto get_worker_state = std::static_pointer_cast<workerapi::GetWorkerState>(action);
    if (get_worker_state != nullptr) {
        auto result = std::make_shared<workerapi::GetWorkerStateResult>();
        result->id = action->id;
        result->action_type = workerapi::getWorkerStateAction;
        runtime->manager->get_worker_memory_info(result->worker);
        result->status = actionSuccess; 
        controller->sendResult(result);
    } else {
        invalidAction(action);
    }
}

LoadModelFromDiskDummy::LoadModelFromDiskDummy( MemoryManagerDummy* Manager, EngineDummy* Engine, 
    std::shared_ptr<workerapi::LoadModelFromDisk> LoadModel, workerapi::Controller* Controller) :LoadModelFromDiskDummyAction(Manager, LoadModel), myEngine(Engine), myController(Controller){}

void LoadModelFromDiskDummy::error(int status_code, std::string message){
    auto result = std::make_shared<workerapi::ErrorResult>();
    result->id = loadmodel->id;
    result->action_type = workerapi::loadModelFromDiskAction;
    result->status = status_code;
    result->message = message;
    result->action_received = adjust_timestamp_dummy(loadmodel->received, -loadmodel->clock_delta);
    result->clock_delta = loadmodel->clock_delta;
    myController->sendResult(result);
    delete this;
}

void LoadModelFromDiskDummy::success(std::shared_ptr<workerapi::LoadModelFromDiskResult> result) {
    //Set timestamps in the result
    result->begin = adjust_timestamp_dummy(start, -loadmodel->clock_delta);
    result->end = adjust_timestamp_dummy(end, -loadmodel->clock_delta);
    result->duration = result->end - result->begin;
    result->action_received = adjust_timestamp_dummy(loadmodel->received, -loadmodel->clock_delta);
    result->clock_delta = loadmodel->clock_delta;
    myController->sendResult(result);
    delete this;
}

LoadWeightsDummy::LoadWeightsDummy(MemoryManagerDummy* Manager, EngineDummy* Engine,
    std::shared_ptr<workerapi::LoadWeights> LoadWeights, workerapi::Controller* Controller) : 
    LoadWeightsDummyAction(Manager, LoadWeights), myEngine(Engine), myController(Controller){}

void LoadWeightsDummy::toComplete(){
    //Add process_completion action to engine
    element* action = new element();
    action->ready = end;
    action->callback = [this]() {this->process_completion();};
    action->defaultfunc = [this]() {this->error(actionCancelled, "Action cancelled");};
    myEngine->addToEnd(workerapi::loadWeightsAction, loadweights->gpu_id,action);
}

void LoadWeightsDummy::success(std::shared_ptr<workerapi::LoadWeightsResult> result){
    result->id = loadweights->id;
    result->action_type = workerapi::loadWeightsAction;
    result->status = actionSuccess;

    //Set timestamps in the result
    result->begin = adjust_timestamp_dummy(start, -loadweights->clock_delta);
    result->end = adjust_timestamp_dummy(end, -loadweights->clock_delta);
    result->duration = result->end - result->begin;
    result->action_received = adjust_timestamp_dummy(loadweights->received, -loadweights->clock_delta);
    result->clock_delta = loadweights->clock_delta;
    
    myController->sendResult(result);
    delete this;
}

void LoadWeightsDummy::error(int status_code, std::string message){
    TaskError* error = new TaskError(status_code,message);
    auto result = std::make_shared<workerapi::ErrorResult>();
    result->id = loadweights->id;
    result->action_type = workerapi::loadWeightsAction;
    result->status = error->status_code;
    result->message = error->message;
    result->action_received = adjust_timestamp_dummy(loadweights->received, -loadweights->clock_delta);
    result->clock_delta = loadweights->clock_delta;
    myController->sendResult(result);
    delete this;
}

EvictWeightsDummy::EvictWeightsDummy(MemoryManagerDummy* Manager, EngineDummy* Engine,
    std::shared_ptr<workerapi::EvictWeights> EvictWeights, workerapi::Controller* Controller) : 
    EvictWeightsDummyAction(Manager, EvictWeights), myEngine(Engine), myController(Controller){}

void EvictWeightsDummy::success(std::shared_ptr<workerapi::EvictWeightsResult> result){
    result->id = evictweights->id;
    result->action_type = workerapi::evictWeightsAction;
    result->status = actionSuccess;

    //Set timestamps in the result
    result->begin = adjust_timestamp_dummy(start, -evictweights->clock_delta);
    result->end = adjust_timestamp_dummy(end, -evictweights->clock_delta);
    result->duration = result->end - result->begin;
    result->action_received = adjust_timestamp_dummy(evictweights->received, -evictweights->clock_delta);
    result->clock_delta = evictweights->clock_delta;
    
    myController->sendResult(result);
    delete this;
}

void EvictWeightsDummy::error(int status_code, std::string message){
    TaskError* error = new TaskError(status_code,message);
    auto result = std::make_shared<workerapi::ErrorResult>();
    result->id = evictweights->id;
    result->action_type = workerapi::evictWeightsAction;
    result->status = error->status_code;
    result->message = error->message;
    result->action_received = adjust_timestamp_dummy(evictweights->received, -evictweights->clock_delta);
    result->clock_delta = evictweights->clock_delta;
    myController->sendResult(result);
    delete this;
}

InferDummy::InferDummy( MemoryManagerDummy* Manager,EngineDummy* Engine,
     std::shared_ptr<workerapi::Infer> Infer,workerapi::Controller* Controller) : 
    InferDummyAction(Manager, Infer), myEngine(Engine), myController(Controller){}

void InferDummy::toComplete(){
    //Add process_completion action to engine
    element* action = new element();
    action->ready = end;
    action->callback = [this]() {this->process_completion();};
    action->defaultfunc = [this]() {this->error(actionCancelled, "Action cancelled");};
    myEngine->addToEnd(workerapi::inferAction,infer->gpu_id, action);
}

void InferDummy::error(int status_code, std::string message){
    TaskError* error = new TaskError(status_code,message);
    auto result = std::make_shared<workerapi::ErrorResult>();
    result->id = infer->id;
    result->action_type = workerapi::inferAction;
    result->status = error->status_code;
    result->message = error->message;
    result->action_received = adjust_timestamp_dummy(infer->received, -infer->clock_delta);
    result->clock_delta = infer->clock_delta;
    myController->sendResult(result);
    delete this;
}

void InferDummy::success(std::shared_ptr<workerapi::InferResult> result){
    result->id = infer->id;
    result->action_type = workerapi::inferAction;
    result->status = actionSuccess;

    //Set timestamps in the result
    result->copy_input.begin = adjust_timestamp_dummy(start, -infer->clock_delta);
    result->exec.begin = adjust_timestamp_dummy(start, -infer->clock_delta);
    result->copy_output.begin = adjust_timestamp_dummy(end, -infer->clock_delta);
    result->copy_input.end = adjust_timestamp_dummy(start, -infer->clock_delta);
    result->exec.end = adjust_timestamp_dummy(end, -infer->clock_delta);
    result->copy_output.end = adjust_timestamp_dummy(end, -infer->clock_delta);
    result->copy_input.duration = result->copy_input.end - result->copy_input.begin;
    result->exec.duration = result->exec.end - result->exec.begin;
    result->copy_output.duration = result->copy_output.end - result->copy_output.begin;

    if (infer->input_size == 0) {
        result->output_size = 0;
    }
    result->output = (char*)nullptr;

    result->gpu_id = infer->gpu_id;
    result->gpu_clock_before = 1380;//Magic number for gpu_clock
    result->gpu_clock = 1380;
    
    result->clock_delta = infer->clock_delta;
    
    myController->sendResult(result);
    delete this;
}


}