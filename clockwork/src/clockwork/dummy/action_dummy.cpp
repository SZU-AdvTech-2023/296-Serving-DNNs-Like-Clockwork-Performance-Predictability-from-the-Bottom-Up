#include <dmlc/logging.h>
#include "clockwork/modeldef.h"
#include "clockwork/dummy/action_dummy.h"

#include <iostream>

namespace clockwork {

void LoadModelFromDiskDummyAction::run(){

    //Check timestamp for running task
    start = util::now();
    std::stringstream err;
    if(start < loadmodel->earliest){
        err << "LoadModelFromDiskTask ran before it was eligible"
            << " (now " << util::millis(start)
            << ", earliest " << util::millis(loadmodel->earliest) << ")";
        error(actionErrorRuntimeError,err.str());
        return;

    }else if(start > loadmodel->latest){
        err << "LoadModelFromDiskTask could not start in time"
            << " (now " << util::millis(start)
            << ", latest " << util::millis(loadmodel->latest) << ")";
        error(actionErrorCouldNotStartInTime, err.str());
        return;
    }

    //Check if model is already loaded
    std::vector<unsigned> gpu_ids;
    for (unsigned gpu_id = 0; gpu_id < myManager->num_gpus; gpu_id++) {
        gpu_ids.push_back(gpu_id);
        for (unsigned i = 0; i < loadmodel->no_of_copies; i++) {
            if (myManager->models->contains(loadmodel->model_id+i, gpu_id)) {
                error(actionErrorInvalidModelID, "LoadModelFromDiskTask specified ID that already exists");
                return;
            }
        }
    }

    
    try{
        // Load data for batch sizes and extract performance profile
        std::vector<unsigned> supported_batch_sizes;
        std::vector<uint64_t> batch_size_exec_times_nanos;
        uint64_t weights_load_time_nanos;

        std::vector<ModelDataDummy> modeldata = loadModelDataDummy(loadmodel->model_path);
        weights_load_time_nanos = modeldata[0].weights_measurement;
        for (ModelDataDummy &d : modeldata) {
                if (d.batch_size <= loadmodel->max_batch_size && 
                    (d.batch_size == 1 || d.exec_measurement <= loadmodel->max_exec_duration)) {
                    supported_batch_sizes.push_back(d.batch_size);
                    batch_size_exec_times_nanos.push_back(d.exec_measurement);        
                }
        }

        //deserialize the model metadata
        model::PageMappedModelDef* spec = new model::PageMappedModelDef();

        model::PageMappedModelDef::ReadFrom(modeldata[0].serialized_spec, *spec);
        CHECK(spec != nullptr) << " spec is nullptr";

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

        //Add model to modelstore
        for (auto &gpu_id : gpu_ids) {
            for (unsigned i = 0; i < loadmodel->no_of_copies; i++) {
                workerapi::ModelInfo* modelInfo = new workerapi::ModelInfo();
                modelInfo->id = loadmodel->model_id + i;
                modelInfo->source = loadmodel->model_path;
                modelInfo->input_size = inputs_size;
                modelInfo->output_size = outputs_size;
                modelInfo->supported_batch_sizes = supported_batch_sizes;
                modelInfo->weights_size = weights_size;
                modelInfo->num_weights_pages = spec->configured_weights_page_size;
                modelInfo->weights_load_time_nanos = weights_load_time_nanos;
                modelInfo->batch_size_exec_times_nanos = batch_size_exec_times_nanos;
                RuntimeModelDummy* rm = new RuntimeModelDummy(modelInfo,gpu_id,weights_pages_count);

                bool success = myManager->models->put_if_absent(
                    loadmodel->model_id + i, 
                    gpu_id, 
                    rm
                );
                CHECK(success) << "Loaded models changed while loading from disk";
            }
        }

        end = util::now();

        //Create success result
        auto result = std::make_shared<workerapi::LoadModelFromDiskResult>();
        result->id = loadmodel->id;
        result->action_type = workerapi::loadModelFromDiskAction;
        result->status = actionSuccess;
        result->input_size = inputs_size;
        result->output_size = outputs_size;
        result->copies_created = loadmodel->no_of_copies;
        result->weights_load_time_nanos = weights_load_time_nanos;
        result->supported_batch_sizes = supported_batch_sizes;
        result->batch_size_exec_times_nanos = batch_size_exec_times_nanos;
        // TODO Verify: I assume that GPU-specific weights_caches have identical page_size
        size_t page_size = myManager->weights_caches[0]->page_size;
        result->num_weights_pages = weights_pages_count;
        result->weights_size_in_cache = result->num_weights_pages * page_size;

        success(result);

    }catch (dmlc::Error &errMessage) {
        error(actionErrorInvalidModelPath, errMessage.what());
        return;
    }catch(NoMeasureFile &errMessage){
        error(errMessage.status_code, errMessage.message);
        return;
    }
    
}

void LoadWeightsDummyAction::run(){
    start = util::now();

    //Check timestamp for running task
    std::stringstream err;
    if(start < loadweights->earliest){
        err << "LoadWeights ran before it was eligible"
            << " (now " << util::millis(start)
            << ", earliest " << util::millis(loadweights->earliest) << ")";
        error(loadWeightsTooEarly, err.str());
        return;

    }else if(start > loadweights->latest){
        err << "LoadWeights could not start in time"
            << " (now " << util::millis(start)
            << ", latest " << util::millis(loadweights->latest) << ")";
        error(loadWeightsTooLate, err.str());
        return;
    }

    //Check if target model is present 
    RuntimeModelDummy* rm = myManager->models->get(loadweights->model_id, loadweights->gpu_id);
    if (rm == nullptr) {
        std::string message = "LoadWeightsTask could not find model";
        message += " with model ID " + std::to_string(loadweights->model_id);
        message += " and GPU ID " + std::to_string(loadweights->gpu_id);
        error(loadWeightsUnknownModel, message);
        return;
    }

    //Alloc weights and update version, leave weights evicted mark unchanged for now
    rm->lock();
    std::atomic_bool alloc_success = true;
    if (!rm->weights) {
        alloc_success = myManager->weights_caches[loadweights->gpu_id]->alloc(rm->weightspagescount);
    }
    version = ++rm->version;
    rm->unlock();

    if(!alloc_success){
        error(loadWeightsInsufficientCache, "LoadWeightsTask failed to allocate pages from cache");
        return;
    }

    end = start + rm->modelinfo->weights_load_time_nanos;

    toComplete();
}

void LoadWeightsDummyAction::process_completion(){
    RuntimeModelDummy* rm = myManager->models->get(loadweights->model_id, loadweights->gpu_id);
    //Check if model weight is changed upon completion
    bool version_unchanged = false;
    rm->lock();
    if (rm->version == version) {
        version_unchanged = true;
    }
    rm->unlock();
    if (version_unchanged) {
        rm->lock();
        rm->weights = true;
        rm->unlock();

        auto result = std::make_shared<workerapi::LoadWeightsResult>();
        success(result);
    }else {
        error(loadWeightsConcurrentModification, "Model weights were modified while being copied");
    }
}

void EvictWeightsDummyAction::run(){
    start = util::now();

    //Check timestamp for running task
    std::stringstream err;
    if(start < evictweights->earliest){
        err << "EvictWeights ran before it was eligible"
            << " (now " << util::millis(start)
            << ", earliest " << util::millis(evictweights->earliest) << ")";
        error(evictWeightsTooEarly, err.str());
        return;

    }else if(start > evictweights->latest){
        err << "EvictWeights could not start in time"
            << " (now " << util::millis(start)
            << ", latest " << util::millis(evictweights->latest) << ")";
        error(evictWeightsTooLate, err.str());
        return;
    }

    //Check if target model is present
    RuntimeModelDummy* rm = myManager->models->get(evictweights->model_id, evictweights->gpu_id);
    if (rm == nullptr) {
        error(evictWeightsUnknownModel, "EvictWeightsTask could not find model with specified id");
        return;
    }

    //Check if target model has weight
    if (!rm->weights) {
        error(evictWeightsNotInCache, "EvictWeightsTask not processed because no weights exist");
        return;
    }

    rm->lock();
    rm->version++;
    rm->weights = false;
    rm->unlock();
    myManager->weights_caches[evictweights->gpu_id]->free(rm->weightspagescount);

    end = util::now();

    auto result = std::make_shared<workerapi::EvictWeightsResult>();
    success(result);
}

void InferDummyAction::run(){
    start = util::now();

    //Check timestamp for running task
    std::stringstream err;
    if(start < infer->earliest){
        err << "Infer ran before it was eligible"
            << " (now " << util::millis(start)
            << ", earliest " << util::millis(infer->earliest) << ")";
        error(execTooEarly, err.str());
        return;
    }else if(start > infer->latest){
        err << "Infer could not start in time"
            << " (now " << util::millis(start)
            << ", latest " << util::millis(infer->latest) << ")";
        error(execTooLate, err.str());
        return;
    }

    //Check if target model is present
    RuntimeModelDummy* rm = myManager->models->get(infer->model_id, infer->gpu_id);
    if (rm == nullptr) {
        error(copyInputUnknownModel, "CopyInputTask could not find model with specified id");
        return;
    }
    //Cauculate legal padded_batch_size. return padded_batch_size_index = -1 if given batch_size is not supported at all. This index is the same as the index for batch_size_exec_times_nanos.
    int padded_batch_size_index = rm->padded_batch_size_index(infer->batch_size);
    if (padded_batch_size_index == -1) {
        err << "CopyInputTask received unsupported batch size " << infer->batch_size;
        error(copyInputInvalidBatchSize, err.str());
        return;
    }
    if (infer->input_size == 0 && myManager->allow_zero_size_inputs) {
        // Used in testing; allow client to send zero-size inputs and generate worker-side
    }else if (rm->input_size(infer->batch_size) != infer->input_size && infer->input_sizes.size() == 0) {
        // Normal behavior requires correctly sized inputs
        err << "CopyInputTask received incorrectly sized input"
            << " (expected " << rm->input_size(infer->batch_size) 
            << ", got " << infer->input_size
            << " (batch_size=" << infer->batch_size << ")";
        error(copyInputInvalidInput, err.str());
        return;
    }

    //Check if target model's weight is present
    if (rm->weights == false) {
        error(execWeightsMissing, "ExecTask failed due to missing model weights");
        return;
    }

    rm->lock();
    version = rm->version;
    rm->unlock();
    end = start + rm->modelinfo->batch_size_exec_times_nanos[padded_batch_size_index];

    toComplete();
}

void InferDummyAction::process_completion(){
    RuntimeModelDummy* rm = myManager->models->get(infer->model_id, infer->gpu_id);
    //Check if weight is changed during infer
    bool version_unchanged = false;
    rm->lock();
    if (rm->version == version && rm->weights) {
        version_unchanged = true;
    }
    int output_size = rm->output_size(infer->batch_size);
    rm->unlock();
    if (version_unchanged) {
        auto result = std::make_shared<workerapi::InferResult>();
        result->output_size = output_size;
        success(result);
    } else {
        error(execConcurrentWeightsModification, "ExecTask failed due to weights version mismatch");
    }
}

}