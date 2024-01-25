#include "clockwork/test/actions.h"

namespace clockwork {

std::atomic_int id_seed(0);

std::shared_ptr<workerapi::LoadModelFromDisk> load_model_from_disk_action() {
    auto action = std::make_shared<workerapi::LoadModelFromDisk>();
    action->id = id_seed++;
    action->action_type = workerapi::loadModelFromDiskAction;
    action->model_id = 0;
    action->model_path = clockwork::util::get_example_model();
	action->no_of_copies = 1;
    action->earliest = util::now();
    action->latest = util::now() + 1000000000;
    return action;
}

std::shared_ptr<workerapi::LoadWeights> load_weights_action(int model_id) {
    auto action = std::make_shared<workerapi::LoadWeights>();
    action->id = id_seed++;
    action->action_type = workerapi::loadWeightsAction;
    action->earliest = util::now();
    action->latest = util::now() + 1000000000;
    action->expected_duration = 0;
    action->model_id = model_id;
    action->gpu_id = 0;
    return action;
}

std::shared_ptr<workerapi::EvictWeights> evict_weights_action() {
    auto action = std::make_shared<workerapi::EvictWeights>();
    action->id = id_seed++;
    action->action_type = workerapi::evictWeightsAction;
    action->earliest = util::now();
    action->latest = util::now() + 1000000000;
    action->model_id = 0;
    action->gpu_id = 0;
    return action;
}

std::shared_ptr<workerapi::Infer> infer_action() {
    auto action = std::make_shared<workerapi::Infer>();
    action->id = id_seed++;
    action->action_type = workerapi::inferAction;
    action->earliest = util::now();
    action->latest = util::now() + 1000000000;
    action->expected_duration = 0;
    action->model_id = 0;
    action->gpu_id = 0;
    action->batch_size = 1;
    action->input_size = 602112; // Hard coded for the example model
    action->input = static_cast<char*>(malloc(602112));
    return action;
}

std::shared_ptr<workerapi::Infer> infer_action2(ClockworkWorker* worker) {
    auto action = std::make_shared<workerapi::Infer>();
    action->id = id_seed++;
    action->action_type = workerapi::inferAction;
    action->earliest = util::now();
    action->latest = util::now() + 1000000000;
    action->expected_duration = 0;
    action->model_id = 0;
    action->gpu_id = 0;
    action->batch_size = 1;
    action->input_size = 602112; // Hard coded for the example model
    action->input = worker->runtime->manager->host_io_pool->alloc(action->input_size);
    return action;
}

std::shared_ptr<workerapi::Infer> infer_action(int batch_size, BatchedModel* model) {
    auto action = infer_action();
    action->batch_size = batch_size;
    action->input_size = model->input_size(batch_size);
    action->input = static_cast<char*>(malloc(model->input_size_with_padding(batch_size)));
    return action;
}

std::shared_ptr<workerapi::GetWorkerState> get_worker_state_action() {
    auto action = std::make_shared<workerapi::GetWorkerState>();
	action->id = id_seed++;
	action->action_type = workerapi::getWorkerStateAction;
	return action;
}

}
