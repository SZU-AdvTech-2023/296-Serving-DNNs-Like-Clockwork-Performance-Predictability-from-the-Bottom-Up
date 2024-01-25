#include "clockwork/workload/workload.h"
#include "clockwork/api/api_common.h"
#include <dmlc/logging.h>
#include "clockwork/util.h"

using namespace clockwork::workload;

void Engine::AddWorkload(Workload* workload, uint64_t start_after) {
	workloads.push_back(workload);
	workload->SetEngine(this);
	workload->start_after = start_after;
}

void Engine::SetTimeout(uint64_t timeout, std::function<void(void)> callback) {
	if (timeout == 0) callback();
	else queue.push(element{now + timeout, callback});
}

void Engine::InferComplete(Workload* workload, unsigned model_index) {
	auto callback = [this, workload, model_index]() {
		workload->InferComplete(now, model_index);
	};
	runqueue.push(callback);
}

void Engine::InferError(Workload* workload, unsigned model_index, int status) {
	std::function<void(void)> callback;
	if (status == clockworkInitializing) {
		callback = [this, workload, model_index]() {
			workload->InferErrorInitializing(now, model_index);
		};
	} else {
		callback = [this, workload, model_index, status]() {
			workload->InferError(now, model_index, status);
		};
	}
	runqueue.push(callback);
}

void Engine::Run(clockwork::Client* client) {
	while (true) {
		try {
			auto models = client->ls();
			break;
		} catch (const clockwork_initializing& e1) {
			std::cout << "Clockwork initializing, retrying " << e1.what() << std::endl;
			usleep(1000000);
		} catch (const std::runtime_error& e2) {
			std::cout << "LS error: " << e2.what() << std::endl;
			exit(1);
		}
	}

	now = util::now();
	for (auto &workload : workloads) {
		running++;
		SetTimeout(workload->start_after, [this, workload]() { workload->Start(now); });
	}
	while (running > 0) {
		// Process all pending results
		now = util::now();
		std::function<void(void)> callback;
		while (runqueue.try_pop(callback)) {
			callback();
		}

		// Run one next request if available
		now = util::now();
		if (!queue.empty() && queue.top().ready <= now) {
			auto next = queue.top();
			queue.pop();
			now = next.ready;
			next.callback();
		} else {
			usleep(1);
			now = util::now();
		}
	}
}

Workload::Workload(int id) : user_id(id) {
}

Workload::Workload(int id, clockwork::Model* model) : user_id(id) {
	AddModel(model);
}

Workload::Workload(int id, std::vector<clockwork::Model*> &models) : user_id(id) {
	AddModels(models);
}

void Workload::AddModel(clockwork::Model* model) {
	model->set_user_id(user_id);
	models.push_back(model);
}

void Workload::AddModels(std::vector<clockwork::Model*> &models) {
	for (auto &model : models) {
		AddModel(model);
	}
}

void Workload::SetEngine(Engine* engine) {
	this->engine = engine;
}

void Workload::Infer(unsigned model_index) {
	CHECK(model_index < models.size()) << "Workload " << user_id
		<< " inferring on non-existent model ";
	auto &model = models[model_index];

	std::string& generated = engine->input_generator.getPrecompressedInput(model->input_size());
	uint8_t* ptr = static_cast<uint8_t*>(static_cast<void*>(generated.data()));
	std::vector<uint8_t> input(ptr, ptr+generated.size());


	auto onSuccess = [this, model_index](std::vector<uint8_t> &output) {
		engine->InferComplete(this, model_index);
	};

	auto onError = [this, model_index](int status, std::string message) {
		engine->InferError(this, model_index, status);
	};

	model->infer(input, onSuccess, onError, true);
}

void Workload::SetTimeout(uint64_t timeout, std::function<void(void)> callback) {
	engine->SetTimeout(timeout, callback);
}


void Workload::InferErrorInitializing(uint64_t now, unsigned model_index) {
	InferError(now, model_index, clockworkInitializing);
}

ClosedLoop::ClosedLoop(int id, clockwork::Model* model, unsigned concurrency) :
	Workload(id, model), concurrency(concurrency), num_requests(UINT64_MAX),
	jitter(0), idx(0) {
	CHECK(concurrency != 0) << "ClosedLoop with concurrency 0 created";
	CHECK(num_requests != 0) << "ClosedLoop with num_requests 0 created";
}

ClosedLoop::ClosedLoop(int id, std::vector<clockwork::Model*> models,
	unsigned concurrency) :
		Workload(id, models), concurrency(concurrency),
		num_requests(UINT64_MAX), jitter(0), idx(0) {
	CHECK(concurrency != 0) << "ClosedLoop with concurrency 0 created";
	CHECK(num_requests != 0) << "ClosedLoop with num_requests 0 created";
}

ClosedLoop::ClosedLoop(int id, clockwork::Model* model, unsigned concurrency,
	uint64_t num_requests, uint64_t jitter) :
	Workload(id, model), concurrency(concurrency), num_requests(num_requests),
	jitter(jitter), idx(0) {
	CHECK(concurrency != 0) << "ClosedLoop with concurrency 0 created";
	CHECK(num_requests != 0) << "ClosedLoop with num_requests 0 created";
}

ClosedLoop::ClosedLoop(int id, std::vector<clockwork::Model*> models,
	unsigned concurrency, uint64_t num_requests, uint64_t jitter) :
	Workload(id, models), concurrency(concurrency), num_requests(num_requests),
	jitter(jitter), idx(0) {
	CHECK(concurrency != 0) << "ClosedLoop with concurrency 0 created";
	CHECK(num_requests != 0) << "ClosedLoop with num_requests 0 created";
}

void ClosedLoop::Start(uint64_t now) {
	if (jitter > 0) { // Dealyed start
		std::cout << "Model " << user_id << "'s start delayed by " << jitter
				  << " seconds" << std::endl;
		SetTimeout(jitter * 1000000000, [this]() { ActualStart(); });
	} else { // Immediate start
		ActualStart();
	}
}

void ClosedLoop::ActualStart() {
	for (unsigned i = 0; i < concurrency; i++) {
		Infer(GetAndUpdateIdx());
	}
}

void ClosedLoop::InferComplete(uint64_t now, unsigned model_index) {
	if ((num_requests--) > 0) { Infer(GetAndUpdateIdx()); } else { engine->running--; }
}

void ClosedLoop::InferError(uint64_t now, unsigned model_index, int status) {
	if ((num_requests--) > 0) { Infer(GetAndUpdateIdx()); } else { engine->running--; }
}

void ClosedLoop::InferErrorInitializing(uint64_t now, unsigned model_index) {
	if ((num_requests--) > 0) { Infer(GetAndUpdateIdx()); } else { engine->running--; }
}

unsigned ClosedLoop::GetAndUpdateIdx() {
	unsigned ret_idx = idx;
	idx = (idx + 1) % models.size();
	return ret_idx;
}
