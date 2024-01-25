#ifndef _CLOCKWORK_CONTROLLER_STRESS_TEST_CONTROLLER_H_
#define _CLOCKWORK_CONTROLLER_STRESS_TEST_CONTROLLER_H_

#include "clockwork/network/controller.h"
#include "clockwork/api/worker_api.h"
#include "controller.h"
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <unordered_map>
#include <unistd.h>
#include <sys/types.h>
#include <tbb/concurrent_queue.h>

using namespace clockwork;

bool log_actions = false;

class StressTestGPU {
public:
	unsigned worker_id;
	unsigned worker_gpu_id;
	int outstanding_loadweights;
	int outstanding_infer;
	std::deque<unsigned> models_on_gpu;
	std::deque<unsigned> models_not_on_gpu;

	unsigned load_weights_errors = 0;
	unsigned infer_errors = 0;
	std::vector<uint64_t> load_weights_measurements;
	std::vector<uint64_t> load_weights_e2e;
	std::vector<uint64_t> infer_measurements;
	std::vector<uint64_t> infer_e2e;
	StressTestGPU(): 
		models_on_gpu(), 
		models_not_on_gpu(), 
		load_weights_measurements(), 
		load_weights_e2e(),
		infer_measurements(), 
		infer_e2e(),
		outstanding_loadweights(0),
		outstanding_infer(0) {}
};

// Simple controller implementation that send requests to worker and wait for response
class StressTestController : public Controller {
public:
	bool stress_infer = true;
	bool stress_loadweights = true;
	bool send_inputs = false;
	bool trigger_load_errors = false;

	std::string model_path = "/home/jcmace/clockwork-modelzoo-volta/resnet50_v2/model";
	unsigned duplicates = 40;
	unsigned max_models_on_gpu = 20;
	size_t input_size = send_inputs ? 602112 : 0;
	char* input;

	int max_outstanding_loadweights = 2;
	int max_outstanding_infer = 2;

	unsigned num_gpus = 2;

	std::atomic_int action_id_seed;

	std::recursive_mutex mutex;
	uint64_t profiled_load_weights;
	uint64_t profiled_exec;
	std::vector<StressTestGPU> gpus;

	unsigned pending_workers;

	std::thread thread;
	std::thread printer;

	std::map<unsigned, std::function<void(std::shared_ptr<workerapi::Result>)>> callbacks;

	tbb::concurrent_queue<std::shared_ptr<workerapi::Result>> results;

	StressTestController(int client_port, std::vector<std::pair<std::string, std::string>> worker_host_port_pairs):
		Controller::Controller(client_port, worker_host_port_pairs), action_id_seed(0), 
		pending_workers(workers.size()),
		results() {
		input = static_cast<char*>(malloc(input_size));

		thread = std::thread(&StressTestController::run, this);
		threading::initHighPriorityThread(thread);
	}

	std::string stats(std::vector<uint64_t> v, std::vector<uint64_t> e2e, uint64_t profiled, uint64_t duration) {
		if (v.size() == 0) {
			return "throughput=0";
		}

		const auto [min, max] = std::minmax_element(v.begin(), v.end());
		const auto [e2emin, e2emax] = std::minmax_element(e2e.begin(), e2e.end());
		int count = v.size();
		double sum = std::accumulate(v.begin(), v.end(), 0.0);
		double sume2e = std::accumulate(e2e.begin(), e2e.end(), 0.0);
		double throughput = count * 1000000000.0 / static_cast<double>(duration);

		std::stringstream s;
		s << std::fixed << std::setprecision(2);
		s << "profiled=" << profiled << " min=" << *min << " max=" << *max << " mean=" << (sum/count) << " e2e=" << (sume2e/count) << " e2emax=" << *e2emax << " throughput=" << throughput << " utilization=" << (sum/((float) duration));
		return s.str();
	}

	void printerThread() {
		uint64_t last_print = util::now();
		uint64_t print_interval = 1000000000UL;

		while (true) {
			if (last_print + print_interval > util::now()) {
				usleep(10000);
				continue;
			}

			std::vector<StressTestGPU> gpu_copies;
			uint64_t now;
			{
				std::lock_guard<std::recursive_mutex> lock(mutex);

				now = util::now();

				for (StressTestGPU &gpu : this->gpus) {
					gpu_copies.push_back(gpu);
					gpu.load_weights_measurements.clear();
					gpu.load_weights_e2e.clear();
					gpu.infer_measurements.clear();
					gpu.infer_e2e.clear();
					gpu.infer_errors = 0;
					gpu.load_weights_errors = 0;
				}
			}
			uint64_t duration = now - last_print;
			last_print = now;

			std::stringstream report;
			for (auto &gpu : gpu_copies) {
				report << "LoadWeights W" << gpu.worker_id << " GPU" << gpu.worker_gpu_id
				       << " errors=" << gpu.load_weights_errors 
				       << " " << stats(gpu.load_weights_measurements, gpu.load_weights_e2e, profiled_load_weights, duration) << std::endl;
			}
			for (auto &gpu : gpu_copies) {
				report << "Infer W" << gpu.worker_id << " GPU" << gpu.worker_gpu_id 
					   << " errors=" << gpu.infer_errors 
					   << " " << stats(gpu.infer_measurements, gpu.infer_e2e, profiled_exec, duration) << std::endl;
			}

			std::cout << report.str();
		}
	}
	
	void save_callback(unsigned action_id, std::function<void(std::shared_ptr<workerapi::Result>)> callback) {
		auto it = callbacks.find(action_id);
		CHECK(it == callbacks.end()) << "Action " << action_id << " already exists";
		callbacks.insert(std::make_pair(action_id, callback));
	}

	void callback(std::shared_ptr<workerapi::Result> result) {
		auto it = callbacks.find(result->id);
		CHECK(it != callbacks.end()) << "Received result with no callback";

		auto f = it->second;
		callbacks.erase(it);

		f(result);
	}

	void init() {
		for (unsigned worker_id = 0; worker_id < this->workers.size(); worker_id++) {
			unsigned action_id = action_id_seed++;

			auto load_model = std::make_shared<workerapi::LoadModelFromDisk>();
			load_model->id = action_id;
			load_model->model_id = 0;
			load_model->model_path = model_path;
			load_model->earliest = 0;
			load_model->latest = util::now() + 600000000000UL;
			load_model->no_of_copies = duplicates;

			save_callback(action_id, std::bind(&StressTestController::onLoadModelsComplete, this, worker_id, std::placeholders::_1));

			this->workers[worker_id]->sendAction(load_model);
		}
	}

	void onLoadModelsComplete(unsigned worker_id, std::shared_ptr<workerapi::Result> result) {
		std::lock_guard<std::recursive_mutex> lock(mutex);

		if (auto error = std::dynamic_pointer_cast<workerapi::ErrorResult>(result)) {
			CHECK(false) << "StressTestController unable to load initial models " << result->str();
		} else if (auto load = std::dynamic_pointer_cast<workerapi::LoadModelFromDiskResult>(result)) {
			profiled_load_weights = load->weights_load_time_nanos;
			profiled_exec = load->batch_size_exec_times_nanos[0];
			for (unsigned i = 0; i < num_gpus; i++) {
				StressTestGPU gpu;
				gpu.worker_id = worker_id;
				gpu.worker_gpu_id = i;
				for (unsigned i = 0; i < duplicates; i++) {
					gpu.models_not_on_gpu.push_back(i);
				}
				gpus.push_back(gpu);
			}
			pending_workers--;
		} else {
			CHECK(false) << "Unexpected response to EvictWeights action";
		}
	}

	bool evictNext(unsigned gpu_id) {
		auto &gpu = gpus[gpu_id];

		if (gpu.models_on_gpu.size() <= max_models_on_gpu) return false;

		unsigned model_id = gpu.models_on_gpu.front();
		gpu.models_on_gpu.pop_front();

		unsigned action_id = action_id_seed++;

		auto evict = std::make_shared<workerapi::EvictWeights>();
		evict->id = action_id;
		evict->model_id = model_id;
		evict->gpu_id = gpu.worker_gpu_id;
		evict->earliest = util::now();
		evict->latest = util::now() + 10000000000UL; // 10s

		save_callback(action_id, std::bind(&StressTestController::onEvictWeightsComplete, this, model_id, gpu_id, std::placeholders::_1));

		this->workers[gpu.worker_id]->sendAction(evict);

		return true;
	}

	void onEvictWeightsComplete(unsigned model_id, unsigned gpu_id, std::shared_ptr<workerapi::Result> result) {
		if (auto error = std::dynamic_pointer_cast<workerapi::ErrorResult>(result)) {
			CHECK(false) << "Error in evict weights action, which should never happen except for fatal errors";
		} else if (auto evict = std::dynamic_pointer_cast<workerapi::EvictWeightsResult>(result)) {
			auto &gpu = gpus[gpu_id];
			gpu.models_not_on_gpu.push_back(model_id);
		} else {
			CHECK(false) << "Unexpected response to EvictWeights action";
		}
	}

	bool loadNext(unsigned gpu_id) {
		auto &gpu = gpus[gpu_id];

		if (gpu.models_not_on_gpu.size() == 0) return false;
		if (gpu.outstanding_loadweights >= max_outstanding_loadweights) return false;

		unsigned model_id = gpu.models_not_on_gpu.front();
		gpu.models_not_on_gpu.pop_front();

		unsigned to_submit = trigger_load_errors ? 2 : 1;
		for (unsigned i = 0; i < to_submit; i++) {
			unsigned action_id = action_id_seed++;

			auto load = std::make_shared<workerapi::LoadWeights>();
			load->id = action_id;
			load->gpu_id = gpu.worker_gpu_id;
			load->model_id = model_id;
			load->earliest = util::now();
			load->latest = util::now() + 10000000000UL; // 10s

			save_callback(action_id, std::bind(&StressTestController::onLoadWeightsComplete, this, model_id, gpu_id, util::now(), std::placeholders::_1));

			if (log_actions) std::cout << "S: " << load->str() << std::endl;
			this->workers[gpu.worker_id]->sendAction(load);
			gpu.outstanding_loadweights++;
		}


		return true;
	}

	void onLoadWeightsComplete(unsigned model_id, unsigned gpu_id, uint64_t submitted_at, std::shared_ptr<workerapi::Result> result) {
		std::lock_guard<std::recursive_mutex> lock(mutex);

		auto &gpu = gpus[gpu_id];
		if (auto error = std::dynamic_pointer_cast<workerapi::ErrorResult>(result)) {
			std::cout << "LoadWeights error " << result->str() << std::endl;
			gpu.load_weights_errors++;
			gpu.models_not_on_gpu.push_back(model_id);
		} else if (auto load = std::dynamic_pointer_cast<workerapi::LoadWeightsResult>(result)) {
			gpu.models_on_gpu.push_back(model_id);
			gpu.load_weights_measurements.push_back(load->duration);
			gpu.load_weights_e2e.push_back(util::now() - submitted_at);
		} else {
			CHECK(false) << "Unexpected response to LoadWeights action";
		}
		gpu.outstanding_loadweights--;
	}

	bool inferNext(unsigned gpu_id) {
		auto &gpu = gpus[gpu_id];
		if (gpu.outstanding_infer >= max_outstanding_infer) return false;

		unsigned next_model_ix = 6 + gpu.outstanding_infer;
		if (gpu.models_on_gpu.size() <= next_model_ix) return false;

		unsigned model_id = gpu.models_on_gpu[next_model_ix++];
		unsigned action_id = action_id_seed++;

		auto infer = std::make_shared<workerapi::Infer>();
		infer->id = action_id;
		infer->model_id = model_id;
		infer->gpu_id = gpu.worker_gpu_id;
		infer->batch_size = 1;
		infer->input = input;
		infer->input_size = input_size;
		infer->earliest = util::now();
		infer->latest = util::now() + 10000000000UL; // 10s

		save_callback(action_id, std::bind(&StressTestController::onInferComplete, this, model_id, gpu_id, util::now(), std::placeholders::_1));


		if (log_actions) std::cout << "S: " << infer->str() << std::endl;
		this->workers[gpu.worker_id]->sendAction(infer);

		gpu.outstanding_infer++;

		return true;
	}

	void onInferComplete(unsigned model_id, unsigned gpu_id, uint64_t submitted_at, std::shared_ptr<workerapi::Result> result) {
		std::lock_guard<std::recursive_mutex> lock(mutex);

		auto &gpu = gpus[gpu_id];
		if (auto error = std::dynamic_pointer_cast<workerapi::ErrorResult>(result)) {
			std::cout << "Infer error " << result->str() << std::endl;
			gpu.infer_errors++;
		} else if (auto infer = std::dynamic_pointer_cast<workerapi::InferResult>(result)) {
			gpu.infer_measurements.push_back(infer->exec.duration);
			gpu.infer_e2e.push_back(util::now() - submitted_at);
		} else {
			CHECK(false) << "Unexpected response to LoadWeights action";
		}
		gpu.outstanding_infer--;
	}

	virtual void sendResult(std::shared_ptr<workerapi::Result> result) {
		if (log_actions) std::cout << "R: " << result->str() << std::endl;

		results.push(result);
	}


	void run() {
		init();

		std::shared_ptr<workerapi::Result> result;

		while (pending_workers > 0) {
			if (results.try_pop(result)) {
				callback(result);
			}
		}

		printer = std::thread(&StressTestController::printerThread, this);
		threading::initLoggerThread(printer);

		while (true) {
			while (results.try_pop(result)) {
				callback(result);
			}

			for (unsigned i = 0; i < gpus.size(); i++) {
				if (stress_loadweights) {
					while (evictNext(i));	
				}
				while (loadNext(i));
				if (stress_infer) {
					while (inferNext(i));
				}
			}
		}
	}

	virtual void infer(clientapi::InferenceRequest &request, std::function<void(clientapi::InferenceResponse&)> callback) {
		CHECK(false) << "infer from client not supported";
	}
	virtual void uploadModel(clientapi::UploadModelRequest &request, std::function<void(clientapi::UploadModelResponse&)> callback) {
		CHECK(false) << "uploadModel not supported";
	}
	virtual void evict(clientapi::EvictRequest &request, std::function<void(clientapi::EvictResponse&)> callback) {
		CHECK(false) << "evict not supported";
	}
	virtual void loadRemoteModel(clientapi::LoadModelFromRemoteDiskRequest &request, std::function<void(clientapi::LoadModelFromRemoteDiskResponse&)> callback) {
		CHECK(false) << "loadRemoteModel not supported";
	}
};

#endif