#ifndef _CLOCKWORK_WORKLOAD_SCALABILITY_H_
#define _CLOCKWORK_WORKLOAD_SCALABILITY_H_

namespace clockwork {
namespace workload {

Engine* scalability_experiment_1(
  clockwork::Client* client, unsigned num_models, double rate_min,
  double rate_max, double rate_inc_factor, std::string rate_inc_op,
  unsigned period) {

  Engine* engine = new Engine();
	std::string model_name = "resnet50_v2";
	std::string modelpath = util::get_clockwork_modelzoo()[model_name];
	auto models = client->load_remote_models(modelpath, num_models);

  double rate_min_per_model = rate_min / num_models;
  double rate_max_per_model = rate_max / num_models;

  std::vector<OpenLoop<Exponential>*> workloads;
	for (unsigned i = 0; i < models.size(); i++) {
    OpenLoop<Exponential>* workload = new PoissonOpenLoop(
			i,                  // client id
			models[i],          // model
			i,                  // rng seed
			rate_min_per_model  // requests/second
		);
    engine->AddWorkload(workload);
    workloads.push_back(workload);
	}

	if (rate_inc_op == "add") {
    double rate_inc_factor_per_model = rate_inc_factor / num_models;
    engine->AddWorkload(new AdjustPoissonRate(
      period, // period in seconds
      rate_min_per_model,
      workloads,
      [rate_inc_factor_per_model] (double current) {
        return current + rate_inc_factor_per_model;
      },
      [rate_max_per_model] (double current) {
        return current > rate_max_per_model;
      }
    ));
	} else if (rate_inc_op == "mul") {
    double rate_inc_factor_per_model = rate_inc_factor;
    engine->AddWorkload(new AdjustPoissonRate(
      period, // period in seconds
      rate_min_per_model,
      workloads,
      [rate_inc_factor_per_model] (double current) {
        return current * rate_inc_factor_per_model;
      },
      [rate_max_per_model] (double current) {
        return current > rate_max_per_model;
      }
    ));
	} else {
		CHECK(false) << rate_inc_op << " operator not yet implemented";
  }

	return engine;
}

Engine* azure_scalability_exp(clockwork::Client* client, unsigned num_workers,
    double load_factor_min, double load_factor_max, double load_factor_inc,
    unsigned load_factor_period, unsigned memory_load_factor) {

  bool use_all_models = true;
  unsigned interval_duration_seconds = 60;
  unsigned trace_id = 1;
  bool randomise_start = false;

	std::cout << "Running azure_scalability_exp workload with"
			  << " num_workers=" << num_workers
			  << " load_factor_min=" << load_factor_min
			  << " load_factor_max=" << load_factor_max
			  << " load_factor_inc=" << load_factor_inc
			  << " load_factor_period=" << load_factor_period
			  << " memory_load_factor=" << memory_load_factor
			  << std::endl;

	unsigned estimated_max_models = 3100;
	double estimated_load_factor_per_worker = 0.125;

	unsigned copies_per_model;
	if (memory_load_factor == 1) {
		if (use_all_models)
			copies_per_model = 3;
		else
			copies_per_model = 200;
	} else if (memory_load_factor == 2) {
		if (use_all_models)
			copies_per_model = 13;
		else
			copies_per_model = 800;
	} else if (memory_load_factor == 3) {
		if (use_all_models)
			copies_per_model = 29;
		else
			copies_per_model = 1800;
	} else if (memory_load_factor == 4) {
		if (use_all_models)
			copies_per_model = 66;
		else
			copies_per_model = 3600;
	}

	auto modelzoo = util::get_clockwork_modelzoo();
	unsigned num_models = use_all_models ? modelzoo.size() : 1;
	unsigned total_models = num_models * copies_per_model;

	std::cout << "Using " << total_models << " models (" << copies_per_model
    << " copies of " << num_models << " models)" << std::endl;

	double scale_factor_min = load_factor_min * estimated_load_factor_per_worker * num_workers;
	double scale_factor_max = load_factor_max * estimated_load_factor_per_worker * num_workers;
	double scale_factor_inc = load_factor_inc;
	double scale_factor_period = load_factor_period;

	std::cout << "Replaying trace " << trace_id
    << " at scale_factor=[" << scale_factor_min << ", " << scale_factor_max << ")"
    << " to achieve load_factor=[" << load_factor_min << ", " << load_factor_max << ")"
    << " on " << num_workers << " workers" << std::endl;

	std::cout << interval_duration_seconds << " second intervals" << std::endl;
	std::cout << load_factor_period << " seconds between scale factor increments" << std::endl;

  // The following code is copied frmo azure_parameterized, with adaditinoal
  // changes for adjusting the scale factor parameter

  bool stripe = false;

	Engine* engine = new Engine();

	auto trace_data = azure::load_trace(trace_id);

	if (total_models > 3100) {
		std::cout << "WARNING: 3100 is about the GPU memory limit for models. "
				  << "Got " << total_models
				  << ".  You might get cuda device memory exhaustion errors."
				  << std::endl;
	}

	srand(0);
	std::vector<Model*> models;
	//auto modelzoo = util::get_clockwork_modelzoo();
	if (use_all_models) {
		unsigned models_remaining = modelzoo.size();
		for (auto &p : modelzoo) {
			unsigned num_copies = total_models / models_remaining;
			total_models -= num_copies;
			models_remaining--;
			std::cout << "Loading " << p.first << " x" << num_copies << std::endl;
			for (auto &model : client->load_remote_models(p.second, num_copies)) {
				// Pseudo-randomly insert each model
				unsigned position = models.size() % rand();
				models.insert(models.begin() + position, model);
			}
		}
	} else {
		std::string model = "resnet50_v2";
		std::cout << "Loading " << model << " x" << total_models << std::endl;
		for (auto &model : client->load_remote_models(modelzoo[model], total_models)) {
			models.push_back(model);
		}
	}

  std::vector<TraceReplay<Exponential>*> workloads;
	for (unsigned i = 0; i < trace_data.size(); i++) {
		Model* model;
		if (stripe) {
			model = models[i % models.size()];
		} else {
			model = models[(models.size() * i) / trace_data.size()];
		}
		
		auto workload = trace_data[i];

		TraceReplay<Exponential>* replay = new PoissonTraceReplay(
			i,                          // client id, just give them all the same ID for this example
			model,                      // model
			i,                          // rng seed
			workload,                   // trace data
			scale_factor_min,           // scale factor; default 1
			interval_duration_seconds,  // interval duration; default 60
			randomise_start ? -1 : 0    // interval to begin at; default 0; set to -1 for random
		);

		engine->AddWorkload(replay);
    workloads.push_back(replay);
	}

  engine->AddWorkload(new AdjustScaleFactor(
    scale_factor_period, // period in seconds
    scale_factor_min,
    workloads,
    [scale_factor_inc] (double current) { return current * scale_factor_inc; },
    [scale_factor_max] (double current) { return current > scale_factor_max; }
  ));

	return engine;
}

}
}

#endif
