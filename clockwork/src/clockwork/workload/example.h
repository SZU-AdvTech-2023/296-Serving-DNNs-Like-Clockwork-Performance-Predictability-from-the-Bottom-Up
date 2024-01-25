#ifndef _CLOCKWORK_WORKLOAD_EXAMPLE_H_
#define _CLOCKWORK_WORKLOAD_EXAMPLE_H_

#include "clockwork/util.h"
#include "clockwork/workload/workload.h"
#include "clockwork/workload/azure.h"
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace clockwork {
namespace workload {

Engine* comparison_experiment(clockwork::Client* client,
								int num_models,
								int total_rps) {
	Engine* engine = new Engine();

	std::string modelpath = util::get_clockwork_modelzoo()["resnet50_v2"];
	auto models = client->load_remote_models(modelpath, num_models);

	double rate = total_rps / ((double) num_models);

	for (int i = 0; i < num_models; i++) {
		engine->AddWorkload(new PoissonOpenLoop(
			i,				// client id
			models[i],  	// model
			i,      		// rng seed
			rate			// requests/second
		));
	}

	return engine;
}

Engine* comparison_experiment_closed(clockwork::Client* client,
								int num_models=15,
								int concurrency=16) {
	Engine* engine = new Engine();

	std::string modelpath = util::get_clockwork_modelzoo()["resnet50_v2"];
	auto models = client->load_remote_models(modelpath, num_models);

	for (int i = 0; i < num_models; i++) {
		engine->AddWorkload(new ClosedLoop(
			i, 			// client id, just use the same for this
			models[i],	// model
			concurrency			// concurrency
		));
	}

	return engine;
}

Engine* fill_memory(clockwork::Client* client) {
	Engine* engine = new Engine();

	unsigned num_copies = 411;
	std::string modelpath = util::get_clockwork_modelzoo()["resnet50_v2"];
	auto models = client->load_remote_models(modelpath, num_copies);

	unsigned i = 0;
	for (; i < 1; i++) {
		engine->AddWorkload(new PoissonOpenLoop(
			i,				// client id
			models[i],  	// model
			i,      		// rng seed
			1000				// requests/second
		));
	}
	for (; i < 11; i++) {
		engine->AddWorkload(new PoissonOpenLoop(
			i,				// client id
			models[i],  	// model
			i,      		// rng seed
			10				// requests/second
		));
	}
	for (; i < 411; i++) {
		engine->AddWorkload(new PoissonOpenLoop(
			i,				// client id
			models[i],  	// model
			i,      		// rng seed
			1				// requests/second
		));
	}

	return engine;
}

Engine* simple(clockwork::Client* client) {
	Engine* engine = new Engine();

	unsigned num_copies = 2;
	std::string modelpath = util::get_clockwork_modelzoo()["resnet50_v2"];
	auto models = client->load_remote_models(modelpath, num_copies);

	for (unsigned i = 0; i < models.size(); i++) {
		engine->AddWorkload(new ClosedLoop(
			0, 			// client id, just use the same for this
			models[i],	// model
			1			// concurrency
		));
	}

	return engine;
}

Engine* simple_slo_factor(clockwork::Client* client) {
	Engine* engine = new Engine();

	unsigned num_copies = 3;
	std::string modelpath = util::get_clockwork_modelzoo()["resnet50_v2"];
	auto models = client->load_remote_models(modelpath, num_copies);

	for (unsigned i = 0; i < models.size(); i++) {
		engine->AddWorkload(new ClosedLoop(
			0, 			// client id, just use the same for this
			models[i],	// model
			1			// concurrency
		));
	}

	// Adjust model 0 multiplicatively
	engine->AddWorkload(new AdjustSLO(
		10.0, 					// period in seconds
		4.0,					// initial slo_factor
		{models[0]},			// The model or models to apply the SLO adjustment to
		[](float current) { 	// SLO update function
			return current * 1.25; 
		},
		[](float current) { return false; } // Terminate condition
	));

	// Adjust model 1 additively
	engine->AddWorkload(new AdjustSLO(
		10.0, 					// period in seconds
		4.0,					// initial slo_factor
		{models[1]},			// The model or models to apply the SLO adjustment to
		[](float current) { 	// SLO update function
			return current + 1.0; 
		},
		[](float current) { return false; } // Terminate condition
	));

	// Adjust model 2 back and forth
	engine->AddWorkload(new AdjustSLO(
		10.0, 					// period in seconds
		10,						// initial slo_factor
		{models[2]},			// The model or models to apply the SLO adjustment to
		[](float current) { 	// SLO update function
			return current = 10 ? 1 : 10;
		},
		[](float current) { return false; } // Terminate condition
	));

	return engine;	
}

Engine* simple_parametric(clockwork::Client* client, unsigned num_copies,
	unsigned num_clients, unsigned concurrency, unsigned num_requests) {
	Engine* engine = new Engine();

	unsigned num_models_per_client = ceil(num_copies / (float)num_clients);
	std::string modelpath = util::get_clockwork_modelzoo()["resnet50_v2"];
	auto models = client->load_remote_models(modelpath, num_copies);

	for (unsigned i = 0; i < num_clients; i++) {

		// note that std::vector initialization ignores the last element
		auto it1 = models.begin() + (i * num_models_per_client);
		auto it2 = models.begin() + ((i + 1) * num_models_per_client);
		std::vector<clockwork::Model*> models_subset(it1, std::min(it2, models.end()));

		std::cout << "Adding ClosedLoop client " << i << " with models:";
		for (auto const &model : models_subset) {
			
			std::cout << " " << model->id();
		}
		std::cout << std::endl;

		engine->AddWorkload(new ClosedLoop(
			i, 				// client id
			models_subset,	// subset of models
			concurrency,	// concurrency
			num_requests,	// max num requests
			0				// jitter
		));

		if (it2 >= models.end()) { break; }
	}

	return engine;
}

Engine* poisson_open_loop(clockwork::Client* client, unsigned num_models,
	double rate) {
	Engine* engine = new Engine();

	std::string model_name = "resnet50_v2";
	std::string modelpath = util::get_clockwork_modelzoo()[model_name];

	std::cout << "Loading " << num_models << " " << model_name
			  << " models" << std::endl;
	std::cout << "Cumulative request rate across models: " << rate
			  << " requests/seconds" << std::endl;
	auto models = client->load_remote_models(modelpath, num_models);

	std::cout << "Adding a PoissonOpenLoop Workload (" << (rate/num_models)
			  << " requests/second) for each model" << std::endl;
	for (unsigned i = 0; i < models.size(); i++) {
		engine->AddWorkload(new PoissonOpenLoop(
			i,				// client id
			models[i],  	// model
			i,      		// rng seed
			rate/num_models	// requests/second
		));
	}

	return engine;
}

Engine* example(clockwork::Client* client) {
	Engine* engine = new Engine();

	unsigned client_id = 0;
	for (auto &p : util::get_clockwork_modelzoo()) {
		std::cout << "Loading " << p.first << std::endl;
		auto model = client->load_remote_model(p.second);

		Workload* fixed_rate = new FixedRate(
			client_id,		// client id
			model,  		// model
			0,      		// rng seed
			5				// requests/second
		);
		Workload* open = new PoissonOpenLoop(
			client_id,		// client id
			model,			// model
			1,				// rng seed
			10				// request/second
		);
		Workload* burstyopen = new BurstyPoissonOpenLoop(
			client_id, 		// client id
			model, 			// model
			1,				// rng seed
			10,				// requests/second
			10,				// burst duration
			20				// idle duration
		);
		Workload* closed = new ClosedLoop(
			client_id, 		// client id
			model,			// model
			1				// concurrency
		);
		Workload* burstyclosed = new BurstyPoissonClosedLoop(
			client_id, 		// client id
			model, 			// model
			1,				// concurrency
			0,				// rng seed
			10, 			// burst duration
			20				// idle duration
		);

		engine->AddWorkload(fixed_rate);
		engine->AddWorkload(open);
		engine->AddWorkload(burstyopen);
		engine->AddWorkload(closed);
		engine->AddWorkload(burstyclosed);

		client_id++;
	}

	return engine;
}

Engine* spam(clockwork::Client* client, std::string model_name = "resnet50_v2") {
	Engine* engine = new Engine();

	unsigned num_copies = 100;
	std::string modelpath = util::get_clockwork_modelzoo()[model_name];
	auto models = client->load_remote_models(modelpath, num_copies);

	for (unsigned i = 0; i < models.size(); i++) {
		engine->AddWorkload(new ClosedLoop(
			0, 			// client id, just use the same for this
			models[i],	// model
			100			// concurrency
		));
	}

	return engine;
}

Engine* spam2(clockwork::Client* client) {
	Engine* engine = new Engine();

	unsigned num_copies = 100;
	std::string modelpath = util::get_clockwork_modelzoo()["resnet50_v2"];
	auto models = client->load_remote_models(modelpath, num_copies);

	for (unsigned i = 0; i < models.size(); i++) {
		engine->AddWorkload(new PoissonOpenLoop(
			i,				// client id
			models[i],  	// model
			i,      		// rng seed
			1000			// requests/second
		));
	}

	return engine;
}

Engine* single_spam(clockwork::Client* client) {
	Engine* engine = new Engine();

	unsigned num_copies = 100;
	std::string modelpath = util::get_clockwork_modelzoo()["resnet50_v2"];
	auto models = client->load_remote_models(modelpath, num_copies);

	for (unsigned i = 0; i < models.size(); i++) {
		engine->AddWorkload(new ClosedLoop(
			0, 			// client id, just use the same for this
			models[i],	// model
			1000			// concurrency
		));
	}

	return engine;
}

Engine* azure(clockwork::Client* client) {
	Engine* engine = new Engine();

	auto trace_data = azure::load_trace();

	unsigned trace_id = 1;
	std::string model = util::get_clockwork_model("resnet50_v2");
	unsigned num_copies = 200;

	auto models = client->load_remote_models(model, num_copies);

	for (unsigned i = 0; i < trace_data.size(); i++) {
		auto model = models[i % num_copies];
		auto workload = trace_data[i];

		Workload* replay = new PoissonTraceReplay(
			0,				// client id, just give them all the same ID for this example
			model,			// model
			i,				// rng seed
			workload,		// trace data
			1.0,			// scale factor; default 1
			60.0,			// interval duration; default 60
			0				// interval to begin at; default 0; set to -1 for random
		);

		engine->AddWorkload(replay);
	}

	return engine;
}

Engine* azure_single(clockwork::Client* client) {
	Engine* engine = new Engine();

	auto trace_data = azure::load_trace();

	unsigned trace_id = 1;
	std::string model = util::get_clockwork_model("resnet50_v2");
	unsigned num_copies = 1;

	auto models = client->load_remote_models(model, num_copies);

	for (unsigned i = 0; i < num_copies; i++) {
		auto model = models[i % num_copies];
		auto workload = trace_data[i];

		Workload* replay = new PoissonTraceReplay(
			0,				// client id, just give them all the same ID for this example
			model,			// model
			i,				// rng seed
			workload,		// trace data
			1.0,			// scale factor; default 1
			1.0,			// interval duration; default 60
			0				// interval to begin at; default 0; set to -1 for random
		);

		engine->AddWorkload(replay);
	}

	return engine;
}

Engine* azure_small(clockwork::Client* client) {
	Engine* engine = new Engine();

	auto trace_data = azure::load_trace(1);

	std::vector<Model*> models;
	for (auto &p : util::get_clockwork_modelzoo()) {
		std::cout << "Loading " << p.first << std::endl;
		for (auto &model : client->load_remote_models(p.second, 3)) {
			models.push_back(model);
		}
	}

	for (unsigned i = 0; i < trace_data.size(); i++) {
		auto model = models[i % models.size()];
		
		auto workload = trace_data[i];

		Workload* replay = new PoissonTraceReplay(
			i,				// client id, just give them all the same ID for this example
			model,			// model
			i,				// rng seed
			workload,		// trace data
			1.0,			// scale factor; default 1
			60.0,			// interval duration; default 60
			0				// interval to begin at; default 0; set to -1 for random
		);

		engine->AddWorkload(replay);
	}

	return engine;
}

Engine* azure_fast(clockwork::Client* client, unsigned trace_id = 1) {
	Engine* engine = new Engine();

	auto trace_data = azure::load_trace(trace_id);

	std::vector<Model*> models;
	for (auto &p : util::get_clockwork_modelzoo()) {
		std::cout << "Loading " << p.first << std::endl;
		for (auto &model : client->load_remote_models(p.second, 3)) {
			models.push_back(model);
		}
	}


	for (unsigned i = 0; i < trace_data.size(); i++) {
		auto model = models[i % models.size()];
		
		auto workload = trace_data[i];

		Workload* replay = new PoissonTraceReplay(
			i,				// client id, just give them all the same ID for this example
			model,			// model
			i,				// rng seed
			workload,		// trace data
			1.0,			// scale factor; default 1
			1.0,			// interval duration; default 60
			-1				// interval to begin at; default 0; set to -1 for random
		);

		engine->AddWorkload(replay);
	}

	return engine;
}

Engine* bursty_experiment2(
		clockwork::Client* client,
		unsigned start_at = 1,
		unsigned increment = 1,
		unsigned stop_at = 3600,
		unsigned interval_duration_seconds = 1, // seconds
		unsigned lead_in = 300, // lead-in intervals
		unsigned total_request_rate = 1000
	) {
	Engine* engine = new Engine();

	std::vector<std::vector<unsigned>> interval_rates(stop_at);

	unsigned num_models = stop_at;

	for (unsigned i = 0; i < num_models; i++) {
		interval_rates[i].resize(lead_in + (stop_at - start_at) / increment, 0);
	}
	unsigned interval = lead_in;
	while (start_at <= stop_at) {
		for (int j = 0; j < 60 * total_request_rate; j++) {
			unsigned model = j % start_at;
			interval_rates[model][interval]++;
		}
		start_at += increment;
		interval++;
	}

	for (unsigned i = 0; i < 5; i++) {
		std::cout << "Model " << i;
		for (auto rate : interval_rates[i]) {
			std::cout << " " << rate;
		}
		std::cout << std::endl;
	}
	std::cout << "..." << std::endl;
	for (unsigned i = num_models - 6; i < num_models; i++) {
		std::cout << "Model " << i;
		for (auto rate : interval_rates[i]) {
			std::cout << " " << rate;
		}
		std::cout << std::endl;
	}

	std::string modelpath = util::get_clockwork_modelzoo()["resnet50_v2"];

	std::vector<Model*> models;
	while (models.size() < num_models) {
		unsigned to_load = std::min((int) (num_models - models.size()), 100);
		std::cout << "Loading " << models.size() << ":" << models.size()+to_load << std::endl;
		for (auto &model : client->load_remote_models(modelpath, to_load)) {
			models.push_back(model);
		}
	}

	Model* hipri = client->load_remote_model(modelpath);

	for (unsigned i = 0; i < models.size(); i++) {
		auto model = models[i];
		
		auto workload = interval_rates[i];

		Workload* replay = new PoissonTraceReplay(
			i,	// client id
			model,	// model
			i, 		// rng seed
			workload, 	// synthetic trace data
			1, 		// scale factor
			interval_duration_seconds,		// interval duration
			0		// start interval
		);

		engine->AddWorkload(replay);
	}

	engine->AddWorkload(new PoissonOpenLoop(
		models.size(),		// client id
		hipri,  	// model
		models.size(),      		// rng seed
		200			// requests/second
	));

	return engine;
}



Engine* bursty_experiment(
		clockwork::Client* client, 
		unsigned trace_id = 1,
		unsigned num_models = 400,
		unsigned interval_duration_seconds = 10,
		unsigned start_at = 60,
		unsigned increment = 60,
		unsigned end_at = 1200
	) {
	Engine* engine = new Engine();

	std::string modelpath = util::get_clockwork_modelzoo()["resnet50_v2"];

	std::vector<std::vector<unsigned>> interval_rates(num_models);
	for (unsigned i = start_at; i <= end_at; i += increment) {
		int total_requests = i * 60;
		for (int models_remaining = num_models; models_remaining > 0; models_remaining--) {
			int requests = total_requests / models_remaining;
			interval_rates[models_remaining-1].push_back(requests);
			total_requests -= requests;
		}
	}
	int intervals = interval_rates[0].size();
	std::cout << intervals << " experiment intervals" << std::endl;

	for (unsigned i = 0; i < 5; i++) {
		std::cout << "Model " << i << ": ";
		for (unsigned j = 0; j < intervals; j++) {
			std::cout << " " << interval_rates[i][j];
		}
		std::cout << std::endl;
	}
	std::cout << "..." << std::endl;
	for (unsigned i = num_models - 6; i < num_models; i++) {
		std::cout << "Model " << i << ": ";
		for (unsigned j = 0; j < intervals; j++) {
			std::cout << " " << interval_rates[i][j];
		}
		std::cout << std::endl;
	}

	std::vector<Model*> models;
	while (models.size() < num_models) {
		unsigned to_load = std::min((int) (num_models - models.size()), 100);
		std::cout << "Loading " << models.size() << ":" << models.size()+to_load << std::endl;
		for (auto &model : client->load_remote_models(modelpath, to_load)) {
			models.push_back(model);
		}
	}

	for (unsigned i = 0; i < models.size(); i++) {
		auto model = models[i];
		
		auto workload = interval_rates[i];

		Workload* replay = new PoissonTraceReplay(
			i,	// client id
			model,	// model
			i, 		// rng seed
			workload, 	// synthetic trace data
			1, 		// scale factor
			interval_duration_seconds,		// interval duration
			0		// start interval
		);

		engine->AddWorkload(replay);
	}

	return engine;
}


	// PoissonTraceReplay(int id, clockwork::Model* model, int rng_seed,
	// 	std::vector<unsigned> &interval_rates, // request rate for each interval, specified as requests per minute
	// 	double scale_factor = 1.0,
	// 	double interval_duration_seconds = 60.0,
	// 	int start_at = 0 // which interval to start at.  if set to -1, will randomise
	// )

Engine* bursty_experiment1(clockwork::Client* client, unsigned trace_id = 1) {
	Engine* engine = new Engine();

	unsigned num_copies = 3600;
	std::string modelpath = util::get_clockwork_modelzoo()["resnet50_v2"];

	std::vector<Model*> models;
	for (int i = 0; i < num_copies; i+=100) {
		std::cout << "Loading " << i << ":" << i+100 << std::endl;
		for (auto &model : client->load_remote_models(modelpath, 100)) {
			models.push_back(model);
		}
	}

	unsigned num_iterations = 10;
	uint64_t iteration_length = 60000000000UL; // 60 seconds

	unsigned increase_per_iteration = ((num_copies-1) / num_iterations) + 1;
	unsigned model_id = 0;
	for (unsigned i = 0; i < num_iterations; i++) {
		unsigned iteration_max = increase_per_iteration * (i+1);
		while (model_id < iteration_max && model_id < models.size()) {
			engine->AddWorkload(new PoissonOpenLoop(
				model_id,				// client id
				models[model_id],  	// model
				model_id,      		// rng seed
				1			// requests/second
			), i * iteration_length);
			model_id++;
		}
	}

	return engine;
}

Engine* bursty_experiment_simple(clockwork::Client* client, unsigned trace_id = 1) {
	Engine* engine = new Engine();

	unsigned num_copies = 3600;
	std::string modelpath = util::get_clockwork_modelzoo()["resnet50_v2"];

	std::vector<Model*> models;
	for (int i = 0; i < num_copies; i+=100) {
		std::cout << "Loading " << i << ":" << i+100 << std::endl;
		for (auto &model : client->load_remote_models(modelpath, 100)) {
			models.push_back(model);
		}
	}

	for (unsigned i = 0; i < models.size(); i++) {
		engine->AddWorkload(new PoissonOpenLoop(
			i,				// client id
			models[i],  	// model
			i,      		// rng seed
			0.1			// requests/second
		));
	}

	return engine;
}

Engine* azure_parameterized(clockwork::Client* client,
		unsigned trace_id = 1,
		double scale_factor = 1.0,
		unsigned interval_duration_seconds = 60,
		bool randomise = false,
		bool use_all_models = true,
		unsigned num_models = 3100,
		bool stripe = false
	) {
	Engine* engine = new Engine();

	std::cout << "Running azure_parameterized"
			  << " trace_id=" << trace_id
			  << " scale_factor=" << scale_factor
			  << " interval_duration_seconds=" << interval_duration_seconds
			  << " randomise=" << randomise
			  << " use_all_models=" << use_all_models
			  << " num_models=" << num_models
			  << std::endl;

	auto trace_data = azure::load_trace(trace_id);

	if (num_models > 3100) {
		std::cout << "WARNING: 3100 is about the GPU memory limit for models. "
				  << "Got " << num_models
				  << ".  You might get cuda device memory exhaustion errors."
				  << std::endl;
	}

	srand(0);
	std::vector<Model*> models;
	auto modelzoo = util::get_clockwork_modelzoo();	
	if (use_all_models) {
		unsigned models_remaining = modelzoo.size();
		for (auto &p : modelzoo) {
			unsigned num_copies = num_models / models_remaining;
			num_models -= num_copies;
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
		std::cout << "Loading " << model << " x" << num_models << std::endl;
		for (auto &model : client->load_remote_models(modelzoo[model], num_models)) {
			models.push_back(model);
		}
	}

	for (unsigned i = 0; i < trace_data.size(); i++) {
		Model* model;
		if (stripe) {
			model = models[i % models.size()];
		} else {
			model = models[(models.size() * i) / trace_data.size()];
		}
		
		auto workload = trace_data[i];

		Workload* replay = new PoissonTraceReplay(
			i,				// client id, just give them all the same ID for this example
			model,			// model
			i,				// rng seed
			workload,		// trace data
			scale_factor,			// scale factor; default 1
			interval_duration_seconds,			// interval duration; default 60
			randomise ? -1 : 0				// interval to begin at; default 0; set to -1 for random
		);

		engine->AddWorkload(replay);
	}

	return engine;
}

Engine* azure2(clockwork::Client* client,
		unsigned num_workers = 1,
		bool use_all_models = true,
		double load_factor = 1.0,
		unsigned memory_load_factor = 4, // 1, 2, 3, or 4
		unsigned interval_duration_seconds = 60,
		unsigned trace_id = 1,
		bool randomise_start = false
	) {
	std::cout << "Running azure2 workload with"
			  << " num_workers=" << num_workers
			  << " use_all_models=" << use_all_models
			  << " load_factor=" << load_factor
			  << " memory_load_factor=" << memory_load_factor
			  << " interval_duration_seconds=" << interval_duration_seconds
			  << " trace_id=" << trace_id
			  << " randomise_start=" << randomise_start
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

	std::cout << "Using " << total_models << " models (" << copies_per_model << " copies of " << num_models << " models)" << std::endl;

	double scale_factor = load_factor * estimated_load_factor_per_worker * num_workers;
	std::cout << "Replaying trace " << trace_id << " at scale_factor=" << scale_factor << " to achieve load_factor " << load_factor << " on " << num_workers << " workers" << std::endl;

	std::cout << interval_duration_seconds << " second intervals" << std::endl;

	return azure_parameterized(client,
			trace_id,
			scale_factor,
			interval_duration_seconds,
			randomise_start,
			use_all_models,
			total_models
		);
}

}
}

#endif 

