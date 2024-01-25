#ifndef _CLOCKWORK_WORKLOAD_SLO_H_
#define _CLOCKWORK_WORKLOAD_SLO_H_

#include "clockwork/util.h"
#include "clockwork/workload/workload.h"
#include "clockwork/workload/azure.h"
#include <cstdlib>
#include <iostream>

namespace clockwork {
namespace workload {

Engine* slo_experiment_1(
	clockwork::Client* client, std::string model_name, unsigned num_copies,
	std::string distribution, double rate, // OpenLoop client params
	unsigned slo_start, unsigned slo_end, double slo_factor, std::string slo_op, unsigned period) { // SLO adjustment params

	Engine* engine = new Engine();

	std::string modelpath;
	try {
		modelpath = util::get_clockwork_modelzoo()[model_name];
	} catch (const std::exception& e) {
		CHECK(false) << "Modelpath not found: " << e.what();
	}

	auto models = client->load_remote_models(modelpath, num_copies);

	for (unsigned i = 0; i < models.size(); i++) {
		if (distribution == "poisson") {
			engine->AddWorkload(new PoissonOpenLoop(
				i,				// client id, just use the same for this
				models[i],		// model
				i,				// rng seed
				rate/num_copies // requests/second
			));
		} else if (distribution == "fixed-rate") {
			engine->AddWorkload(new FixedRate(
				i,				// client id, just use the same for this
				models[i],		// model
				i,				// rng seed
				rate/num_copies // requests/second
			));
		} else {
			CHECK(false) << distribution << " distribution not yet implemented";
		}
	}

	if (slo_op == "add") {
		// adjust all models additively
		engine->AddWorkload(new AdjustSLO(
			period, 	// period in seconds
			slo_start,	// initial slo
			models,		// apply to all models, for now
			[slo_factor](float current) { return current + slo_factor; },	// slo update function
			[slo_end](float current) { return current > slo_end; } 	// slo termination condition
		));
	} else if (slo_op == "mul") {
		engine->AddWorkload(new AdjustSLO(
			period, 	// period in seconds
			slo_start,	// initial slo
			models,		// apply to all models, for now
			[slo_factor](float current) { return current * slo_factor; },	// slo update function
			[slo_end](float current) { return current > slo_end; } 	// slo termination condition
		));
	} else {
		CHECK(false) << slo_op << " operator not yet implemented";
	}

	return engine;	
}

Engine* slo_experiment_2(
	clockwork::Client* client, std::string model_name, // global params
	unsigned num_copies_fg, std::string distribution_fg, double rate_fg, // FG params
	unsigned slo_start_fg, unsigned slo_end_fg, double slo_factor_fg, std::string slo_op_fg, unsigned period_fg, // FG SLO params
	unsigned num_copies_bg, unsigned concurrency_bg, double slo_bg) { // BG params

	Engine* engine = new Engine();

	std::string modelpath;
	std::vector<clockwork::Model*> models_fg, models_bg;
	double rate_fg_avg, min_bg, max_bg, step_bg;

	try {
		modelpath = util::get_clockwork_modelzoo()[model_name];
	} catch (const std::exception& e) {
		CHECK(false) << "Modelpath not found: " << e.what();
	}

	if (num_copies_fg != 0) {
		models_fg = client->load_remote_models(modelpath, num_copies_fg);
		rate_fg_avg = rate_fg / num_copies_fg;
		CHECK(rate_fg > 0) << "Rate cannot be zero";
		CHECK(distribution_fg == "poisson" or distribution_fg == "fixed-rate") << "Distribution " << distribution_fg << " not yet implemented";
		CHECK(slo_op_fg == "add" or slo_op_fg == "mul") << "Operator " << slo_op_fg << " not yet implemented";

		for (unsigned i = 0; i < models_fg.size(); i++) {
			if (distribution_fg == "poisson") {
				engine->AddWorkload(new PoissonOpenLoop(
					i,				// client id, just use the same for this
					models_fg[i],	// model
					i,				// rng seed
					rate_fg_avg		// requests/second
				));
			} else if (distribution_fg == "fixed-rate") {
				engine->AddWorkload(new FixedRate(
					i,				// client id, just use the same for this
					models_fg[i],	// model
					i,				// rng seed
					rate_fg_avg		// requests/second
				));
			}
		}

		if (slo_op_fg == "add") {
			engine->AddWorkload(new AdjustSLO(
				period_fg, 		// period in seconds
				slo_start_fg,	// initial slo
				models_fg,		// apply to all models, for now
				[slo_factor_fg](float current) { return current + slo_factor_fg; },
				[slo_end_fg](float current) { return current > slo_end_fg; }
			));
		} else if (slo_op_fg == "mul") {
			engine->AddWorkload(new AdjustSLO(
				period_fg, 		// period in seconds
				slo_start_fg,	// initial slo
				models_fg,		// apply to all models, for now
				[slo_factor_fg](float current) { return current * slo_factor_fg; },
				[slo_end_fg](float current) { return current > slo_end_fg; }
			));
		}
	}

	if (num_copies_bg != 0) {
		models_bg = client->load_remote_models(modelpath, num_copies_bg);
		for (unsigned i = 0; i < models_bg.size(); i++) {
			models_bg[i]->set_slo_factor(slo_bg);
			engine->AddWorkload(new ClosedLoop(
				i + models_fg.size(),	// client id
				models_bg[i],			// model
				concurrency_bg,			// concurrency
				UINT64_MAX,				// number of requests
				10						// Initial jitter in seconds
			));
		}
	}

	return engine;	
}

//Engine* slo_experiment_2(
//	clockwork::Client* client,
//	std::string model_name,
//	std::string distribution,
//	unsigned num_copies_fg,
//	double rate_fg,
//	double slo_fg,
//	unsigned num_copies_bg,
//	double rate_bg_min,
//	double rate_bg_max,
//	double rate_bg_step_size,
//	unsigned period,
//	double slo_bg) {
//
//	Engine* engine = new Engine();
//
//	std::string modelpath;
//	try {
//		modelpath = util::get_clockwork_modelzoo()[model_name];
//	} catch (const std::exception& e) {
//		CHECK(false) << "Modelpath not found: " << e.what();
//	}
//
//	std::vector<clockwork::Model*> models_fg, models_bg;
//	if (num_copies_fg != 0) { models_fg = client->load_remote_models(modelpath, num_copies_fg); }
//	if (num_copies_bg != 0) { models_bg = client->load_remote_models(modelpath, num_copies_bg); }
//
//	double rate_fg_avg, min_bg, max_bg, step_bg;
//
//	if (num_copies_fg != 0) {
//		rate_fg_avg = rate_fg / num_copies_fg;
//		CHECK(rate_fg > 0) << "Rate cannot be zero";
//	}
//
//	if (num_copies_bg != 0) {
//		min_bg = rate_bg_min / num_copies_bg;
//		max_bg = rate_bg_max / num_copies_bg;
//		//step_bg = rate_bg_step_size / num_copies_bg;
//		step_bg = rate_bg_step_size;  // as multiplicative increase
//		CHECK(rate_bg_min > 0) << "Rate cannot be zero";
//		CHECK(rate_bg_min > 1e-2) << "Rate cannot be very small "
//			<< "(since rate update only takes effect after timeout callback)";
//		CHECK(10 * (1 / min_bg) < period) << "Initial rate updates may not be observed";
//	}
//
//	if (distribution == "poisson") {
//		for (unsigned i = 0; i < models_fg.size(); i++) {
//			models_fg[i]->set_slo_factor(slo_fg);
//			engine->AddWorkload(new PoissonOpenLoop(
//				i,				// client id, just use the same for this
//				models_fg[i],	// model
//				i,				// rng seed
//				rate_fg_avg 			// requests/second
//			));
//		}
//
//		std::vector<OpenLoop<Exponential>*> bg_workloads;
//		for (unsigned i = 0; i < models_bg.size(); i++) {
//			models_bg[i]->set_slo_factor(slo_bg);
//			OpenLoop<Exponential>* workload = new PoissonOpenLoop(
//				models_bg.size() + i,	// client id, just use the same for this
//				models_bg[i],			// model
//				models_bg.size() + i,	// rng seed
//				min_bg 					// requests/second
//			);
//			engine->AddWorkload(workload);
//			bg_workloads.push_back(workload);
//		}
//
//		if (num_copies_bg != 0) {
//			// adjust all bg model rates additively
//			engine->AddWorkload(new AdjustPoissonRate(
//				period, 		// period in seconds
//				min_bg,			// initial rate
//				bg_workloads,	// apply to all bg models
//				[step_bg] (double current) { return current * step_bg; },
//				[max_bg](double current) { return current > max_bg; }
//			));
//		}
//	} else if (distribution == "fixed-rate") {
//		for (unsigned i = 0; i < models_fg.size(); i++) {
//			models_fg[i]->set_slo_factor(slo_fg);
//			std::cout << "Request rate for FG Model " << models_fg[i]->id()
//					  << ": " << rate_fg_avg << "rps" << std::endl;
//			engine->AddWorkload(new FixedRate(
//				i,				// client id, just use the same for this
//				models_fg[i],	// model
//				i,				// rng seed
//				rate_fg_avg			// requests/second
//			));
//		}
//
//		std::vector<OpenLoop<Static>*> bg_workloads;
//		for (unsigned i = 0; i < models_bg.size(); i++) {
//			std::cout << "Request rate for BG Model " << models_bg[i]->id()
//					  << ": " << min_bg << "rps" << std::endl;
//			models_bg[i]->set_slo_factor(slo_bg);
//			OpenLoop<Static>* workload = new FixedRate(
//				models_bg.size() + i,	// client id, just use the same for this
//				models_bg[i],			// model
//				models_bg.size() + i,	// rng seed
//				min_bg					// requests/second
//			);
//			engine->AddWorkload(workload);
//			bg_workloads.push_back(workload);
//		}
//
//		if (num_copies_bg != 0) {
//			// adjust all bg model rates additively
//			engine->AddWorkload(new AdjustFixedRate(
//				period, 		// period in seconds
//				min_bg,			// initial rate
//				bg_workloads,	// apply to all bg models
//				[step_bg] (double current) { return current * step_bg; },
//				[max_bg](double current) { return current > max_bg; }
//			));
//		}
//	} else {
//		CHECK(false) << distribution << " distribution not yet implemented";
//	}
//
//	return engine;
//}

}
}

#endif 
