#include "clockwork/util.h"
#include "clockwork/client.h"
#include "clockwork/workload/workload.h"
#include "clockwork/workload/example.h"
#include <string>
#include <iostream>
#include "clockwork/workload/azure.h"
#include "clockwork/workload/slo.h"
#include "clockwork/workload/scalability.h"
#include "clockwork/thread.h"

using namespace clockwork;

std::pair<std::string, std::string> split(std::string addr) {
	auto split = addr.find(":");
	std::string hostname = addr.substr(0, split);
	std::string port = addr.substr(split+1, addr.size());
	return {hostname, port};
}

void printUsage() {
	std::cerr << "Usage: client [address] [workload] "
			  << "[workload parameters (if required)]" << std::endl
			  << "Available workloads with parameters:" << std::endl

			  << "\t example" << std::endl

			  << "\t fill_memory" << std::endl
			  << "\t\t creates 500 copies of resnet50, more than can fit in memory" << std::endl
			  << "\t\t 100 of them are closed loop, 400 are gentle open loop" << std::endl

			  << "\t spam [modelname]" << std::endl
			  << "\t\t default modelname is resnet50_v2" << std::endl
			  << "\t\t 100 instances, each with 100 closed loop" << std::endl

			  << "\t single-spam" << std::endl
			  << "\t\t resnet50_v2 x 1, with 1000 closed loop" << std::endl

			  << "\t simple" << std::endl

			  << "\t simple-slo-factor" << std::endl
			  << "\t\t 3 models with closed-loop concurrency of 1" << std::endl
			  << "\t\t Updates each model's slo factor every 10 seconds" << std::endl

			  << "\t simple-parametric models clients concurrency requests" << std::endl
			  << "\t\t Workload parameters:" << std::endl
			  << "\t\t\t models: number of model copies" << std::endl
			  << "\t\t\t clients: number of clients among which the models are partitioned" << std::endl
			  << "\t\t\t concurrency: number of concurrent requests per client" << std::endl
			  << "\t\t\t requests: total number of requests per client (for termination)" << std::endl

			  << "\t poisson-open-loop num_models rate " << std::endl
			  << "\t\t Rate should be provided in requests/second" << std::endl
			  << "\t\t Rate is split across all models" << std::endl

			  << "\t scalability-exp-1 num-models rate-min rate-max rate-factor rate-op period" << std::endl
			  << "\t\t Workload parameters:" << std::endl
			  << "\t\t\t num-models: number of \"resnet50_v2\" models" << std::endl
			  << "\t\t\t rate-min: minimum total request rate" << std::endl
			  << "\t\t\t rate-max: maximum total request rate" << std::endl
			  << "\t\t\t rate-factor: rate increment factor" << std::endl
			  << "\t\t\t rate-op: operator (\"add\"/\"mul\") for incrementing rate" << std::endl
			  << "\t\t\t period: number of seconds before changing rate" << std::endl

			  << "\t slo-exp-1 model copies dist rate slo-start slo-end slo-factor slo-op period" << std::endl
			  << "\t\t Workload parameters:" << std::endl
			  << "\t\t\t model: model name (e.g., \"resnet50_v2\")" << std::endl
			  << "\t\t\t copies: number of model instances" << std::endl
			  << "\t\t\t dist: arrival distribution (\"poisson\"/\"fixed-rate\")" << std::endl
			  << "\t\t\t rate: arrival rate (in requests/second)" << std::endl
			  << "\t\t\t slo-start: starting slo multiplier" << std::endl
			  << "\t\t\t slo-end: ending slo multiplier" << std::endl
			  << "\t\t\t slo-factor: factor by which the slo multiplier should change" << std::endl
			  << "\t\t\t slo-op: operator (\"add\"/\"mul\") for incrementing slo" << std::endl
			  << "\t\t\t period: number of seconds before changing slo" << std::endl
			  << "\t\t Examples:" << std::endl
			  << "\t\t\t client volta04:12346 slo-exp-1 resnet50_v2 4 poisson 100 2 32 2 mul 7" << std::endl
			  << "\t\t\t\t (increases slo every 7s as follows: 2 4 8 16 32)" << std::endl
			  << "\t\t\t client volta04:12346 slo-exp-1 resnet50_v2 4 poisson 100 10 100 10 add 3" << std::endl
			  << "\t\t\t\t (increases slo every 3s as follows: 10 20 30 ... 100)" << std::endl
			  << "\t\t In each case, an open loop client is used" << std::endl

			  << "\t slo-exp-2 model copies-fg dist-fg rate-fg slo-start-fg slo-end-fg slo-factor-fg slo-op-fg period-fg copies-bg concurrency-bg slo-bg" << std::endl
			  << "\t\t Description: Running latency-sensitive (foreground or FG) and batch (background or BG) workloads simultaneously" << std::endl
			  << "\t\t Workload parameters:" << std::endl
			  << "\t\t\t model: model name (e.g., \"resnet50_v2\")" << std::endl
			  << "\t\t\t copies-fg: number of FG models" << std::endl
			  << "\t\t\t dist-fg: arrival distribution (\"poisson\"/\"fixed-rate\") for open loop clients for FG models" << std::endl
			  << "\t\t\t rate-fg: total arrival rate (in requests/second) for FG models" << std::endl
			  << "\t\t\t slo-start-fg: starting slo multiplier for FG models" << std::endl
			  << "\t\t\t slo-end-fg: ending slo multiplier for FG models" << std::endl
			  << "\t\t\t slo-factor-fg: factor by which the slo multiplier should change for FG models" << std::endl
			  << "\t\t\t slo-op-fg: operator (\"add\"/\"mul\") for applying param slo-factor-fg" << std::endl
			  << "\t\t\t period-fg: number of seconds before changing FG models' slo" << std::endl
			  << "\t\t\t copies-bg: number of BG models (for which requests arrive in closed loop)" << std::endl
			  << "\t\t\t concurrency-bg: number of concurrent requests for BG model' closed loop clients" << std::endl
			  << "\t\t\t slo-bg: slo multiplier for BG moels (ideally, should be a relaxed slo)" << std::endl
			  << "\t\t Examples:" << std::endl
			  << "\t\t\t client volta04:12346 slo-exp-2 resnet50_v2    2 poisson 200  2 32 2 mul 7    4 1 100" << std::endl
			  << "\t\t\t\t (2 FG models with PoissonOpenLoop clients sending requests at 200 rps)" << std::endl
			  << "\t\t\t\t (the SLO factor of each FG model is updated every 7 seconds as follows: 2 4 8 16 32)" << std::endl
			  << "\t\t\t\t (4 BG models with a relaxed SLO factor of 100 and respective ClosedLoop clients configured with a concurrency factor of 1)" << std::endl

			  << "\t comparison_experiment" << std::endl
			  << "\t\t Description: runs multiple copies of resnet50_v2" << std::endl
			  << "\t\t Workload parameters:" << std::endl
			  << "\t\t\t num_models: (int, default 15) the number of models you're using" << std::endl
			  << "\t\t\t total_requests: (int, default 1000) the total requests across all models, per second" << std::endl

			  << "\t comparison_experiment2" << std::endl
			  << "\t\t Description: closed-loop version of comparison experiment" << std::endl
			  << "\t\t Workload parameters:" << std::endl
			  << "\t\t\t num_models: (int, default 15) the number of models you're using" << std::endl
			  << "\t\t\t concurrency: (int, default 16) closed loop workload concurrency" << std::endl

			  << "\t azure" << std::endl
			  << "\t\t Description: replay an azure workload trace.  Can be run with no arguments, in which case default values are used.  The defaults will load 3100 models and replay a trace that will give approximately the total load the system can handle." << std::endl
			  << "\t\t Workload parameters:" << std::endl
			  << "\t\t\t num_workers: (int, default 1) the number of workers you're using" << std::endl
			  << "\t\t\t use_all_models: (bool, default 1) load all models or just resnet50_v2" << std::endl
			  << "\t\t\t load_factor: (float, default 1.0) the workload will generate approximately this much load to the system.  e.g. 0.5 will load by approximately 1/2; 2.0 will overload by a factor of 2" << std::endl
			  << "\t\t\t memory_load_factor: (1, 2, 3, or 4; default 4):" << std::endl
			  << "\t\t\t\t 1: loads approx. 200 models" << std::endl
			  << "\t\t\t\t 2: loads approx. 800 models" << std::endl
			  << "\t\t\t\t 3: loads approx. 1800 models" << std::endl
			  << "\t\t\t\t 4: loads approx. 4000 models" << std::endl
			  << "\t\t\t interval: (int, default 60) interval duration in seconds" << std::endl
			  << "\t\t\t trace: (int, 1 to 13 inclusive, default 1) trace ID to replay" << std::endl
			  << "\t\t\t randomise: (bool, default false) randomize each client's starting point in the trace" << std::endl

			  << "\t azure_scalability_exp" << std::endl
			  << "\t\t Description: Same as the azure workload above, but with an added mechanism to periodically increase the load factor. Only the necessary workload parameters have been retained." << std::endl
			  << "\t\t Workload parameters:" << std::endl
			  << "\t\t\t num_workers: (int, default 1) the number of workers you're using" << std::endl
			  << "\t\t\t load_factor_min: (float, default 0.1) the minimum load factor" << std::endl
			  << "\t\t\t load_factor_max: (float, default 2.0) the maximum load factor" << std::endl
			  << "\t\t\t load_factor_inc: (float, default 2.0) the factor by which load factor is incremented" << std::endl
			  << "\t\t\t load_factor_period: (float, default 1.0) the period in seconds after which the load factor is incremented" << std::endl
			  << "\t\t\t memory_load_factor: (1, 2, 3, or 4; default 4):" << std::endl
			  << "\t\t\t\t 1: loads approx. 200 models" << std::endl
			  << "\t\t\t\t 2: loads approx. 800 models" << std::endl
			  << "\t\t\t\t 3: loads approx. 1800 models" << std::endl
			  << "\t\t\t\t 4: loads approx. 4000 models" << std::endl

			  << "\tbursty_experiment" << std::endl
			  << "\t\t\t num_models: (int, default 3600) number of 'major' workload models" << std::endl;
}

int main(int argc, char *argv[])
{
	threading::initProcess();

	if (argc < 3) {
		printUsage();
		return 1;
	}
	std::string workload = argv[2];
	auto address = split(std::string(argv[1]));

	std::cout << "Running workload `" << workload
	          << "` on " << address.first 
	          << ":" << address.second << std::endl;

	if (util::client_inputs_disabled()) {
		std::cout << "Client inputs are DISABLED.  Set CLOCKWORK_DISABLE_INPUTS=0 to enable inputs." << std::endl;
	} else {
		std::cout << "Client is sending inputs with requests.  Set CLOCKWORK_DISABLE_INPUTS=1 to disable inputs." << std::endl;		
	}

	bool verbose = false; // Log every request and response?
	bool summary = true;  // Log summary once per second?

	clockwork::Client *client = clockwork::Connect(
		address.first, address.second, 
		verbose, summary);

	workload::Engine* engine;
	if (workload == "example") 
		engine = workload::example(client);
	else if (workload == "spam") {
		if (argc > 3) engine = workload::spam(client, argv[3]);
		else engine = workload::spam(client);
	}
	else if (workload == "single-spam") 
		engine = workload::single_spam(client);
	else if (workload == "fill_memory") 
		engine = workload::fill_memory(client);
	else if (workload == "simple")
		engine = workload::simple(client);
	else if (workload == "simple-slo-factor")
		engine = workload::simple_slo_factor(client);
	else if (workload == "simple-parametric")
		engine = workload::simple_parametric(
			client,
			std::stoul(argv[3]),	// total number of models
			std::stoul(argv[4]),	// number of clients
			std::stoul(argv[5]),	// concurrency per client
			std::stoul(argv[6]));	// number of requests per client
	else if (workload == "poisson-open-loop")
		engine = workload::poisson_open_loop(client, std::stoul(argv[3]),
			std::stod(argv[4]));
	else if (workload == "scalability-exp-1")
    // num-models rate-min rate-max rate-factor rate-op
    engine = workload::scalability_experiment_1(
      client,
      std::stoul(argv[3]),    // total number of models
      std::stod(argv[4]),     // total request rate (minimum)
      std::stod(argv[5]),     // total request rate (maximum)
      std::stod(argv[6]),     // request rate increment factor
			std::string(argv[7]),   // request rate increment operator
			std::stoull(argv[8]));	// seconds between rate increments
	else if (workload == "slo-exp-1")
		engine = workload::slo_experiment_1(
			client,
			std::string(argv[3]), 	// model name
			std::stoul(argv[4]),	// num of copies
			std::string(argv[5]),	// arrival distribution
			std::stod(argv[6]),		// arrival rate (requests/second)
			std::stoull(argv[7]),	// starting slo
			std::stoull(argv[8]),	// ending slo
			std::stod(argv[9]),		// slo increase factor
			std::string(argv[10]),	// slo increase operator
			std::stoull(argv[11]));	// slo step duration (in seconds)
	else if (workload == "slo-exp-2")
		engine = workload::slo_experiment_2(
			client,
			std::string(argv[3]), 	// model name
			std::stoul(argv[4]),	// copies for FG models
			std::string(argv[5]),	// arrival distribution for FG model clients
			std::stod(argv[6]),		// arrival rate (requests/second) for FG model clients
			std::stoull(argv[7]),	// starting SLO factor for FG models
			std::stoull(argv[8]),	// ending SLO factor for FG models
			std::stod(argv[9]),		// SLO increase factor for FG models
			std::string(argv[10]),	// SLO increase operator for FG models
			std::stoull(argv[11]),	// SLO step duration (in seconds) for FG models
			std::stoul(argv[12]),	// copies for BG models
			std::stoul(argv[13]),	// concurrency for BG model clients
			std::stod(argv[14]));	// SLO factor for BG models
	else if (workload == "azure") {
		int i = 2;
		unsigned num_workers = argc > ++i ? atoi(argv[i]) : 1;
		bool use_all_models = argc > ++i ? (atoi(argv[i])!=0) : true;
		double load_factor = argc > ++i ? atof(argv[i]) : 1.0;
		unsigned memory_load_factor = argc > ++i ? atoi(argv[i]) : 4;
		unsigned interval_duration_seconds = argc > ++i ? atoi(argv[i]) : 60;
		unsigned trace_id = argc > ++i ? atoi(argv[i]) : 1;
		bool randomise_start = argc > ++i ? atoi(argv[i]) != 0 : false;
		engine = workload::azure2(client,
			num_workers,
			use_all_models,
			load_factor,
			memory_load_factor,
			interval_duration_seconds,
			trace_id,
			randomise_start
		);
	} else if (workload == "azure_scalability_exp") {
		engine = workload::azure_scalability_exp(
      client,
      atoi(argv[3]),  // number of workers
      atof(argv[4]),  // minimum load factor
      atof(argv[5]),  // maximum load factor
      atof(argv[6]),  // factor by which load factor is incremented
      atoi(argv[7]),  // seconds after which load factor is incremented
      atoi(argv[8])); // memory load factor
	} else if (workload == "comparison_experiment") {
		int i = 2;
		unsigned num_models = argc > ++i ? atoi(argv[i]) : 15;
		unsigned total_requests = argc > ++i ? atoi(argv[i]) : 1000;
		engine = workload::comparison_experiment(client, num_models, total_requests);
	} else if (workload == "comparison_experiment2") {
		int i = 2;
		unsigned num_models = argc > ++i ? atoi(argv[i]) : 15;
		unsigned concurrency = argc > ++i ? atoi(argv[i]) : 16;
		engine = workload::comparison_experiment_closed(client, num_models, concurrency);
	} else if (workload == "azure_half")
		engine = workload::azure_parameterized(client);
	else if (workload == "azure_small")
		engine = workload::azure_small(client);
	else if (workload == "azure_single")
		engine = workload::azure_single(client);
	else if (workload == "azure_fast")
		engine = workload::azure_fast(client);
	else if (workload == "bursty_experiment") {
		int i = 2;
		unsigned num_models = argc > ++i ? atoi(argv[i]) : 3600;
		engine = workload::bursty_experiment2(client, 1, 1, num_models);
	} else {
		std::cout << "Unknown workload " << workload << std::endl << std::endl;
		printUsage();
		return 1;
	}

	engine->Run(client);
}
