#include "clockwork/network/controller.h"
#include "clockwork/controller/direct_controller.h"
#include "clockwork/controller/stress_test_controller.h"
#include "clockwork/controller/smart_scheduler.h"
#include "clockwork/controller/concurrent_infer_and_load_scheduler.h"
#include "clockwork/controller/infer5/infer5_scheduler.h"
#include "clockwork/telemetry/controller_request_logger.h"
#include <csignal>
#include <sstream>
#include <string>
#include <vector>
#include "clockwork/thread.h"


using namespace clockwork;
using namespace clockwork::controller;

RequestTelemetryLogger* logger = nullptr;

void signalHandler(int signum) {
    std::cout << "Interrupt signal (" << signum << ") received." << std::endl;
    if (logger != nullptr) { logger->shutdown(false); }
    std::cout << "Clockwork Controller Exiting" << std::endl;
    exit(signum);
}


void printUsage() {
    std::cerr << "Usage: controller [CONTROLLERTYPE] [workers] [options] "
              << "[workload parameters (if required)]" << std::endl
              << "Available workloads with parameters:" << std::endl
              << "  example" << std::endl
              << "  spam" << std::endl
              << "  simple" << std::endl
              << "  simple-parametric num_models concurrency requests_per_model"
              << std::endl
              << "  azure" << std::endl
              << "  azure_small" << std::endl;
}


void show_usage() {
    std::stringstream s;
    s << "USAGE:\n";
    s << "  controller [TYPE] [WORKERS] [OPTIONS]\n";
    s << "DESCRIPTION\n";
    s << "  Run the controller of the given TYPE. Connects to the specified workers. All  \n";
    s << "  subsequent options are controller-specific and passed to that controller.     \n";
    s << "TYPE\n";
    s << "  DIRECT    Used for testing\n";
    s << "  ECHO      Used for testing\n";
    s << "  STRESS    Used for testing\n";
    s << "  INFER5    The Clockwork Scheduler.  You should usually be using this.  Options:\n";
    s << "       generate_inputs    (bool, default false)  Should inputs and outputs be generated if not present.  Set to true to test network capacity\n";
    s << "       max_gpus           (int, default 100)  Set to a lower number to limit the number of GPUs.\n";
    s << "       schedule_ahead     (int, default 10000000)  How far ahead, in nanoseconds, should the scheduler schedule.\n";
    s << "       default_slo        (int, default 100000000)  The default SLO to use if client's don't specify slo_factor.  Default 100ms\n";
    s << "       max_exec        (int, default 25000000)  Don't use batch sizes >1, whose exec time exceeds this number.  Default 25ms \n";
    s << "       max_batch        (int, default 16)  Don't use batch sizes that exceed this number.  Default 16. \n";
    s << "WORKERS\n";
    s << "  Comma-separated list of worker host:port pairs.  e.g.:                        \n";
    s << "    volta03:12345,volta04:12345,volta05:12345                                   \n";
    s << "OPTIONS\n";
    s << "  -h,  --help\n";
    s << "        Print this message\n";
    s << "All other options are passed to the specific scheduler on init\n";
    std::cout << s.str();
}

std::vector<std::string> split(std::string string, char delimiter = ',') {
    std::stringstream ss(string);
    std::vector<std::string> result;

    while( ss.good() )
    {
        std::string substr;
        getline( ss, substr, delimiter);
        result.push_back( substr );
    }
    return result;
}

int main(int argc, char *argv[]) {
    if ( argc < 3) {
        show_usage();
        return 1;
    }

    // register signal SIGINT and signal handler
    signal(SIGTERM, signalHandler);
    signal(SIGINT, signalHandler);

    threading::initProcess();

    std::cout << "Starting Clockwork Controller" << std::endl;
    
    std::string controller_type = argv[1];

    std::vector<std::string> workers = split(argv[2]);
    std::vector<std::pair<std::string, std::string>> worker_host_port_pairs;
    for (std::string worker : workers) {
        std::vector<std::string> p = split(worker,':');
        worker_host_port_pairs.push_back({p[0], p[1]});
    }

    int client_requests_listen_port = 12346;

    std::string actions_filename = util::get_controller_log_dir() + "/clockwork_action_log.tsv";
    std::string requests_filename = util::get_controller_log_dir() + "/clockwork_request_log.tsv";

    if (controller_type == "DIRECT") {
        DirectControllerImpl* controller = new DirectControllerImpl(client_requests_listen_port, worker_host_port_pairs);
        controller->join();
    } else if (controller_type == "STRESS") {
        StressTestController* controller = new StressTestController(client_requests_listen_port, worker_host_port_pairs);
        controller->join();
    } else if (controller_type == "INFER4") {
        int i = 2;
        bool generate_inputs = argc > ++i ? atoi(argv[i]) != 0 : false;
        int max_gpus = argc > ++i ? std::stoull(argv[i]) : 100;
        uint64_t schedule_ahead = argc > ++i ? std::stoull(argv[i]) : 10000000UL;
        uint64_t default_slo = argc > ++i ? std::stoull(argv[i]) : 100000000UL;
        uint64_t max_exec_time = argc > ++i ? std::stoull(argv[i]) : 250000000UL;
        int max_batch_size = argc > ++i ? atoi(argv[i]) : 16;
        std::cout << "Logging requests to " << requests_filename << std::endl;
        std::cout << "Logging actions to " << actions_filename << std::endl;
        Scheduler* scheduler = new scheduler::infer4::Scheduler(
            default_slo,
            schedule_ahead, schedule_ahead,
            generate_inputs,
            max_gpus,
            max_exec_time,
            max_batch_size,
            actions_filename
        );
        controller::ControllerWithStartupPhase* controller = new controller::ControllerWithStartupPhase(
            client_requests_listen_port,
            worker_host_port_pairs,
            1000000000UL, // 10s load stage timeout
            new controller::ControllerStartup(max_batch_size, max_exec_time), // in future the startup type might be customizable
            scheduler,
            ControllerRequestTelemetry::log_and_summarize(
                requests_filename,     // 
                10000000000UL
            )
        );
        controller->join();
    } else if (controller_type == "INFER5") {
        int i = 2;
        bool generate_inputs = argc > ++i ? atoi(argv[i]) != 0 : false;
        int max_gpus = argc > ++i ? std::stoull(argv[i]) : 100;
        uint64_t schedule_ahead = argc > ++i ? std::stoull(argv[i]) : (generate_inputs ? 15000000UL : 10000000UL);
        uint64_t default_slo = argc > ++i ? std::stoull(argv[i]) : 100000000UL;
        uint64_t max_exec_time = argc > ++i ? std::stoull(argv[i]) : 250000000UL;
        int max_batch_size = argc > ++i ? atoi(argv[i]) : 8;
        std::cout << "Logging requests to " << requests_filename << std::endl;
        std::cout << "Logging actions to " << actions_filename << std::endl;
        Scheduler* scheduler = new scheduler::infer5::Scheduler(
            default_slo,
            schedule_ahead, schedule_ahead,
            generate_inputs,
            max_gpus,
            max_exec_time,
            max_batch_size,
            actions_filename
        );
        controller::ControllerWithStartupPhase* controller = new controller::ControllerWithStartupPhase(
            client_requests_listen_port,
            worker_host_port_pairs,
            1000000000UL, // 10s load stage timeout
            new controller::ControllerStartup(max_batch_size, max_exec_time), // in future the startup type might be customizable
            scheduler,
            ControllerRequestTelemetry::log_and_summarize(
                requests_filename,     // 
                10000000000UL
            )
        );
        controller->join();
    } else if (controller_type == "ECHO") {
        Scheduler* scheduler = new EchoScheduler();
        controller::ControllerWithStartupPhase* controller = new controller::ControllerWithStartupPhase(
            client_requests_listen_port,
            worker_host_port_pairs,
            10000000000UL, // 10s load stage timeout
            new controller::ControllerStartup(), // in future the startup type might be customizable
            scheduler,
            ControllerRequestTelemetry::log_and_summarize(
                requests_filename,     // 
                10000000000UL           // print request summary every 10s
            )
        );
        controller->join();
    } else if (controller_type == "SMART") {

		int i = 2;
        int max_gpus = argc > ++i ? std::stoull(argv[i]) : 100;
        uint64_t default_slo = argc > ++i ? std::stoull(argv[i]) : 100000000UL;
        uint64_t max_exec_time = argc > ++i ? std::stoull(argv[i]) : 25000000UL;
        int max_batch_size = argc > ++i ? atoi(argv[i]) : 8;
		std::string action_telemetry_file = argc > ++i ? argv[i] : actions_filename;
        Scheduler* scheduler = new SmartScheduler(
            default_slo,
            max_gpus,
            max_exec_time,
            max_batch_size,
			action_telemetry_file
        );

		controller::ControllerWithStartupPhase* controller = new controller::ControllerWithStartupPhase(
            client_requests_listen_port,
            worker_host_port_pairs,
            1000000000UL, // 10s load stage timeout
            new controller::ControllerStartup(), // in future the startup type might be customizable
            scheduler,
            ControllerRequestTelemetry::log_and_summarize(
                requests_filename,     // 
                10000000000UL           // print request summary every 10s
            )
        );
        controller->join();
		
    } else {
        std::cout << "Invalid controller type " << controller_type << std::endl;
        show_usage();
    }

    std::cout << "Clockwork Controller Exiting" << std::endl;
}
