#include "clockwork/dummy/worker_dummy.h"
#include "clockwork/dummy/network/worker_dummy.h"
#include "clockwork/config.h"
#include "clockwork/thread.h"
#include <string>

void show_usage() {
    std::stringstream s;
    s << "USAGE:\n";
    s << "  worker [OPTIONS]\n";
    s << "DESCRIPTION\n";
    s << "  Run a clockwork worker.  Will use all GPUs on this machine.\n";
    s << "OPTIONS\n";
    s << "  -h,  --help\n";
    s << "        Print this message\n";
    s << "  -n,  --num_gpus\n";
    s << "        The number of GPUs to emulate; defaults to 1\n";
    s << "  -c,  --config\n";
    s << "        Specify a Clockwork config to use; otherwise the defaults will be used\n";
    s << "  -p,  --port\n";
    s << "        Specify a port to run the server on; default 12345\n";
    std::cout << s.str();
}

int main(int argc, char *argv[]) {
	// Read args
	std::string config_file_path;
	uint64_t num_gpus = 1;
    int port = 12345;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-h") || (arg == "--help")) {
            show_usage();
            return 0;
        } else if ((arg == "-c") || (arg == "--config")) {
        	config_file_path = argv[++i];
        } else if ((arg == "-n") || (arg == "--num_gpus")) {
        	num_gpus = std::stoul(argv[++i]);
        } else if ((arg == "-p") || (arg == "--port")) {
            port = std::atoi(argv[++i]);
        } else {
        	std::cout << "Unknown option " << arg << std::endl;
        	return 1;
        }
    }
    
    threading::initProcess();

	std::cout << "Starting Clockwork Worker" << std::endl;

	ClockworkWorkerConfig config(config_file_path);
	config.num_gpus = num_gpus;

	clockwork::ClockworkDummyWorker* clockwork = new clockwork::ClockworkDummyWorker(config);
	clockwork::network::worker::Server* server = new clockwork::network::worker::Server(clockwork, port);
	clockwork->setController(server);

	clockwork->join();

	std::cout << "Clockwork Worker Exiting" << std::endl;
}
