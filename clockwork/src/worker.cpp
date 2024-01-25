#include "clockwork/worker.h"
#include "clockwork/network/worker.h"
#include "clockwork/runtime.h"
#include "clockwork/config.h"
#include "clockwork/worker.h"
#include "clockwork/thread.h"

void show_usage() {
    std::stringstream s;
    s << "USAGE:\n";
    s << "  worker [OPTIONS]\n";
    s << "DESCRIPTION\n";
    s << "  Run a clockwork worker.  Will use all GPUs on this machine.\n";
    s << "OPTIONS\n";
    s << "  -h,  --help\n";
    s << "        Print this message\n";
    s << "  -c,  --config\n";
    s << "        Specify a Clockwork config to use; otherwise the defaults will be used\n";
    s << "  -p,  --port\n";
    s << "        Specify a port to run the server on; default 12345\n";
    std::cout << s.str();
}

int main(int argc, char *argv[]) {
	// Read args
    
	std::string config_file_path;
    int port = 12345;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-h") || (arg == "--help")) {
            show_usage();
            return 0;
        } else if ((arg == "-c") || (arg == "--config")) {
            config_file_path = argv[++i];
        } else if ((arg == "-p") || (arg == "--port")) {
            port = std::atoi(argv[++i]);
        } else {
        	std::cout << "Unknown option " << arg << std::endl;
        	return 1;
        }
    }

    // Init process
	threading::initProcess();
	util::setCudaFlags();
	util::printCudaVersion();

	std::cout << "Starting Clockwork Worker" << std::endl;

	ClockworkWorkerConfig config(config_file_path);

	clockwork::ClockworkWorker* clockwork = new clockwork::ClockworkWorker(config);
	clockwork::network::worker::Server* server = new clockwork::network::worker::Server(clockwork, port);
	clockwork->controller = server;

	threading::setDefaultPriority(); // Revert thread priority
	clockwork->join();

	std::cout << "Clockwork Worker Exiting" << std::endl;
}
