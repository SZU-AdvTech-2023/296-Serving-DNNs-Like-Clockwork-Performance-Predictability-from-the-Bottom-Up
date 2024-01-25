#include "clockwork/network/client.h"
#include "clockwork/api/client_api.h"
#include "clockwork/client.h"
#include <cstdlib>
#include <unistd.h>
#include <libgen.h>
#include "clockwork/test/util.h"
#include <nvml.h>
#include <iostream>
#include "clockwork/util.h"
#include <stdexcept>
#include <vector>
#include <functional>
#include "clockwork/thread.h"

using namespace clockwork;

class SimpleConnection {
public:
	size_t input_size = 602112;
	char* input;
	std::function<void(clientapi::InferenceResponse&)> callback;
	std::atomic_int request_id_seed;

	network::client::Connection* connection;

	std::atomic_int completed;

	std::thread printer;

	SimpleConnection(network::client::Connection* connection) : connection(connection), request_id_seed(0), completed(0) {
		input = static_cast<char*>(malloc(input_size));
		callback = std::bind(&SimpleConnection::inferComplete, this, std::placeholders::_1);
		this->printer = std::thread(&SimpleConnection::printThread, this);
		threading::initLoggerThread(printer);
	}

	void printThread() {
		uint64_t print_every = 1000000000UL;
		uint64_t last_print = util::now();
		int last_completed = 0;
		size_t message_size = (32 + input_size); // 32 is approx overhead of request header + params
		while (true) {
			uint64_t now = util::now();
			if (last_print + print_every < now) {
				int new_completed = completed;
				uint64_t duration = (now - last_print); // duration in nanoseconds
				int delta = new_completed - last_completed;
				std::cout << (1000 * (message_size * delta)) / duration << " MB/s" << std::endl;
				last_completed = new_completed;
				last_print = now;
			}
		}
	}

	void sendInfer() {
		clientapi::InferenceRequest request;
		request.header.user_id = 0;
		request.header.user_request_id = request_id_seed++;
		request.model_id = 0;
		request.batch_size = 1;
		request.input_size = input_size;
		request.input = input;
		connection->infer(request, callback);
	}

	void inferComplete(clientapi::InferenceResponse &response) {
		CHECK(response.header.status == clockworkSuccess);
		completed++;
		sendInfer();
	}

};

int main(int argc, char *argv[])
{
	// model_id, workload_type, rate, burst_factor, burst_length, burst_gap

	if (argc != 3)
	{
		std::cerr << "Usage: netclient HOST PORT" << std::endl;
		return 1;
	}

	network::client::ConnectionManager* manager = new network::client::ConnectionManager();
	network::client::Connection* clockwork_connection = manager->connect(argv[1], argv[2]);

	SimpleConnection* connection = new SimpleConnection(clockwork_connection);

	int concurrency = 10;
	for (int i = 0; i < concurrency; i++) {
		connection->sendInfer();
	}

	manager->join();

	std::cout << "Clockwork Client Exiting" << std::endl;
}
