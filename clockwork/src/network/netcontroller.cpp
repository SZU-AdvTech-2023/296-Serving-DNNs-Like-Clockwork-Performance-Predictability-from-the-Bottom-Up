#include "clockwork/network/controller.h"
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

using namespace clockwork;

class ClientEchoController : public clientapi::ClientAPI {
public:
	size_t output_size = 1000;

	network::controller::Server* client_facing_server;
	ClientEchoController(int client_port) {
		client_facing_server = new network::controller::Server(this, client_port);
	}

	// clientapi -- requests from clients call these functions
	virtual void uploadModel(clientapi::UploadModelRequest &request, std::function<void(clientapi::UploadModelResponse&)> callback) {
		clientapi::UploadModelResponse rsp;
		rsp.header.user_request_id = request.header.user_request_id;
		rsp.header.status = clockworkSuccess;
		callback(rsp);
	}

	virtual void infer(clientapi::InferenceRequest &request, std::function<void(clientapi::InferenceResponse&)> callback) {
		clientapi::InferenceResponse rsp;
		rsp.header.user_request_id = request.header.user_request_id;
		rsp.header.status = clockworkSuccess;
		rsp.output_size = 0;//output_size;
		rsp.output = nullptr;//static_cast<char*>(malloc(output_size));;
		callback(rsp);
	}

	virtual void evict(clientapi::EvictRequest &request, std::function<void(clientapi::EvictResponse&)> callback) {
		clientapi::EvictResponse rsp;
		rsp.header.user_request_id = request.header.user_request_id;
		rsp.header.status = clockworkSuccess;
		callback(rsp);
	}

	virtual void loadRemoteModel(clientapi::LoadModelFromRemoteDiskRequest &request, std::function<void(clientapi::LoadModelFromRemoteDiskResponse&)> callback) {
		clientapi::LoadModelFromRemoteDiskResponse rsp;
		rsp.header.user_request_id = request.header.user_request_id;
		rsp.header.status = clockworkSuccess;
		callback(rsp);
	}

	virtual void ls(clientapi::LSRequest &request, std::function<void(clientapi::LSResponse&)> callback) {
		clientapi::LSResponse rsp;
		rsp.header.user_request_id = request.header.user_request_id;
		rsp.header.status = clockworkSuccess;
		callback(rsp);
	}
};

int main(int argc, char *argv[])
{
	// model_id, workload_type, rate, burst_factor, burst_length, burst_gap

	if (argc != 2)
	{
		std::cerr << "Usage: netcontroller PORT" << std::endl;
		return 1;
	}

	ClientEchoController* controller = new ClientEchoController(atoi(argv[1]));
	controller->client_facing_server->join();

	std::cout << "Clockwork Client Exiting" << std::endl;
}
