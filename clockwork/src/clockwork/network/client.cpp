#include "clockwork/network/client.h"
#include <sstream>
#include <iomanip>

namespace clockwork {
namespace network {
namespace client {

using asio::ip::tcp;
using namespace clockwork::clientapi;

Connection::Connection(asio::io_service& io_service): net_rpc_conn(io_service), connected(false),
	logger_thread(&Connection::run_logger_thread, this) {
	threading::initLoggerThread(logger_thread);
}

void Connection::ready() {
	connected.store(true);
}

void Connection::run_logger_thread() {
	uint64_t log_every = 10000000000UL;
	uint64_t last_print = 0;
    network::connection_stats previous_stats;
	while (true) {
		uint64_t now = util::now();
		if (last_print + log_every <= now) {

	        network::connection_stats stats;
	        stats += this->stats;
	        stats -= previous_stats;
	        previous_stats = stats;


	        float duration = (now - last_print) / 1000000000.0;
	        stats /= duration;

	        std::stringstream msg;
	        msg << std::fixed << std::setprecision(1);
	        msg << "Client->Controller: ";
	        msg << (stats.bytes_sent / (1024*1024.0)) << "MB/s ";
	        msg << "(" << stats.messages_sent << " msgs) snd, ";
	        msg << (stats.bytes_received / (1024*1024.0)) << "MB/s ";
	        msg << "(" << stats.messages_received << " msgs) rcv, ";
	        msg << std::endl;

	        std::cout << msg.str();
			last_print = now;

		}

		usleep(100000);
	}
}

void Connection::request_done(net_rpc_base &req) {
	delete &req;
}

void Connection::uploadModel(UploadModelRequest &request, std::function<void(UploadModelResponse&)> callback) {
	auto rpc = new net_rpc<msg_upload_model_req_tx, msg_upload_model_rsp_rx>(
	  [callback](msg_upload_model_rsp_rx &rsp) {
	    UploadModelResponse response;
	    rsp.get(response);
	    callback(response);
	  }
	);

	rpc->req.set(request);
	send_request(*rpc);
}

void Connection::infer(InferenceRequest &request, std::function<void(InferenceResponse&)> callback) {
	auto rpc = new net_rpc_receive_payload<msg_inference_req_tx, msg_inference_rsp_rx>(
		[callback](msg_inference_rsp_rx &rsp) {
			InferenceResponse response;
			rsp.get(response);
			callback(response);
		});

	rpc->req.set(request);
	send_request(*rpc);
}

void Connection::evict(EvictRequest &request, std::function<void(EvictResponse&)> callback){
	auto rpc = new net_rpc<msg_evict_req_tx, msg_evict_rsp_rx>(
		[callback](msg_evict_rsp_rx &rsp) {
			EvictResponse response;
			rsp.get(response);
			callback(response);
		});

	rpc->req.set(request);
	send_request(*rpc);
}

  /** This is a 'backdoor' API function for ease of experimentation */
void Connection::loadRemoteModel(LoadModelFromRemoteDiskRequest &request, std::function<void(LoadModelFromRemoteDiskResponse&)> callback) {
	auto rpc = new net_rpc<msg_load_remote_model_req_tx, msg_load_remote_model_rsp_rx>(
		[callback](msg_load_remote_model_rsp_rx &rsp) {
			LoadModelFromRemoteDiskResponse response;
			rsp.get(response);
			callback(response);
		});

	rpc->req.set(request);
	send_request(*rpc);
}

void Connection::ls(LSRequest &request, std::function<void(LSResponse&)> callback) {
	auto rpc = new net_rpc<msg_ls_req_tx, msg_ls_rsp_rx>(
		[callback](msg_ls_rsp_rx &rsp) {
			LSResponse response;
			rsp.get(response);
			callback(response);
		});

	rpc->req.set(request);
	send_request(*rpc);	
}

ConnectionManager::ConnectionManager() : alive(true), network_thread(&ConnectionManager::run, this) {
	threading::initNetworkThread(network_thread);
}

void ConnectionManager::run() {
	while (alive) {
		try {
			asio::io_service::work work(io_service);
			io_service.run();
		} catch (std::exception& e) {
			alive.store(false);
			CHECK(false) << "Exception in network thread: " << e.what();
		} catch (const char* m) {
			alive.store(false);
			CHECK(false) << "Exception in network thread: " << m;
		}
	}
}

void ConnectionManager::shutdown(bool awaitCompletion) {
	alive.store(false);
	io_service.stop();
	if (awaitCompletion) {
		join();
	}
}

void ConnectionManager::join() {
	network_thread.join();
}

Connection* ConnectionManager::connect(std::string host, std::string port) {
	try {
		Connection* c = new Connection(io_service);
		c->connect(host, port);
		std::cout << "Connecting to clockwork @ " << host << ":" << port << std::endl;
		while (alive.load() && !c->connected.load()); // If connection fails, alive sets to false
		std::cout << "Connection established" << std::endl;
		return c;
	} catch (std::exception& e) {
		alive.store(false);
		io_service.stop();
		CHECK(false) << "Exception in network thread: " << e.what();
	} catch (const char* m) {
		alive.store(false);
		io_service.stop();
		CHECK(false) << "Exception in network thread: " << m;
	}
	return nullptr;
}

}
}
}