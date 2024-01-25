#ifndef _CLOCKWORK_NETWORK_WORKER_DUMMY_H_
#define _CLOCKWORK_NETWORK_WORKER_DUMMY_H_

#include <utility>
#include <cstring>
#include <string>
#include <iostream>
#include <boost/bind.hpp>
#include <asio.hpp>
#include "clockwork/dummy/worker_dummy.h"
#include "clockwork/network/network.h"
#include "clockwork/network/worker_api.h"

namespace clockwork {
namespace network {
namespace worker {

using asio::ip::tcp;

class ConnectionStats {
public:
	std::atomic_uint64_t load;
	std::atomic_uint64_t evict;
	std::atomic_uint64_t infer;
	std::atomic_uint64_t total_pending;
	std::atomic_uint64_t errors;
};

/* Worker side of the Controller<>Worker API network impl.
A connection to the Clockwork Controller */
class Connection : public message_connection, public message_handler, public workerapi::Controller  {
private:
	ClockworkDummyWorker* worker;
	message_sender msg_tx_;
	std::function<void(void)> on_close;
	std::atomic_bool alive;
	ConnectionStats stats;
	std::thread printer;

public:
	Connection(asio::io_service &io_service, ClockworkDummyWorker* worker, std::function<void(void)> on_close);

private:
	void print();

protected:

	virtual message_rx *new_rx_message(message_connection *tcp_conn, uint64_t header_len,
			uint64_t body_len, uint64_t msg_type, uint64_t msg_id);

	virtual void aborted_receive(message_connection *tcp_conn, message_rx *req);

	virtual void completed_receive(message_connection *tcp_conn, message_rx *req);

	virtual void completed_transmit(message_connection *tcp_conn, message_tx *req);

	virtual void aborted_transmit(message_connection *tcp_conn, message_tx *req);

	virtual void ready();
	virtual void closed();

public:
	
	// workerapi::Controller::sendResult
	virtual void sendResult(std::shared_ptr<workerapi::Result> result);

};

/* Worker-side server that accepts connections from the Controller */
class Server : public workerapi::Controller {
private:
	std::atomic_bool is_started;
	ClockworkDummyWorker* worker;
	asio::io_service io_service;
	std::thread network_thread;

	Connection* current_connection;

public:
	Server(ClockworkDummyWorker* worker, int port = 12345);
	~Server();

	void shutdown(bool awaitShutdown);
	void join();
	void run(int port);
	
	// workerapi::Controller::sendResult
	virtual void sendResult(std::shared_ptr<workerapi::Result> result);

private:
	void start_accept(tcp::acceptor* acceptor);

	void handle_accept(Connection* connection, tcp::acceptor* acceptor, const asio::error_code& error);

};

}
}
}

#endif