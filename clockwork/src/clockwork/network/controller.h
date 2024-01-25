#ifndef _CLOCKWORK_NETWORK_CONTROLLER_H_
#define _CLOCKWORK_NETWORK_CONTROLLER_H_

#include <atomic>
#include <string>
#include <asio.hpp>
#include <functional>
#include <tbb/concurrent_queue.h>
#include "clockwork/worker.h"
#include "clockwork/network/network.h"
#include "clockwork/network/worker_api.h"
#include "clockwork/network/client_api.h"

namespace clockwork {
namespace network {
namespace controller {

using asio::ip::tcp;

typedef std::function<void(void)> Callback;

/* Controller side of the Controller<>Worker API network impl.
Represents a connection to a single worker. */
class WorkerConnection : public message_connection, public message_handler, public workerapi::Worker  {
private:
	workerapi::Controller* controller;
	message_sender msg_tx_;
	Callback callback_;

public:
	std::atomic_bool connected;

	WorkerConnection(asio::io_service &io_service, workerapi::Controller* controller);

protected:
	virtual message_rx *new_rx_message(message_connection *tcp_conn, uint64_t header_len,
			uint64_t body_len, uint64_t msg_type, uint64_t msg_id);

	virtual void ready();
	virtual void closed();

	virtual void aborted_receive(message_connection *tcp_conn, message_rx *req);

	virtual void completed_receive(message_connection *tcp_conn, message_rx *req);

	virtual void completed_transmit(message_connection *tcp_conn, message_tx *req);

	virtual void aborted_transmit(message_connection *tcp_conn, message_tx *req);

public:

	virtual void sendActions(std::vector<std::shared_ptr<workerapi::Action>> &actions);

	void sendAction(std::shared_ptr<workerapi::Action> action);

	void setTransmitCallback(Callback callback);

};

/* WorkerManager is used to connect to multiple workers.
Connect can be called multiple times, to connect to multiple workers.
Each WorkerConnection will handle a single worker.
The WorkerManager internally has just one IO thread to handle IO for all connections */
class WorkerManager {
private:
	std::atomic_bool alive;
	asio::io_service io_service;
	std::thread network_thread;

public:
	WorkerManager();

	void run();

	void shutdown(bool awaitCompletion = false);

	void join();
	WorkerConnection* connect(std::string host, std::string port, workerapi::Controller* controller);

};

class Server;

/* Controller side of the Client<>Controller API network impl */
class ClientConnection : public message_connection, public message_handler  {
private:
	Server* server;

public:
	message_sender msg_tx_;
	std::atomic_bool connected;

	ClientConnection(asio::io_service &io_service, Server* server);

protected:
	virtual message_rx *new_rx_message(message_connection *tcp_conn, uint64_t header_len,
			uint64_t body_len, uint64_t msg_type, uint64_t msg_id);

	virtual void ready();
	virtual void closed();

	virtual void aborted_receive(message_connection *tcp_conn, message_rx *req);

	virtual void completed_receive(message_connection *tcp_conn, message_rx *req);

	virtual void completed_transmit(message_connection *tcp_conn, message_tx *req);

	virtual void aborted_transmit(message_connection *tcp_conn, message_tx *req);

};


/* Controller-side server for the Client API.  
Accepts connections and requests from users/clients.
Creates ClientConnection for incoming client connections.
The Server internally maintains one IO thread for all client connections. */
class Server {
private:
	clientapi::ClientAPI* api;
	std::atomic_bool alive;
	asio::io_service io_service;
	std::vector<std::thread> network_threads;
	std::vector<std::thread> process_threads;
	tcp::acceptor* acceptor;
	struct client_message { ClientConnection* client; message_rx* req; };
	tbb::concurrent_bounded_queue<client_message> messages;


public:
	Server(clientapi::ClientAPI* api, int port = 12346);

	void shutdown(bool awaitShutdown);
	void join();
	void run_network_thread();
	void run_process_thread();

	void completed_receive(ClientConnection* client, message_rx *req);

private:
	void start_accept(tcp::acceptor* acceptor);

	void handle_accept(ClientConnection* connection, tcp::acceptor* acceptor, const asio::error_code& error);
	void process_message(ClientConnection* client, message_rx *req);

};

}
}
}

#endif