#include "clockwork/dummy/network/worker_dummy.h"
#include "clockwork/util.h"
#include <sstream>
#include "clockwork/thread.h"

namespace clockwork {
namespace network {
namespace worker {

using asio::ip::tcp;

bool verbose = false;


class infer_action_rx_using_io_pool : public infer_action_rx {
private:
  //clockwork::MemoryPool* host_io_pool;


public:
	/*
  infer_action_rx_using_io_pool(clockwork::MemoryPool* host_io_pool):
    infer_action_rx(), host_io_pool(host_io_pool) {
  }*/

  infer_action_rx_using_io_pool():
    infer_action_rx() {
  }

  virtual ~infer_action_rx_using_io_pool() {
    delete static_cast<uint8_t*>(body_);
  }

  virtual void get(workerapi::Infer &action) {
  	infer_action_rx::get(action);

  	// Copy the input body into a cached page
    //action.input = host_io_pool->alloc(body_len_);
    //CHECK(action.input != nullptr) << "Unable to alloc from host_io_pool for infer action input";
    //std::memcpy(action.input, body_, body_len_);
  }
};

class infer_result_tx_using_io_pool : public infer_result_tx {
private:
  //clockwork::MemoryPool* host_io_pool;

public:
	/*
  infer_result_tx_using_io_pool(clockwork::MemoryPool* host_io_pool):
    infer_result_tx(), host_io_pool(host_io_pool) {
  }*/

  infer_result_tx_using_io_pool():
    infer_result_tx() {
  }

  virtual ~infer_result_tx_using_io_pool() {
  	delete static_cast<uint8_t*>(body_);
  }

  virtual void set(workerapi::InferResult &result) {
  	// Memory allocated with cudaMallocHost doesn't play nicely with asio.
  	// Until we solve it, just do a memcpy here :(
  	infer_result_tx::set(result);
  	if (result.output_size > 0) {
  		body_ = new uint8_t[result.output_size];
  	}
    //body_ = new uint8_t[result.output_size];
    //std::memcpy(body_, result.output, result.output_size);
    //host_io_pool->free(result.output);
  }

};

/*
class InferUsingIOPool : public workerapi::Infer {
public:
	clockwork::MemoryPool* host_io_pool;
	InferUsingIOPool(clockwork::MemoryPool* host_io_pool):
		host_io_pool(host_io_pool) {}
	~InferUsingIOPool() {
		host_io_pool->free(input);
	}
};*/

Connection::Connection(asio::io_service &io_service, ClockworkDummyWorker* worker, std::function<void(void)> on_close) :
		message_connection(io_service, *this),
		msg_tx_(this, *this),
		worker(worker),
		on_close(on_close),
		stats(),
		alive(true) {
}

class StatTracker {
public:
	uint64_t previous_value = 0;
	uint64_t update(std::atomic_uint64_t &counter) {
		uint64_t current_value = counter.load();
		uint64_t delta = current_value - previous_value;
		previous_value = current_value;
		return delta;
	}
};

void Connection::print() {
	uint64_t print_every = 10000000000UL; // 10s
	uint64_t last_print = util::now();

	StatTracker load;
	StatTracker evict;
	StatTracker infer;
	StatTracker errors;

	while (alive) {
		uint64_t now = util::now();
		if (last_print + print_every > now) {
			usleep(200000); // 200ms sleep
			continue;
		}

		uint64_t dload = load.update(stats.load);
		uint64_t dinfer = infer.update(stats.infer);
		uint64_t devict = evict.update(stats.evict);

		uint64_t pending = stats.total_pending;
		uint64_t derrors = errors.update(stats.errors);

		std::stringstream s;
		s << "Clock Skew=" << estimate_clock_delta()
		  << "  RTT=" << estimate_rtt()
		  << "  LdWts=" << dload
		  << "  Inf=" << dinfer
		  << "  Evct=" << devict
		  << "  || Total Pending=" << pending
		  << "  Errors=" << derrors
		  << std::endl;

		std::cout << s.str();
		last_print = now;
	}
}

message_rx* Connection::new_rx_message(message_connection *tcp_conn, uint64_t header_len,
		uint64_t body_len, uint64_t msg_type, uint64_t msg_id) {
	using namespace clockwork::workerapi;

	if (msg_type == ACT_LOAD_MODEL_FROM_DISK) {
		auto msg = new load_model_from_disk_action_rx();
		msg->set_msg_id(msg_id);
		return msg;
	} else if (msg_type == ACT_LOAD_WEIGHTS) {
		auto msg = new load_weights_action_rx();
		msg->set_msg_id(msg_id);
		return msg;
	} else if (msg_type == ACT_INFER) {
		//auto msg = new infer_action_rx_using_io_pool(worker->runtime->manager->host_io_pool);
		auto msg = new infer_action_rx_using_io_pool();
		msg->set_body_len(body_len);
		msg->set_msg_id(msg_id);
		return msg;
	} else if (msg_type == ACT_GET_WORKER_STATE) {
		auto msg = new get_worker_state_action_rx();
		msg->set_msg_id(msg_id);
		return msg;
	} else if (msg_type == ACT_EVICT_WEIGHTS) {
		auto msg = new evict_weights_action_rx();
		msg->set_msg_id(msg_id);
		return msg;
	} else if (msg_type == ACT_CLEAR_CACHE) {
		auto msg = new clear_cache_action_rx();
		msg->set_msg_id(msg_id);
		return msg;
	}
	
	CHECK(false) << "Unsupported msg_type " << msg_type;
	return nullptr;
}

void Connection::aborted_receive(message_connection *tcp_conn, message_rx *req) {
	delete req;
}

void Connection::completed_receive(message_connection *tcp_conn, message_rx *req) {
	std::vector<std::shared_ptr<workerapi::Action>> actions;

	uint64_t now = util::now();

	if (auto load_model = dynamic_cast<load_model_from_disk_action_rx*>(req)) {
		auto action = std::make_shared<workerapi::LoadModelFromDisk>();
		load_model->get(*action);
		actions.push_back(action);

		if (!verbose) std::cout << "Received " << actions[0]->str() << std::endl;

	} else if (auto load_weights = dynamic_cast<load_weights_action_rx*>(req)) {
		auto action = std::make_shared<workerapi::LoadWeights>();
		load_weights->get(*action);
		actions.push_back(action);

		stats.load++;
	} else if (auto infer = dynamic_cast<infer_action_rx_using_io_pool*>(req)) {
		//auto action = std::make_shared<InferUsingIOPool>(worker->runtime->manager->host_io_pool);
		auto action = std::make_shared<workerapi::Infer>();
		infer->get(*action);
		actions.push_back(action);

		stats.infer++;
	} else if (auto evict = dynamic_cast<evict_weights_action_rx*>(req)) {
		auto action = std::make_shared<workerapi::EvictWeights>();
		evict->get(*action);
		actions.push_back(action);

		stats.evict++;
	} else if (auto clear_cache = dynamic_cast<clear_cache_action_rx*>(req)) {
		auto action = std::make_shared<workerapi::ClearCache>();
		clear_cache->get(*action);
		actions.push_back(action);
	} else if (auto get_worker_state = dynamic_cast<get_worker_state_action_rx*>(req)) {
		auto action = std::make_shared<workerapi::GetWorkerState>();
		get_worker_state->get(*action);
		actions.push_back(action);

		if (!verbose) std::cout << "Received " << actions[0]->str() << std::endl;

	} else {
		CHECK(false) << "Received an unsupported message_rx type";
	}
	if (verbose) std::cout << "Received " << actions[0]->str() << std::endl;

	actions[0]->clock_delta = estimate_clock_delta();
	actions[0]->received = now;

	stats.total_pending++;

	delete req;
	worker->sendActions(actions);
}

void Connection::completed_transmit(message_connection *tcp_conn, message_tx *req) {
	delete req;
}

void Connection::aborted_transmit(message_connection *tcp_conn, message_tx *req) {
	delete req;
}

void Connection::sendResult(std::shared_ptr<workerapi::Result> result) {
	if (verbose) std::cout << "Sending " << result->str() << std::endl;
	using namespace workerapi;
	result->result_sent = util::now() - result->clock_delta;
	if (auto load_model = std::dynamic_pointer_cast<LoadModelFromDiskResult>(result)) {
		auto tx = new load_model_from_disk_result_tx();
		tx->set(*load_model);
		msg_tx_.send_message(*tx);

		if (!verbose) std::cout << "Sending " << result->str() << std::endl;
	} else if (auto load_weights = std::dynamic_pointer_cast<LoadWeightsResult>(result)) {
		auto tx = new load_weights_result_tx();
		tx->set(*load_weights);
		msg_tx_.send_message(*tx);

	} else if (auto infer = std::dynamic_pointer_cast<InferResult>(result)) {
		//auto tx = new infer_result_tx_using_io_pool(worker->runtime->manager->host_io_pool);
		auto tx = new infer_result_tx_using_io_pool();
		tx->set(*infer);
		msg_tx_.send_message(*tx);

	} else if (auto evict_weights = std::dynamic_pointer_cast<EvictWeightsResult>(result)) {
		auto tx = new evict_weights_result_tx();
		tx->set(*evict_weights);
		msg_tx_.send_message(*tx);

	} else if (auto clear_cache = std::dynamic_pointer_cast<ClearCacheResult>(result)) {
		auto tx = new clear_cache_result_tx();
		tx->set(*clear_cache);
		msg_tx_.send_message(*tx);

	} else if (auto get_worker_state = std::dynamic_pointer_cast<GetWorkerStateResult>(result)) {
		auto tx = new get_worker_state_result_tx();
		tx->set(*get_worker_state);
		msg_tx_.send_message(*tx);

		if (!verbose) std::cout << "Sending " << result->str() << std::endl;
	} else if (auto error = std::dynamic_pointer_cast<ErrorResult>(result)) {
		auto tx = new error_result_tx();
		tx->set(*error);
		msg_tx_.send_message(*tx);

		stats.errors++;
	} else {
		CHECK(false) << "Sending an unsupported result type";
	}

	stats.total_pending--;
}

void Connection::ready() {
	this->printer = std::thread(&Connection::print, this);
	threading::initLoggerThread(this->printer);
}

void Connection::closed() {
	alive = false;
	this->on_close();
}

Server::Server(ClockworkDummyWorker* worker, int port) :
		is_started(false),
		worker(worker),
		io_service(),
		network_thread(&Server::run, this, port) {
	threading::initNetworkThread(network_thread);
}

Server::~Server() {}

void Server::shutdown(bool awaitShutdown) {
	io_service.stop();
	if (awaitShutdown) {
		join();
	}
}

void Server::join() {
	while (!is_started);
	network_thread.join();
}

void Server::run(int port) {
	try {
		auto endpoint = tcp::endpoint(tcp::v4(), port);
		is_started.store(true);
		tcp::acceptor acceptor(io_service, endpoint);
		start_accept(&acceptor);
		std::cout << "IO service thread listening on " << endpoint << std::endl;
		io_service.run();
	} catch (std::exception& e) {
		CHECK(false) << "Exception in network thread: " << e.what();
	} catch (const char* m) {
		CHECK(false) << "Exception in network thread: " << m;
	}
	std::cout << "Server exiting" << std::endl;
}

// workerapi::Controller::sendResult
void Server::sendResult(std::shared_ptr<workerapi::Result> result) {
	if (current_connection == nullptr) {
		std::cout << "Dropping result " << result->str() << std::endl;
	} else {
		current_connection->sendResult(result);
	}
}	

void Server::start_accept(tcp::acceptor* acceptor) {
	auto connection = new Connection(acceptor->get_io_service(), worker, [this]{
		this->current_connection = nullptr;
		delete this->current_connection;
	});

	acceptor->async_accept(connection->get_socket(),
		boost::bind(&Server::handle_accept, this, connection, acceptor,
			asio::placeholders::error));
}

void Server::handle_accept(Connection* connection, tcp::acceptor* acceptor, const asio::error_code& error) {
	if (error) {
		throw std::runtime_error(error.message());
	}

	connection->established();
	this->current_connection = connection;
	start_accept(acceptor);
}

}
}
}
