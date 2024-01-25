#ifndef _CLOCKWORK_NETWORK_CLIENT_H_
#define _CLOCKWORK_NETWORK_CLIENT_H_

#include <atomic>
#include <string>
#include <asio.hpp>
#include "clockwork/worker.h"
#include "clockwork/network/network.h"
#include "clockwork/network/rpc.h"
#include "clockwork/network/client_api.h"

namespace clockwork {
namespace network {
namespace client {

using asio::ip::tcp;
using namespace clockwork::clientapi;

/* Client side of the Client<>Controller API network impl.
Represents a connection of a client to the Clockwork controller */
class Connection: public net_rpc_conn, public ClientAPI {
public:
  std::atomic_bool connected;
  std::thread logger_thread;

  Connection(asio::io_service& io_service);

  void run_logger_thread();

  // net_rpc_conn methods
  virtual void ready();
  virtual void request_done(net_rpc_base &req);

  // clientapi methods
  virtual void uploadModel(UploadModelRequest &request, std::function<void(UploadModelResponse&)> callback);
  virtual void infer(InferenceRequest &request, std::function<void(InferenceResponse&)> callback);
  virtual void evict(EvictRequest &request, std::function<void(EvictResponse&)> callback);
  virtual void loadRemoteModel(LoadModelFromRemoteDiskRequest &request, std::function<void(LoadModelFromRemoteDiskResponse&)> callback);
  virtual void ls(LSRequest &request, std::function<void(LSResponse&)> callback);

};

/* ConnectionManager is used to connect multiple times to the clockwork controller
Connect can be called multiple times to represent multiple clients
The ConnectionManager internally has just one IO thread to handle IO for all connections */
class ConnectionManager {
private:
  std::atomic_bool alive;
  asio::io_service io_service;
  std::thread network_thread;

  void run();

public:
  ConnectionManager();

  void shutdown(bool awaitCompletion = false);
  void join();

  Connection* connect(std::string host, std::string port);

};

}
}
}

#endif