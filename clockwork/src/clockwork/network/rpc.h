#ifndef _CLOCKWORK_NETWORK_RPC_H_
#define _CLOCKWORK_NETWORK_RPC_H_

#include <mutex>
#include <atomic>
#include <clockwork/network/message.h>

namespace clockwork {
namespace network {

class net_rpc_base
{
private:
  uint64_t id;
public:
  virtual uint64_t get_id() {
    return id;
  }

  virtual void set_id(uint64_t id) {
    this->id = id;
  }

  virtual message_tx &request() = 0;
  virtual message_rx &make_response(uint64_t msg_type, uint64_t body_len) = 0;
  virtual void done() = 0;
};

template<class TReq, class TRes>
class net_rpc : public net_rpc_base
{
public:
  net_rpc(std::function<void(TRes&)> c) : req(), comp(c) {}

  virtual void set_id(uint64_t id) {
    net_rpc_base::set_id(id);
    req.set_msg_id(id);
  }

  virtual message_tx &request()
  {
    return req;
  }

  virtual message_rx &make_response(uint64_t msg_type, uint64_t body_len)
  {
    do_make_response(msg_type, body_len);
    return *rsp;
  }

  virtual void done()
  {
    comp(*rsp);
  }

protected:

  virtual void do_make_response(uint64_t msg_type, uint64_t body_len)
  {
    if (msg_type != TRes::MsgType)
      throw "unexpected message type in response";

    rsp = new TRes();
    rsp->set_msg_id(get_id());
  }

public:
  TReq req;
  TRes *rsp;
  std::function<void(TRes&)> comp;
};

template<class TReq, class TRes>
class net_rpc_receive_payload : public net_rpc<TReq, TRes>
{
public:
  net_rpc_receive_payload(std::function<void(TRes&)> c)
    : net_rpc<TReq, TRes>(c) {}

protected:

  virtual void do_make_response(uint64_t msg_type, uint64_t body_len)
  {
    net_rpc<TReq, TRes>::do_make_response(msg_type, body_len);
    net_rpc<TReq, TRes>::rsp->set_body_len(body_len);
  }
};


class net_rpc_conn :
  public message_connection, public message_handler
{

public:
  net_rpc_conn(asio::io_service& io_service)
    : message_connection(io_service, *this), msg_tx_(this, *this), request_id_seed(0)
  {
  }

protected:
  virtual void request_done(net_rpc_base &req) {}

  void send_request(net_rpc_base &rb)
  {
    std::lock_guard<std::mutex> lock(requests_mutex);

    uint64_t request_id = request_id_seed++;
    rb.set_id(request_id);
    requests[request_id] = &rb;
    msg_tx_.send_message(rb.request());
  }

  virtual message_rx *new_rx_message(message_connection *tcp_conn, uint64_t header_len,
      uint64_t body_len, uint64_t msg_type, uint64_t msg_id)
  {
    std::lock_guard<std::mutex> lock(requests_mutex);

    auto it = requests.find(msg_id);
    CHECK(it != requests.end()) << "No RPC request with ID " << msg_id;

    auto rb = it->second;
    message_rx &mrx = rb->make_response(msg_type, body_len);

    return &mrx;
  }

  virtual void aborted_receive(message_connection *tcp_conn, message_rx *req)
  {
  }

  virtual void completed_receive(message_connection *tcp_conn, message_rx *req)
  {
    net_rpc_base* rb = nullptr;
    {
      std::lock_guard<std::mutex> lock(requests_mutex);

      uint64_t msg_id = req->get_msg_id();
      auto it = requests.find(msg_id);
      if (it != requests.end()) {
        rb = it->second;
        requests.erase(it);
      }
    }
    CHECK(rb != nullptr) << "Received response to non-existent request";
    
    rb->done();
    request_done(*rb);
  }

  virtual void completed_transmit(message_connection *tcp_conn, message_tx *req)
  {
  }

  virtual void aborted_transmit(message_connection *tcp_conn, message_tx *req)
  {
  }

  message_sender msg_tx_;
  std::mutex requests_mutex;
  std::map<uint64_t, net_rpc_base *> requests;
  std::atomic_int request_id_seed;

};

}
}

#endif // ndef CLOCKWORK_NET_RPC_H_
