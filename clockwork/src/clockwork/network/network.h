#ifndef _CLOCKWORK_NETWORK_NETWORK_H_
#define _CLOCKWORK_NETWORK_NETWORK_H_

#include <mutex>
#include <utility>
#include <queue>
#include <boost/bind.hpp>
#include <asio.hpp>
#include <atomic>
#include "clockwork/network/message.h"
#include "tbb/concurrent_queue.h"
#include "clockwork/util.h"
#include "clockwork/sliding_window.h"

namespace clockwork {
namespace network {

class message_connection;

class connection_stats {
public:
  std::atomic_uint64_t bytes_sent = 0;
  std::atomic_uint64_t bytes_received = 0;
  std::atomic_uint64_t messages_sent = 0;
  std::atomic_uint64_t messages_received = 0;

  void message_received(uint64_t size) {
    bytes_received += size;
    messages_received++;
  }

  void message_sent(uint64_t size) {
    bytes_sent += size;
    messages_sent++;
  }

  connection_stats& operator= (const connection_stats& rhs) {
    (*this)+=rhs;
    return *this;
  }

  connection_stats& operator+=(const connection_stats& rhs){
    bytes_sent += rhs.bytes_sent;
    bytes_received += rhs.bytes_received;
    messages_sent += rhs.messages_sent;
    messages_received += rhs.messages_received;
    return *this;
  }

  connection_stats& operator-=(const connection_stats& rhs){
    bytes_sent -= rhs.bytes_sent;
    bytes_received -= rhs.bytes_received;
    messages_sent -= rhs.messages_sent;
    messages_received -= rhs.messages_received;
    return *this;
  }

  connection_stats& operator/=(const float& rhs){
    bytes_sent = bytes_sent / rhs;
    bytes_received = bytes_received / rhs;
    messages_sent = messages_sent / rhs;
    messages_received = messages_received / rhs;
    return *this;
  }

  // connection_stats& operator/=(const connection_stats& rhs){
  //   bytes_sent += rhs.bytes_sent;
  //   bytes_received += rhs.bytes_received;
  //   messages_sent += rhs.messages_sent;
  //   messages_received += rhs.messages_received;
  // }

  // connection_stats operator+ (const connection_stats& first) const {
  //   connection_stats result;
  //   result.bytes_sent = bytes_sent + first.bytes_sent;
  //   result.bytes_received = bytes_received + first.bytes_received;
  //   result.messages_sent = messages_sent + first.messages_sent;
  //   result.messages_received = messages_received + first.messages_received;
  //   return result;
  // }
};

class message_handler {
public:
  /* header length,  body length, message type, message id */
  virtual message_rx *new_rx_message(message_connection *tcp_conn,
      uint64_t header_len, uint64_t body_len, uint64_t msg_type,
      uint64_t msg_id) = 0;
  virtual void aborted_receive(message_connection *tcp_conn,
      message_rx *req) = 0;
  virtual void completed_receive(message_connection *tcp_conn,
      message_rx *req) = 0;

  virtual void aborted_transmit(message_connection *tcp_conn,
      message_tx *req) = 0;
  virtual void completed_transmit(message_connection *tcp_conn,
      message_tx *req) = 0;

  // time synchronization
  void synchronize(int64_t local_delta, int64_t remote_delta) {
    local_delta_tracker.insert(local_delta);
    remote_delta_tracker.insert(remote_delta);
    local_delta_ = local_delta_tracker.get_min();
    remote_delta_ = remote_delta_tracker.get_min();
  }

  int64_t estimate_clock_delta() {
    if (remote_delta_ == INT64_MAX) return 0;
    return (local_delta_ - remote_delta_) / 2;
  }

  int64_t estimate_rtt() {
    if (remote_delta_ == INT64_MAX) return 0;
    return (local_delta_ + remote_delta_);
  }

  int64_t local_delta_ = INT64_MAX;
  int64_t remote_delta_ = INT64_MAX;
  SlidingWindowT<int64_t> local_delta_tracker = SlidingWindowT<int64_t>(1024);
  SlidingWindowT<int64_t> remote_delta_tracker = SlidingWindowT<int64_t>(1024);

};


const size_t max_header_len = 10*1024*1024;


class message_sender {
public:
  message_sender(message_connection *conn, message_handler &handler);
  void send_message(message_tx &req);

private:
  void start_send(message_tx &req);
  void try_send();
  void send_next_message();

  void handle_prehdr_sent(const asio::error_code& error,
      size_t bytes_transferred);
  void handle_hdr_sent(const asio::error_code& error,
      size_t bytes_transferred);
  void handle_body_seg_sent(const asio::error_code& error,
      size_t bytes_transferred);
  void next_body_seg();
  void abort_connection(const char *msg);
  void abort_connection(std::string msg) {
    abort_connection(msg.c_str());
  }


  asio::ip::tcp::socket &socket_;

  char header_buf[max_header_len];
  /* header length,  body length, message type, message id, timestamp, clock_delta */
  uint64_t pre_header[6];

  message_connection *conn_;
  message_tx *req_;
  message_handler &handler_;
  size_t body_left;
  size_t body_seg_sent_;

  std::mutex queue_mutex;
  tbb::concurrent_queue<message_tx*> tx_queue_;
};


class message_receiver {
public:
  message_receiver(message_connection *conn, message_handler &handler);
  void start();

private:
  void abort_connection(const char *msg);
  void abort_connection(std::string msg) {
    abort_connection(msg.c_str());
  }
  /* begin reading a new message */
  void read_new_message();
  /* common pre header received */
  void handle_pre_read(const asio::error_code& error,
      size_t bytes_transferred);
  /* header received */
  void handle_header_read(const asio::error_code& error,
      size_t bytes_transferred);
  /* body segment received */
  void handle_body_seg_read(const asio::error_code& error,
      size_t bytes_transferred);
  /* initiate rx for next body segment or finish message */
  void next_body_seg();


  asio::ip::tcp::socket &socket_;

  char header_buf[max_header_len];
  /* header length,  body length, message type, message id, timestamp, clock_delta */
  uint64_t pre_header[6];

  message_connection *conn_;
  message_handler &handler_;
  message_rx *req_;
  size_t body_left;

  uint64_t rx_begin_;
};


class message_connection {
public:
  message_connection(asio::io_service& io_service, message_handler &handler);
  /* establish outgoing connection */
  void connect(const std::string& server, const std::string& service);
  /* connection on socket established externally (e.g. through acceptor) */
  void established();
  asio::ip::tcp::socket &get_socket();
  void close(const char* reason = nullptr);

protected:
  virtual void ready();
  virtual void closed();

private:
  void handle_resolved(const asio::error_code& err,
      asio::ip::tcp::resolver::iterator endpoint_iterator);
  void handle_established(const asio::error_code& err);
  void abort_connection(const char *msg);
  void abort_connection(std::string msg) {
    abort_connection(msg.c_str());
  }

  std::atomic_flag is_closed;
  message_receiver msg_rx_;
  asio::ip::tcp::resolver resolver_;
  asio::ip::tcp::socket socket_;
public:
  asio::io_service& io_service_;

  connection_stats stats;
};


}
}

#endif