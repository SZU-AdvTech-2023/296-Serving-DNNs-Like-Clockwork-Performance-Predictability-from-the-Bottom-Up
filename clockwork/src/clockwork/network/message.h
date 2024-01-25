#ifndef _CLOCKWORK_NETWORK_MESSAGE_H_
#define _CLOCKWORK_NETWORK_MESSAGE_H_

#include <clockwork.pb.h>
#include <dmlc/logging.h>

namespace clockwork {
namespace network {

class message_tx {
public:
  virtual ~message_tx() {}
  virtual uint64_t get_tx_msg_type() const = 0;
  virtual uint64_t get_tx_req_id() const = 0;
  virtual uint64_t get_tx_header_len() const = 0;
  virtual uint64_t get_tx_body_len() const = 0;
  virtual void serialize_tx_header(void *dest) = 0;
  virtual void tx_complete() = 0;
  virtual std::pair<const void *,size_t> next_tx_body_buf() = 0;
};


class message_rx {
public:
  virtual ~message_rx() {}
  virtual uint64_t get_msg_id() const = 0;
  virtual void header_received(const void *hdr, size_t hdr_len) = 0;
  virtual std::pair<void *,size_t> next_body_rx_buf() = 0;
  virtual void body_buf_received(size_t len) = 0;
  virtual void rx_complete() = 0;
};

template <uint64_t TMsgType, class TMsg, class TReq>
class msg_protobuf_tx : public message_tx {
protected:
  uint64_t req_id_;

public:
  TMsg msg;
  static const uint64_t MsgType = TMsgType;

  void set_msg_id(uint64_t msg_id) { req_id_ = msg_id; }

  virtual void set(TReq &request) = 0;

  virtual uint64_t get_tx_msg_type() const { return TMsgType; }
  virtual uint64_t get_tx_req_id() const { return req_id_; }
  virtual uint64_t get_tx_header_len() const { return msg.ByteSize(); }

  virtual void serialize_tx_header(void *dest) {
    msg.SerializeWithCachedSizesToArray(reinterpret_cast<google::protobuf::uint8 *>(dest));
  }

  virtual void tx_complete() {}

  /* default to no body */
  virtual uint64_t get_tx_body_len() const { return 0; }

  virtual std::pair<const void *,size_t> next_tx_body_buf() {
    throw "Should not be called";
  }
};

template <uint64_t TMsgType, class TMsg, class TRsp>
class msg_protobuf_rx : public message_rx {
protected:
  uint64_t req_id_;

public:
  TMsg msg;
  static const uint64_t MsgType = TMsgType;

  void set_msg_id(uint64_t msg_id) { req_id_ = msg_id; }

  virtual uint64_t get_msg_id() const { return req_id_; }

  virtual void header_received(const void *hdr, size_t hdr_len) {
    if (!msg.ParseFromArray(hdr, hdr_len)) throw "parsing failed";
  }

  virtual std::pair<void *,size_t> next_body_rx_buf() {
    throw "Should not be called";
  }

  virtual void body_buf_received(size_t len) {
    throw "Should not be called";
  }

  virtual void get(TRsp &response) = 0;

  virtual void rx_complete() {}

};

template <uint64_t TMsgType, class TMsg, class TRsp>
class msg_protobuf_tx_with_body : public msg_protobuf_tx<TMsgType, TMsg, TRsp> {
protected:
  size_t body_len_ = 0;
  void* body_ = nullptr;

public:

  virtual void set_body_len(size_t body_len) { body_len_ = body_len; }

  virtual uint64_t get_tx_body_len() const { return body_len_; }

  virtual std::pair<const void *,size_t> next_tx_body_buf() {
    return std::make_pair(body_, body_len_);
  }

};

template <uint64_t TMsgType, class TMsg, class TRsp>
class msg_protobuf_rx_with_body : public msg_protobuf_rx<TMsgType, TMsg, TRsp> {
protected:
  size_t body_len_ = 0;
  void* body_ = nullptr;

public:

  virtual void set_body_len(size_t body_len) {
    body_len_ = body_len;
    body_ = new uint8_t[body_len];
  }

  virtual std::pair<void *,size_t> next_body_rx_buf() {
    return std::make_pair(body_, body_len_);
  }

  virtual void body_buf_received(size_t len) {
    CHECK(len == body_len_) << "Message length did not match expected payload size";
  }

};

}
}

#endif