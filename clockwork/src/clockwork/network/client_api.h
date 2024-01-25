#ifndef _CLOCKWORK_NETWORK_CLIENT_API_H_
#define _CLOCKWORK_NETWORK_CLIENT_API_H_

#include "clockwork/api/client_api.h"
#include "clockwork/network/message.h"

namespace clockwork {
namespace network {

void set_header(RequestHeader &request_header, RequestHeaderProto* proto);
void set_header(ResponseHeader &response_header, ResponseHeaderProto* proto);
void get_header(RequestHeader &request_header, const RequestHeaderProto &proto);
void get_header(ResponseHeader &response_header, const ResponseHeaderProto &proto);

class msg_inference_req_tx : public msg_protobuf_tx_with_body<REQ_INFERENCE, ModelInferenceReqProto, clientapi::InferenceRequest> {
public:
  virtual void set(clientapi::InferenceRequest &request);
};

class msg_inference_req_rx : public msg_protobuf_rx_with_body<REQ_INFERENCE, ModelInferenceReqProto, clientapi::InferenceRequest> {
public:
  virtual void get(clientapi::InferenceRequest &request);
};

class msg_inference_rsp_tx : public msg_protobuf_tx_with_body<RSP_INFERENCE, ModelInferenceRspProto, clientapi::InferenceResponse> {
public:
  ~msg_inference_rsp_tx() {
    if (body_ != nullptr) free(body_);
  }
  void set(clientapi::InferenceResponse &response);
};

class msg_inference_rsp_rx : public msg_protobuf_rx_with_body<RSP_INFERENCE, ModelInferenceRspProto, clientapi::InferenceResponse> {
public:
  void get(clientapi::InferenceResponse &response);
};

class msg_evict_req_tx : public msg_protobuf_tx<REQ_EVICT, EvictReqProto, clientapi::EvictRequest> {
public:
  void set(clientapi::EvictRequest &request);
};

class msg_evict_req_rx : public msg_protobuf_rx<REQ_EVICT, EvictReqProto, clientapi::EvictRequest> {
public:
  void get(clientapi::EvictRequest &request);
};

class msg_evict_rsp_tx : public msg_protobuf_tx<RSP_EVICT, EvictRspProto, clientapi::EvictResponse> {
public:
  void set(clientapi::EvictResponse &request);
};

class msg_evict_rsp_rx : public msg_protobuf_rx<RSP_EVICT, EvictRspProto, clientapi::EvictResponse> {
public:
  void get(clientapi::EvictResponse &request);
};

class msg_load_remote_model_req_tx : public msg_protobuf_tx<REQ_LOAD_REMOTE_MODEL, LoadModelFromDiskReqProto, clientapi::LoadModelFromRemoteDiskRequest> {
public:  
  void set(clientapi::LoadModelFromRemoteDiskRequest &request);
};

class msg_load_remote_model_req_rx : public msg_protobuf_rx<REQ_LOAD_REMOTE_MODEL, LoadModelFromDiskReqProto, clientapi::LoadModelFromRemoteDiskRequest> {
public:
  void get(clientapi::LoadModelFromRemoteDiskRequest &request);
};

class msg_load_remote_model_rsp_tx : public msg_protobuf_tx<RSP_LOAD_REMOTE_MODEL, LoadModelFromDiskRspProto, clientapi::LoadModelFromRemoteDiskResponse> {
public:
  void set(clientapi::LoadModelFromRemoteDiskResponse &response);
};

class msg_load_remote_model_rsp_rx : public msg_protobuf_rx<RSP_LOAD_REMOTE_MODEL, LoadModelFromDiskRspProto, clientapi::LoadModelFromRemoteDiskResponse> {
public:
  void get(clientapi::LoadModelFromRemoteDiskResponse &response);
};

class msg_ls_req_tx : public msg_protobuf_tx<REQ_LS, LSReqProto, clientapi::LSRequest> {
public:
  void set(clientapi::LSRequest &request);
};

class msg_ls_req_rx : public msg_protobuf_rx<REQ_LS, LSReqProto, clientapi::LSRequest> {
public:
  void get(clientapi::LSRequest &request);
};

class msg_ls_rsp_tx : public msg_protobuf_tx<RSP_LS, LSRspProto, clientapi::LSResponse> {
public:
  void set(clientapi::LSResponse &request);
};

class msg_ls_rsp_rx : public msg_protobuf_rx<RSP_LS, LSRspProto, clientapi::LSResponse> {
public:
  void get(clientapi::LSResponse &request);
};


class msg_upload_model_req_tx : public msg_protobuf_tx<REQ_UPLOAD_MODEL, ModelUploadReqProto, clientapi::UploadModelRequest> {
private:
  enum body_tx_state {
    BODY_SEND_SO,
    BODY_SEND_CLOCKWORK,
    BODY_SEND_CWPARAMS,
    BODY_SEND_DONE,
  } body_tx_state = BODY_SEND_SO;

  std::string blob_so, blob_cw, blob_cwparams;
public:
  void set(clientapi::UploadModelRequest &request);

  virtual uint64_t get_tx_body_len() const;

  virtual std::pair<const void *,size_t> next_tx_body_buf();
};

class msg_upload_model_req_rx : public msg_protobuf_rx<REQ_UPLOAD_MODEL, ModelUploadReqProto, clientapi::UploadModelRequest> {
private:
  enum body_rx_state {
    BODY_RX_SO,
    BODY_RX_CLOCKWORK,
    BODY_RX_CWPARAMS,
    BODY_RX_DONE,
  } body_rx_state = BODY_RX_SO;

  size_t body_len_ = 0;
  void* buf_so = nullptr;
  void* buf_clockwork = nullptr;
  void* buf_cwparams = nullptr;

public:
  void get(clientapi::UploadModelRequest &request);

  void set_body_len(size_t body_len);

  virtual void header_received(const void *hdr, size_t hdr_len);

  virtual std::pair<void *,size_t> next_body_rx_buf();

  virtual void body_buf_received(size_t len);
};

class msg_upload_model_rsp_tx : public msg_protobuf_tx<RSP_UPLOAD_MODEL, ModelUploadRspProto, clientapi::UploadModelResponse> {
public:
  void set(clientapi::UploadModelResponse &response);
};

class msg_upload_model_rsp_rx : public msg_protobuf_rx<RSP_UPLOAD_MODEL, ModelUploadRspProto, clientapi::UploadModelResponse> {
public:
  void get(clientapi::UploadModelResponse &response);
};

}
}

#endif