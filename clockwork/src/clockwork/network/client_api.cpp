#include "clockwork/network/client_api.h"
#include <iostream>

namespace clockwork {
namespace network {

void set_header(RequestHeader &request_header, RequestHeaderProto* proto) {
	proto->set_user_id(request_header.user_id);
	proto->set_user_request_id(request_header.user_request_id);
}

void set_header(ResponseHeader &response_header, ResponseHeaderProto* proto) {
	proto->set_user_request_id(response_header.user_request_id);
	proto->set_status(response_header.status);
	proto->set_message(response_header.message);
}

void get_header(RequestHeader &request_header, const RequestHeaderProto &proto) {
	request_header.user_id = proto.user_id();
	request_header.user_request_id = proto.user_request_id();
}

void get_header(ResponseHeader &response_header, const ResponseHeaderProto &proto) {
	response_header.user_request_id = proto.user_request_id();
	response_header.status = proto.status();
	response_header.message = proto.message();
}

void msg_inference_req_tx::set(clientapi::InferenceRequest &request) {
  	set_header(request.header, msg.mutable_header());
  	msg.set_model_id(request.model_id);
  	msg.set_batch_size(request.batch_size);
    msg.set_slo_factor(request.slo_factor);
  	body_len_ = request.input_size;
  	body_ = request.input;
}

void msg_inference_req_rx::get(clientapi::InferenceRequest &request) {
	get_header(request.header, msg.header());
	request.model_id = msg.model_id();
	request.batch_size = msg.batch_size();
  request.slo_factor = msg.slo_factor();
	request.input_size = body_len_;
	request.input = body_;
}

void msg_inference_rsp_tx::set(clientapi::InferenceResponse &response) {
  	set_header(response.header, msg.mutable_header());
  	msg.set_model_id(response.model_id);
  	msg.set_batch_size(response.batch_size);
  	body_len_ = response.output_size;
  	body_ = response.output;
}

void msg_inference_rsp_rx::get(clientapi::InferenceResponse &response) {
    get_header(response.header, msg.header());
    response.model_id = msg.model_id();
    response.batch_size = msg.batch_size();
    response.output_size = body_len_;
    response.output = body_;
}

void msg_evict_req_tx::set(clientapi::EvictRequest &request) {
    set_header(request.header, msg.mutable_header());
    msg.set_model_id(request.model_id);
}

void msg_evict_req_rx::get(clientapi::EvictRequest &request) {
    get_header(request.header, msg.header());
    request.model_id = msg.model_id();
}

void msg_evict_rsp_tx::set(clientapi::EvictResponse &request) {
    set_header(request.header, msg.mutable_header());
}

void msg_evict_rsp_rx::get(clientapi::EvictResponse &request) {
    get_header(request.header, msg.header());
}

void msg_ls_req_tx::set(clientapi::LSRequest &request) {
    set_header(request.header, msg.mutable_header());
}

void msg_ls_req_rx::get(clientapi::LSRequest &request) {
    get_header(request.header, msg.header());
}

void set_client_model_info(const clientapi::ClientModelInfo &model, ClientModelInfoProto* proto) {
    proto->set_model_id(model.model_id);
    proto->set_remote_path(model.remote_path);
    proto->set_input_size(model.input_size);
    proto->set_output_size(model.output_size);
}

void msg_ls_rsp_tx::set(clientapi::LSResponse &request) {
    set_header(request.header, msg.mutable_header());
    for (auto &model : request.models) {
      ClientModelInfoProto* modelproto = msg.add_models();
      set_client_model_info(model, modelproto);
    }
}

void get_client_model_info(clientapi::ClientModelInfo &model, const ClientModelInfoProto &proto) {
    model.model_id = proto.model_id();
    model.remote_path = proto.remote_path();
    model.input_size = proto.input_size();
    model.output_size = proto.output_size();
}

void msg_ls_rsp_rx::get(clientapi::LSResponse &request) {
    get_header(request.header, msg.header());
    for (unsigned i = 0; i < msg.models_size(); i++) {
      clientapi::ClientModelInfo model;
      get_client_model_info(model, msg.models(i));
      request.models.push_back(model);
    }
}

void msg_load_remote_model_req_tx::set(clientapi::LoadModelFromRemoteDiskRequest &request) {
    set_header(request.header, msg.mutable_header());
    msg.set_remote_path(request.remote_path);
    msg.set_no_of_copies(request.no_of_copies);
}

void msg_load_remote_model_req_rx::get(clientapi::LoadModelFromRemoteDiskRequest &request) {
    get_header(request.header, msg.header());
    request.remote_path = msg.remote_path();
    request.no_of_copies = msg.no_of_copies();
}

void msg_load_remote_model_rsp_tx::set(clientapi::LoadModelFromRemoteDiskResponse &response) {
    set_header(response.header, msg.mutable_header());
    msg.set_model_id(response.model_id);
    msg.set_input_size(response.input_size);
    msg.set_output_size(response.output_size);
    msg.set_copies_created(response.copies_created);
}

void msg_load_remote_model_rsp_rx::get(clientapi::LoadModelFromRemoteDiskResponse &response) {
	get_header(response.header, msg.header());
	response.model_id = msg.model_id();
	response.input_size = msg.input_size();
	response.output_size = msg.output_size();
	response.copies_created = msg.copies_created();
}

void msg_upload_model_req_tx::set(clientapi::UploadModelRequest &request) {
	set_header(request.header, msg.mutable_header());
	msg.set_params_len(request.weights_size);

	// For now assume one instance with a batch size of 1
	msg.set_so_len(request.instances[0].so_size);
	msg.set_clockwork_len(request.instances[0].spec_size);
}

uint64_t msg_upload_model_req_tx::get_tx_body_len() const {
	return blob_so.size() + blob_cw.size() + blob_cwparams.size();
}

std::pair<const void *,size_t> msg_upload_model_req_tx::next_tx_body_buf() {
	switch (body_tx_state) {
	  case BODY_SEND_SO: {
	    body_tx_state = BODY_SEND_CLOCKWORK;
	    return std::make_pair(blob_so.data(), blob_so.size());
	  }
	  case BODY_SEND_CLOCKWORK: {
	    body_tx_state = BODY_SEND_CWPARAMS;
	    return std::make_pair(blob_cw.data(), blob_cw.size());        
	  }
	  case BODY_SEND_CWPARAMS: {
	    body_tx_state = BODY_SEND_DONE;
	    return std::make_pair(blob_cwparams.data(), blob_cwparams.size());
	  }
	  default: {
	    CHECK(false) << "upload_model_req in invalid state";
	  }
	}
}

void msg_upload_model_req_rx::get(clientapi::UploadModelRequest &request) {
	get_header(request.header, msg.header());
	request.weights_size = msg.params_len();
	request.weights = buf_cwparams;

	clientapi::UploadModelRequest::ModelInstance instance;
	instance.batch_size = 1;
	instance.so_size = msg.so_len();
	instance.so = buf_so;
	instance.spec_size = msg.clockwork_len();
	instance.spec = buf_clockwork;
}

void msg_upload_model_req_rx::set_body_len(size_t body_len) {
	body_len_ = body_len;
}

void msg_upload_model_req_rx::header_received(const void *hdr, size_t hdr_len) {
    msg_protobuf_rx::header_received(hdr, hdr_len);

    CHECK(msg.so_len() + msg.clockwork_len() + msg.params_len() == body_len_) 
      << "load_model body length did mot match expected length";

    buf_so = new uint8_t[msg.so_len()];
    buf_clockwork = new uint8_t[msg.clockwork_len()];
    buf_cwparams = new uint8_t[msg.params_len()];
}

std::pair<void *,size_t> msg_upload_model_req_rx::next_body_rx_buf() {
    switch (body_rx_state) {
      case BODY_RX_SO: {
        body_rx_state = BODY_RX_CLOCKWORK;
        return std::make_pair(buf_so, msg.so_len());        
      };
      case BODY_RX_CLOCKWORK: {
        body_rx_state = BODY_RX_CWPARAMS;
        return std::make_pair(buf_clockwork, msg.clockwork_len());
      };
      case BODY_RX_CWPARAMS: {
        body_rx_state = BODY_RX_DONE;
        return std::make_pair(buf_cwparams, msg.params_len());
      };
      default: CHECK(false) << "upload_model_req in invalid state";
    }
}

void msg_upload_model_req_rx::body_buf_received(size_t len) {
    size_t expected;

    if (body_rx_state == BODY_RX_CLOCKWORK) {
      expected = msg.so_len();
    } else if (body_rx_state == BODY_RX_CWPARAMS) {
      expected = msg.clockwork_len();
    } else if (body_rx_state == BODY_RX_DONE) {
      expected = msg.params_len();
    } else {
      throw "TODO";
    }

    if (expected != len)
      throw "unexpected body rx len";
}

void msg_upload_model_rsp_tx::set(clientapi::UploadModelResponse &response) {
    set_header(response.header, msg.mutable_header());
    msg.set_model_id(response.model_id);
    msg.set_input_size(response.input_size);
    msg.set_output_size(response.output_size);
}

void msg_upload_model_rsp_rx::get(clientapi::UploadModelResponse &response) {
	get_header(response.header, msg.header());
	response.model_id = msg.model_id();
	response.input_size = msg.input_size();
	response.output_size = msg.output_size();
}

}
}
