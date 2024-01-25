#include "clockwork/api/client_api.h"
#include "clockwork/util.h"
#include <sstream>

namespace clockwork {
namespace clientapi {

uint64_t initial_offset = util::now();

std::string millis(uint64_t t) {
	std::stringstream ss;
	ss << (t / 1000000) << "." << ((t % 1000000) / 100000); // Crude way of printing as ms
	return ss.str();	
}

std::string offset(uint64_t t) {
	if (t < initial_offset) t = initial_offset;
	return millis(t - initial_offset);
}

std::string window(uint64_t earliest, uint64_t latest) {
	uint64_t now = util::now();
	earliest = earliest < now ? 0 : (earliest - now);
	latest = latest < now ? 0 : (latest - now);
	std::stringstream ss;
	ss << "[" << millis(earliest) << ", " << millis(latest) << "]";
	return ss.str();		
}

std::string UploadModelRequest::str() {
	std::stringstream ss;
	ss << "Req" << header.user_request_id << ":UploadModel"
	   << " weights=" << weights_size
	   << " batches=[";
	for (unsigned i = 0; i < instances.size(); i++) {
		if (i > 0) {
			ss << ",";
		}
		ss << instances[i].batch_size;
	}
	ss << "]";
	return ss.str();
}

std::string UploadModelResponse::str() {
	std::stringstream ss;
	ss << "Rsp" << header.user_request_id << ":UploadModel";
	if (header.status != clockworkSuccess) {
		ss << " error " << header.status << ": " << header.message;
	} else {
		ss << " model_id=" << model_id << " input=" << input_size << " output=" << output_size;
	}
	return ss.str();
}

std::string InferenceRequest::str() {
	std::stringstream ss;
	ss << "Req" << header.user_request_id << ":Infer"
	   << " model=" << model_id
	   << " b=" << batch_size
	   << " input=" << input_size;
	return ss.str();
}

std::string InferenceResponse::str() {
	std::stringstream ss;
	ss << "Rsp" << header.user_request_id << ":Infer";
	if (header.status != clockworkSuccess) {
		ss << " error " << header.status << ": " << header.message;
	} else {
		ss << " model_id=" << model_id << " b=" << batch_size << " output=" << output_size;
	}
	return ss.str();
}

std::string EvictRequest::str() {
	std::stringstream ss;
	ss << "Req" << header.user_request_id << ":Evict"
	   << " model=" << model_id;
	return ss.str();
}

std::string EvictResponse::str() {
	std::stringstream ss;
	ss << "Rsp" << header.user_request_id << ":Evict";
	if (header.status != clockworkSuccess) {
		ss << " error " << header.status << ": " << header.message;
	} else {
		ss << " success";
	}
	return ss.str();
}

std::string LoadModelFromRemoteDiskRequest::str() {
	std::stringstream ss;
	ss << "Req" << header.user_request_id << ":LoadModel"
	   << " path=" << remote_path;
	return ss.str();
}

std::string LoadModelFromRemoteDiskResponse::str() {
	std::stringstream ss;
	ss << "Rsp" << header.user_request_id << ":LoadModel";
	if (header.status != clockworkSuccess) {
		ss << " error " << header.status << ": " << header.message;
	} else {
		ss << " model_id=[" << model_id << "->" << (model_id + copies_created - 1) << "] input=" << input_size << " output=" << output_size;
	}
	return ss.str();
}

std::string LSRequest::str() {
	std::stringstream ss;
	ss << "Req" << header.user_request_id << ":LS";
	return ss.str();	
}

std::string ClientModelInfo::str() {
	std::stringstream ss;
	ss << "M-" << model_id
	   << " src=" << remote_path
	   << " input=" << input_size
	   << " output=" << output_size;
	return ss.str();
}

std::string LSResponse::str() {
	std::stringstream ss;
	ss << "Rsp" << header.user_request_id << ":LS";
	if (header.status != clockworkSuccess) {
		ss << " error " << header.status << ": " << header.message;
	} else {
		ss << " " << models.size() << " models:";
		for (auto &model : models) {
			ss << std::endl << " " << model.str();
		}
	}
	return ss.str();	
}

}
}
