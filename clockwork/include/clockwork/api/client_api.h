#ifndef _CLOCKWORK_API_CLIENT_API_H_
#define _CLOCKWORK_API_CLIENT_API_H_

#include <vector>
#include <functional>
#include <string>
#include "clockwork/api/api_common.h"

/**
This is the semi-public API between Clockwork front-end server and the Clockwork client library.

Clockwork clients should use the frontdoor API defined in client.h rather than this API, as this API has
more internal metadata than the frontdoor API.

For the non-Clockwork runtimes (Threadpool runtime and greedy runtime), workers directly implement
this API, rather than the API defined in worker_api.h.
*/

namespace clockwork {

namespace clientapi {

/* This is only currently used during setup/tear down phase of Clockwork.  Currently this
receives precompiled TVM models; in future we will handle compilation server-side and 
instead take ONNX files as input. */
struct UploadModelRequest {
	RequestHeader header;

	/* Weights are shared across batch sizes */
	size_t weights_size;
	void* weights;

	struct ModelInstance {
		/* Each batch size has different code and spec */
		int batch_size;
		size_t so_size;
		void* so;
		size_t spec_size;
		void* spec;
	};
	
	/* Code and params for different batch sizes */
	std::vector<ModelInstance> instances;

	std::string str();
};

/* This is only currently used during setup/tear down phase of Clockwork */
struct UploadModelResponse {
	ResponseHeader header;
	int model_id;
	size_t input_size;
	size_t output_size;

	std::string str();
};

/* Make an inference request for a model.  Clients can send in arbitrary
batches of inputs.  For now, clockwork does not break down batches into 
multiple smaller batches, but might combine multiple requests into one
single batch. */
struct InferenceRequest {
	RequestHeader header;
	int model_id;
	int batch_size;
	size_t input_size;
	void* input;
	uint64_t deadline = 0;
	float slo_factor;

	// Not sent over the network; used by controller
	uint64_t arrival = 0;

	std::string str();
};

/* The response to a specific inference request. */
struct InferenceResponse {
	ResponseHeader header;
	int model_id;
	int batch_size;
	size_t output_size;
	void* output;

	// Not sent over the network; used by controller
	uint64_t deadline = 0;
	uint64_t departure = 0;
	unsigned arrival_count = 0;
	unsigned departure_count = 0;

	std::string str();
};

/** This is a 'backdoor' API function for ease of experimentation */
struct EvictRequest {
	RequestHeader header;
	int model_id;

	std::string str();
};

/** This is a 'backdoor' API function for ease of experimentation */
struct EvictResponse {
	ResponseHeader header;

	std::string str();
};

struct LSRequest {
	RequestHeader header;

	std::string str();
};

struct ClientModelInfo {
	int model_id;
	std::string remote_path; // For convenience
	size_t input_size;
	size_t output_size;

	std::string str();
};

struct LSResponse {
	ResponseHeader header;
	std::vector<ClientModelInfo> models;

	std::string str();
};

/** This is a 'backdoor' API function for ease of experimentation */
struct LoadModelFromRemoteDiskRequest {
	RequestHeader header;
	std::string remote_path;
	int no_of_copies;

	std::string str();
};

/** This is a 'backdoor' API function for ease of experimentation */
struct LoadModelFromRemoteDiskResponse {
	ResponseHeader header;
	int model_id;
	int copies_created;
	size_t input_size;
	size_t output_size;

	std::string str();
};

class ClientAPI {
public:

	virtual void uploadModel(UploadModelRequest &request, std::function<void(UploadModelResponse&)> callback) = 0;

	virtual void infer(InferenceRequest &request, std::function<void(InferenceResponse&)> callback) = 0;

	/** This is a 'backdoor' API function for ease of experimentation */
	virtual void evict(EvictRequest &request, std::function<void(EvictResponse&)> callback) = 0;

	/** This is a 'backdoor' API function for ease of experimentation */
	virtual void loadRemoteModel(LoadModelFromRemoteDiskRequest &request, std::function<void(LoadModelFromRemoteDiskResponse&)> callback) = 0;

	virtual void ls(LSRequest &request, std::function<void(LSResponse&)> callback) = 0;

};

}
}

#endif
