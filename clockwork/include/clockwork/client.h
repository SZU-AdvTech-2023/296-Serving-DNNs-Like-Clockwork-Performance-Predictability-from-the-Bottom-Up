#ifndef _CLOCKWORK_CLIENT_H_
#define _CLOCKWORK_CLIENT_H_

#include <future>
#include <functional>
#include <string>
#include <stdexcept>

/**
This is the user-facing Clockwork Client
*/

namespace clockwork {

class clockwork_error : public std::runtime_error {
public:
	clockwork_error(std::string what) : std::runtime_error(what) {}
};

class clockwork_initializing : public std::runtime_error {
public:
	clockwork_initializing(std::string what) : std::runtime_error(what) {}
};

/* Represents a model that can be inferred */
class Model {
public:
	virtual int id() = 0;
	virtual std::string source() = 0;
	virtual size_t input_size() = 0;
	virtual size_t output_size() = 0;
	virtual int user_id() = 0;
	virtual void set_user_id(int user_id) = 0;
	virtual void set_slo_factor(float slo_factor) = 0;
	virtual void disable_inputs() = 0;

	/* 
	Perform an inference with the provided input and return the output.
	Blocks until the inference has completed.
	Input can be compressed using lz4 compression; if so, set compressed=true
	Can throw exceptions.
	*/
	virtual std::vector<uint8_t> infer(std::vector<uint8_t> &input, bool compressed=false) = 0;

	/*
	Asynchronous version of infer.
	Performans an inference on the provided input.
	Returns a future that will receive the result of the inference (the output size and the output)
	If an exception occurs, it will be thrown by calls to future.get().
	*/
	virtual std::future<std::vector<uint8_t>> infer_async(std::vector<uint8_t> &input, bool compressed=false) = 0;

	virtual void infer(std::vector<uint8_t> &input, 
		std::function<void(std::vector<uint8_t>&)> onSuccess, 
		std::function<void(int, std::string&)> onError,
		bool compressed=false) = 0;

	/*
	This is a backdoor API call that's useful for testing.
	Instructs the server to evict the weights of this model from the GPU.
	Will throw an exception is the weights aren't loaded.
	*/
	virtual void evict() = 0;
	virtual std::future<void> evict_async() = 0;

};

typedef std::unordered_map<unsigned, Model*> ModelSet;

/* 
Represents a client to Clockwork 
Clockwork can be either local or remote,
and Clockwork can have multiple clients
*/
class Client {
public:

	virtual ~Client(){}

	/*
	Gets an existing Model from Clockwork that can then be inferenced.
	Can throw exceptions including if the model doesn't exist.
	*/
	virtual Model* get_model(int model_id) = 0;
	virtual std::future<Model*> get_model_async(int model_id) = 0;

	/*
	Uploads a model to Clockwork.  Returns a Model for the model
	Can throw exceptions including if the model is invalid
	*/
	virtual Model* upload_model(std::vector<uint8_t> &serialized_model) = 0;
	virtual std::future<Model*> upload_model_async(std::vector<uint8_t> &serialized_model) = 0;


	/*
	Backdoor client function to load a model that resides on the remote machine.
	Not part of the proper API but useful for experimentation
	*/
	virtual Model* load_remote_model(std::string model_path) = 0;
	virtual std::future<Model*> load_remote_model_async(std::string model_path) = 0;
	virtual std::vector<Model*> load_remote_models(std::string model_path, int no_of_copies) = 0;
	virtual std::future<std::vector<Model*>> load_remote_models_async(std::string model_path, int no_of_copies) = 0;


	virtual ModelSet ls() = 0;
	virtual std::future<ModelSet> ls_async() = 0;


};


/* Connect to a Clockwork instance */
extern "C" Client* Connect(const std::string &hostname, const std::string &port, bool verbose = false, bool summary = false);

}

#endif
