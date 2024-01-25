#include "clockwork/util.h"
#include "clockwork/client.h"
#include "clockwork/workload/workload.h"
#include "clockwork/workload/example.h"
#include <string>
#include <iostream>
#include "clockwork/workload/azure.h"
#include "clockwork/workload/slo.h"
#include "clockwork/thread.h"

using namespace clockwork;

std::pair<std::string, std::string> split(std::string addr) {
	auto split = addr.find(":");
	std::string hostname = addr.substr(0, split);
	std::string port = addr.substr(split+1, addr.size());
	return {hostname, port};
}

void printUsage() {
	std::cerr << "Usage: simpleclient [address] [model] [input_filename] " << std::endl;
}

int main(int argc, char *argv[])
{
	if (argc < 4) {
		printUsage();
		return 1;
	}
	auto address = split(std::string(argv[1]));
	std::string modelname = argv[2];
	std::string inputfile_name = argv[3];

	std::cout << "Inferring " << modelname
	          << " on " << address.first 
	          << ":" << address.second << std::endl;
	std::cout << "Reading input from " << inputfile_name << std::endl;


	bool verbose = true; // Log every request and response?
	bool summary = true;  // Log summary once per second?

	clockwork::Client *client = clockwork::Connect(
		address.first, address.second, 
		verbose, summary);

	// Get the available models
	auto modelzoo = util::get_clockwork_modelzoo();
	auto it = modelzoo.find(modelname);
	if (it == modelzoo.end()) {
		std::cout << "Unknown model " << modelname << std::endl;
		std::cout << "Valid models: " << std::endl;
		for (auto i = modelzoo.begin(); i != modelzoo.end(); i++) {
			std::cout << i->first << " ";
		}
		std::cout << std::endl;
		return 2;
	}
	auto modelpath = it->second;

    // Load the model on Clockwork
	auto model = client->load_remote_model(modelpath);

    // Read in the input data if provided
    std::string inputfile_data;
    util::readFileAsString(inputfile_name, inputfile_data);

    // Make sure provided input has correct size
    if (inputfile_data.size() != model->input_size()) {
        std::cout << "Error with provided input " << inputfile_name
                  << ", expected size " << model->input_size()
                  << " but got " << inputfile_data.size() << std::endl;
        return 3;
    }

    // Create inference input
    std::vector<uint8_t> input(inputfile_data.size());
    std::memcpy(input.data(), inputfile_data.data(), inputfile_data.size());

    while (true) {
	    try {
	    	std::vector<uint8_t> output = model->infer(input);

	    	// Just for fun, print top-5 from output; this kinda assumes image classification
	    	if (output.size() < 20) break;

	    	// Convert to floats
            float* outputf = static_cast<float*>(static_cast<void*>(output.data()));
            unsigned output_size = model->output_size()/4;
 
            // Order
            std::priority_queue<std::pair<float, unsigned>> q;
            for (unsigned i = 0; i < output_size; i++) {
                q.push(std::pair<float, unsigned>(outputf[i], i));
            }
            int topn = 5;
            for (unsigned i = 0; i < topn; i++) {
                std::cout << "output[" << q.top().second << "] = " << q.top().first << std::endl;
                q.pop();
            }

	    	break;
	    } catch (const clockwork_initializing& e) {
	    	// Wait for controller to finish initializing then retry
	    	usleep(1000000);
	    }
	}
}
