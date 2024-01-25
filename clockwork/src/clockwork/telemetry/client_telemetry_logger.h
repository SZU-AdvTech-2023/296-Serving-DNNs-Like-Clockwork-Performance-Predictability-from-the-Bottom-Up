#ifndef _CLOCKWORK_TELEMETRY_CLIENT_TELEMETRY_LOGGER_H_
#define _CLOCKWORK_TELEMETRY_CLIENT_TELEMETRY_LOGGER_H_

#include <thread>
#include <atomic>
#include "tbb/concurrent_queue.h"
#include <unistd.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include "clockwork/util.h"
#include "clockwork/telemetry.h"
#include <dmlc/logging.h>
#include <pods/pods.h>
#include <pods/binary.h>
#include <pods/buffers.h>
#include <pods/streams.h>
#include <tbb/concurrent_queue.h>
#include <iomanip>
#include "clockwork/thread.h"


namespace clockwork {

class ClientTelemetryLogger {
public:
	std::atomic_uint64_t outstanding = 0;
	std::atomic_uint64_t submitted = 0;
	std::atomic_uint64_t completed = 0;
	virtual void log(int user_id, int model_id, int batch_size, 
		size_t input_size, size_t output_size,
		uint64_t request_sent, uint64_t response_received,
		bool success) = 0;
	void incrOutstanding() { outstanding++; submitted++; }
	void decrOutstanding() { outstanding--; completed++; }
	virtual void shutdown(bool awaitCompletion) = 0;
};

class NoOpClientTelemetryLogger : public ClientTelemetryLogger {
public:
	virtual void log(int user_id, int model_id, int batch_size, 
		size_t input_size, size_t output_size,
		uint64_t request_sent, uint64_t response_received,
		bool success) {};
	virtual void shutdown(bool awaitCompletion) {};
};

struct Sample {
	int user_id;
	int model_id;
	int batch_size;
	size_t input_size;
	size_t output_size;
	uint64_t request_sent;
	uint64_t response_received;
	bool success;
};

struct Summary {
	int count;
	double sum;
	uint64_t min;
	uint64_t max;
	double throughput;
	uint64_t duration;

	Summary(uint64_t duration, std::vector<Sample> samples): duration(duration) {
		if (samples.size() == 0) {
			count = 0;
			sum = 0;
			min = 0;
			max = 0;
			throughput = 0;
			duration = duration;
		} else {
			count = samples.size();
			sum = 0;
			min = UINT64_MAX;
			max = 0;
			duration = duration;
			for (auto &sample : samples) {
				uint64_t latency = sample.response_received - sample.request_sent;
				sum += latency;
				min = std::min(min, latency);
				max = std::max(max, latency);
			}
			throughput = count * 1000000000.0 / static_cast<double>(duration);
		}
	}

	std::string str() {
		std::stringstream s;
		s << std::fixed << std::setprecision(2);
		s << "throughput=" << throughput;
		s << " min=" << (min/1000000.0) << " max=" << (max/1000000.0) << " mean=" << (sum/(1000000.0*count));
		return s.str();
	}
};

class ClientTelemetrySummarizer : public ClientTelemetryLogger {
public:
	uint64_t print_interval;
	std::atomic_bool alive = true;
	tbb::concurrent_queue<Sample> samples;
	std::thread thread;
	std::atomic_int errors = 0;

	ClientTelemetrySummarizer(uint64_t print_interval = 10000000000UL) : 
		thread(&ClientTelemetrySummarizer::run, this), print_interval(print_interval) {
		threading::initLoggerThread(thread);
	}


	void run() {
		uint64_t last_print = util::now();
		bool begun = false;

		std::queue<Sample> newSamples;
		while (alive) {
			Sample sample;
			while (samples.try_pop(sample)) {
				newSamples.push(sample);
				begun = true;
			}

			uint64_t now = util::now();
			if (last_print + print_interval > now) {
				usleep(10000);
				continue;
			}

			std::vector<Sample> all_samples;
			// std::map<unsigned, std::vector<Sample>> samples_by_user;
			while (!newSamples.empty()) {
				sample = newSamples.front();
				// samples_by_user[sample.user_id].push_back(sample);
				all_samples.push_back(sample);
				newSamples.pop();
			}

			std::stringstream report;
			report << "total=" << submitted.exchange(0) << " ";
			if (begun && all_samples.size() == 0) {
				report << "throughput=0" << std::endl;
			} else if (begun) {
				report << Summary(now - last_print,all_samples).str() 
				       << std::endl;
			}
			std::cout << report.str();

			// std::stringstream report;
			// if (begun && samples_by_user.size() == 0) {
			// 	report << "throughput=0" << std::endl;
			// } else {
			// 	for (auto &p : samples_by_user) {
			// 		report << "User " << p.first << " " 
			// 		       << Summary(now - last_print, p.second).str() 
			// 		       << std::endl;
			// 	}
			// }
			// std::cout << report.str();

			last_print = now;
		}
	}

	virtual void log(int user_id, int model_id, int batch_size, 
		size_t input_size, size_t output_size,
		uint64_t request_sent, uint64_t response_received,
		bool success)
	{
		if (success) {
			samples.push(Sample{user_id, model_id, batch_size, 
				input_size, output_size, request_sent, 
				response_received, success});
		} else {
			errors++;
		}
	}

	virtual void shutdown(bool awaitCompletion)
	{
		alive = false;
		if (awaitCompletion) {
			thread.join();
		}
	}

};

}

#endif
