#ifndef _CLOCKWORK_CONFIG_H_
#define _CLOCKWORK_CONFIG_H_

#include <atomic>
#include <memory>
#include <unordered_map>
#include <deque>
#include <memory>
#include "clockwork/api/worker_api.h"
#include "clockwork/cache.h"
#include "clockwork/model/batched.h"
#include "tbb/concurrent_queue.h"
#include <boost/asio/ip/host_name.hpp>
#include "clockwork/util.h"


using namespace clockwork;

class ClockworkWorkerConfig {

public:
	bool task_telemetry_logging_enabled = false;
	bool action_telemetry_logging_enabled = false;

	std::string task_telemetry_log_file = boost::asio::ip::host_name() + "_task_telemetry.raw";
	std::string action_telemetry_log_file = boost::asio::ip::host_name() + "_action_telemetry.raw";

	std::string telemetry_log_dir;

	unsigned num_gpus = util::get_num_gpus();

	size_t weights_cache_size;
	size_t weights_cache_page_size;
	size_t io_pool_size;
	size_t workspace_pool_size;
	size_t host_io_pool_size;

	bool allow_zero_size_inputs = false;

	ClockworkWorkerConfig(std::string config_file_path = "");

};

#endif
