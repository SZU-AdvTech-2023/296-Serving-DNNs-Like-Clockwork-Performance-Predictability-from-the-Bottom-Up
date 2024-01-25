#include "config.h"
#include "clockwork/cuda_common.h"
#include <exception>
#include <libconfig.h++>


using namespace clockwork;

libconfig::Config default_config;
libconfig::Config user_config;

template<typename T> T lookup(std::string setting_name){
	T value = default_config.lookup(setting_name);
	bool success = user_config.lookupValue(setting_name, value);
	if (success)
		std::cout << "The default setting " << setting_name << " was overwritten to " << value << std::endl;
	return value;
}

void check_user_config() {

	const libconfig::Setting& root = user_config.getRoot();

	std::string settings [] = {"telemetry_settings", "memory_settings", "log_dir", "allow_zero_size_inputs"};

	std::string variables [] = {"enable_task_telemetry","enable_action_telemetry", "telemetry_log_dir",
			"weights_cache_size", "weights_cache_page_size", "io_pool_size", "workspace_pool_size", "host_io_pool_size"};
	try {
		const libconfig::Setting& worker_config = root.lookup("WorkerConfig");

		for(int i = 0; i < worker_config.getLength(); i++) {
			const libconfig::Setting& setting = worker_config[i];
			std::string name = setting.getName();
			auto* it = std::find(std::begin(settings), std::end(settings), name);
			if (it == std::end(settings))
				throw "The WorkerConfig should have settings named \"telemetry_settings\" or \"memory_settings\" or \"log_dir\"";
			for (int k = 0; k < setting.getLength(); k++) {
				const libconfig::Setting& var = setting[k];
				std::string name = var.getName();
				auto* it = std::find(std::begin(variables), std::end(variables), name);
				if (it == std::end(variables))
					throw "The allowed settings are provided in the default config file";
			}
		}
	}
	catch (const char* msg) {
		std::cerr << msg << std::endl;
		throw "The provided config file is malformed please use the default config file as a guidline";
	}
	catch (std::exception& e) {
		std::cerr << "User config file is malformed, the root name should be \"WorkerConfig\"" << std::endl;
		throw e;
	}

}

ClockworkWorkerConfig::ClockworkWorkerConfig(std::string config_file_path) {
	std::string default_config_file = util::get_clockwork_directory() + "/config/default.cfg";
	std::string user_config_file;

	std::cout << "Loading Clockwork worker default config from " << default_config_file << std::endl;

	if(config_file_path != "")
		user_config_file = config_file_path;
	else if(const char* env_config = std::getenv("CLOCKWORK_CONFIG_FILE"))
		user_config_file = env_config;

	if(user_config_file != "") {
		std::cout << "Overriding default config using values in " << user_config_file << std::endl;
		user_config.readFile(user_config_file.c_str());
		check_user_config();
	}


	default_config.readFile(default_config_file.c_str());

	try {
		task_telemetry_logging_enabled = lookup<bool>("WorkerConfig.telemetry_settings.enable_task_telemetry");
		action_telemetry_logging_enabled = lookup<bool>("WorkerConfig.telemetry_settings.enable_action_telemetry");
	} catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		std::cout << "Config file should contain the setting \"telemetry_settings\"" <<
				"with boolean variables \"enable_task_telemetry\" and \"enable_action_telemetry\"" << std::endl;
	}

	try{
		weights_cache_size = lookup<long long>("WorkerConfig.memory_settings.weights_cache_size");
		weights_cache_page_size = lookup<long long>("WorkerConfig.memory_settings.weights_cache_page_size");
		io_pool_size = lookup<long long>("WorkerConfig.memory_settings.io_pool_size");
		workspace_pool_size = lookup<long long>("WorkerConfig.memory_settings.workspace_pool_size");
		host_io_pool_size = lookup<long long>("WorkerConfig.memory_settings.host_io_pool_size");
	} catch(const std::exception& e) {
		std::cerr << e.what() << std::endl;
		std::cout << "Config file should contain the setting \"memory_settings\" " <<
				"with variables \"weights_cache_size\", \"weights_cache_page_size\", " <<
				"\"io_pool_size\", \"workspace_pool_size\", \"host_io_pool_size\"" << std::endl;
		throw e;
	}

	try {
		telemetry_log_dir = lookup<std::string>("WorkerConfig.log_dir.telemetry_log_dir");
	} catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		std::cout << "Config file should contain the setting \"log_dir\" with variable \"telemetry_log_dir\"" << std::endl;
	}

	try {
		allow_zero_size_inputs = lookup<bool>("WorkerConfig.allow_zero_size_inputs");
	} catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		std::cout << "Config file should contain the setting \"allow_zero_size_inputs\"" << std::endl;
	}
}

