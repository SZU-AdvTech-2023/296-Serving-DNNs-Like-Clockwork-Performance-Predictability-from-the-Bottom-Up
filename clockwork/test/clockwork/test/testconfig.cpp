#include <climits>
#include <stdio.h>
#include <unistd.h>
#include <libgen.h>
#include <fstream>
#include <algorithm>
#include <exception>

#include <cuda_runtime.h>
#include "clockwork/api/worker_api.h"
#include "clockwork/test/util.h"
#include "clockwork/worker.h"
#include "clockwork/config.h"
#include <catch2/catch.hpp>
#include "clockwork/test/actions.h"
#include "clockwork/test/controller.h"
#include "tbb/concurrent_queue.h"


using namespace clockwork;

std::string default_path = util::get_clockwork_dir() + "/config/default.cfg";
std::string temp_path = util::get_clockwork_dir() + "/config/temp.cfg";
std::string  default_config() {
	std::string s = "WorkerConfig: \n"
	"{\n"
	"memory_settings:\n"
	"{\n"
		"weights_cache_size = 30064771072L;\n"
		"weights_cache_page_size = 16777216L;\n"
		"io_pool_size = 134217728L;\n"
		"workspace_pool_size = 536870912L;\n"
		"host_io_pool_size = 226492416L;\n"
	"};\n"
	"telemetry_settings:\n"
	"{\n"
		"enable_task_telemetry = false;\n"
		"enable_action_telemetry = false;\n"
	"};\n"
	"log_dir:\n"
	"{\n"
		"telemetry_log_dir = \"./\";\n"
	"};\n"
	"};\n";
	return s;
}

void replace_default_config(std::string content) {

	rename(default_path.c_str(), temp_path.c_str());

	if (content == "")
		return;

	std::ofstream myfile;
	myfile.open (default_path);
	myfile << content;
	myfile.close();

}


void restore_default_config() {

	remove(default_path.c_str());
	rename(temp_path.c_str(), default_path.c_str());
}

TEST_CASE("Missing default config file", "[config]") {

	int exceptions = 0;
	replace_default_config("");

	try {
		ClockworkWorker worker = ClockworkWorker();
	} catch (const std::exception& e) {
		exceptions ++;
	}

	REQUIRE(exceptions == 1);

	restore_default_config();
}


TEST_CASE("Default config file is missing a default value", "[config]") {
	
	int exceptions = 0;

	
	std::string missing_config = "WorkerConfig = { memory_settings =  { weights_cache_size = 12345;}} \n";

	replace_default_config(missing_config);

	try {
		ClockworkWorker worker = ClockworkWorker();
	} catch (const std::exception& e) {
		exceptions ++;
	}

	REQUIRE(exceptions == 1);

	restore_default_config();
}


TEST_CASE("Malformed user config file", "[config]") {

	int exceptions = 0;
	std::string config = default_config();

	replace_default_config(config);

	std::string user_config = "Config: \n"
	"{\n"
	"memory_settings:\n"
	"{\n"
		"weights_cache_size = 12345L;\n"
	"};\n"
	"};\n";
	
	std::ofstream myfile;
	myfile.open ("./temp.cfg");
	myfile << user_config;
	myfile.close();


	ClockworkWorkerConfig* worker_config; 

	try {
		worker_config = new ClockworkWorkerConfig("./temp.cfg");
		ClockworkWorker worker = ClockworkWorker(*worker_config);
	} catch (...) {
		exceptions ++;
	}	

	REQUIRE(exceptions == 1);
	if(exceptions == 0)
		delete worker_config;

	restore_default_config();

}


TEST_CASE("Overriding a default value", "[config]") {

	int exceptions = 0;
	std::string config = default_config();

	replace_default_config(config);

	std::string user_config = "WorkerConfig: \n"
	"{\n"
	"memory_settings:\n"
	"{\n"
		"weights_cache_size = 12345L;\n"
	"};\n"
	"};\n";
	
	std::ofstream myfile;
	myfile.open ("./temp.cfg");
	myfile << user_config;
	myfile.close();

	ClockworkWorkerConfig* worker_config; 
	ClockworkWorker* worker;
	try {
		worker_config = new ClockworkWorkerConfig("./temp.cfg");
		worker = new ClockworkWorker(*worker_config);
	} catch (const std::exception& e) {
		exceptions ++;
	}

	REQUIRE(exceptions == 0);
	REQUIRE(worker_config->weights_cache_size == 12345L);

	delete worker_config;

	restore_default_config();
}
