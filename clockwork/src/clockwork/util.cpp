#include "clockwork/util.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <string.h>
#include <pthread.h>
#include <thread>
#include <atomic>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <istream>
#include "clockwork/cuda_common.h"
#include <cuda_runtime.h>
#include <dmlc/logging.h>
#include <nvml.h>
#include <boost/filesystem.hpp>
#include <sys/stat.h>
#include <libgen.h>
#include <filesystem>
#include <cstdlib>
#include "clockwork/thread.h"
#include <lz4.h>

namespace clockwork {
namespace util {

uint64_t calculate_steady_clock_delta() {
  auto t1 = std::chrono::steady_clock::now();
  auto t2 = std::chrono::system_clock::now();
  uint64_t nanos_t1 = std::chrono::duration_cast<std::chrono::nanoseconds>(t1.time_since_epoch()).count();
  uint64_t nanos_t2 = std::chrono::duration_cast<std::chrono::nanoseconds>(t2.time_since_epoch()).count();
  CHECK(nanos_t2 > nanos_t1) << "Assumptions about steady clock aren't true";
  return nanos_t2 - nanos_t1;
}

uint64_t steady_clock_offset = calculate_steady_clock_delta();

std::uint64_t now() {
  return nanos(hrt());
}

std::string millis(uint64_t t) {
	// Crude way of printing as ms
	std::stringstream ss;
	ss << (t / 1000000) << "." << ((t % 1000000) / 100000);
	return ss.str();
}

clockwork::time_point hrt() {
  return std::chrono::steady_clock::now();
}

clockwork::time_point epoch = hrt();
uint64_t epoch_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count();

std::uint64_t nanos(clockwork::time_point t) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(t - epoch).count() + epoch_time;
}

std::string nowString() {
  std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
  auto duration = now.time_since_epoch();
  typedef std::chrono::duration<int, std::ratio_multiply<std::chrono::hours::period, std::ratio<8>>::type> Days;

  Days days = std::chrono::duration_cast<Days>(duration);                               duration -= days;
  auto hours = std::chrono::duration_cast<std::chrono::hours>(duration);                duration -= hours;
  auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);            duration -= minutes;
  auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);            duration -= seconds;
  auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration);  duration -= milliseconds;
  auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration);  duration -= microseconds;
  auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(duration);

  std::stringstream ss;
  ss << hours.count() << ":" << minutes.count() << ":" << seconds.count() << "." << milliseconds.count() << " " << microseconds.count() << " " << nanoseconds.count();
  return ss.str();
}

unsigned get_num_gpus() {
  nvmlReturn_t status;

  unsigned deviceCount;
  status = nvmlDeviceGetCount(&deviceCount);
  if (status == NVML_ERROR_UNINITIALIZED) {
    status = nvmlInit();
    if (status == NVML_ERROR_DRIVER_NOT_LOADED) {
      return 0;
    }
    CHECK(status == NVML_SUCCESS) << status;
    status = nvmlDeviceGetCount(&deviceCount);
  }
  CHECK(status == NVML_SUCCESS);

  return deviceCount;
}


void setCudaFlags() {
  cudaError_t error = cudaSetDeviceFlags(cudaDeviceScheduleSpin);
  if(error != cudaSuccess) {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}

std::string getGPUmodel(int deviceNumber) {
  int nDevices;
  cudaGetDeviceCount(&nDevices);

  if (deviceNumber < 0 || deviceNumber >= nDevices) {
    std::cout << nDevices << " devices found, invalid deviceNumber " << deviceNumber << std::endl;
    exit(-1);
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, deviceNumber);
  return std::string(prop.name);
}


char* getGPUModelToBuffer(int deviceNumber, char* buf) {
  strcpy(buf, getGPUmodel(deviceNumber).c_str());
  return buf;
}


void readFileAsString(const std::string &filename, std::string &dst) {
  std::ifstream in(filename, std::ios::binary);
  dst = std::string(
      std::istreambuf_iterator<char>(in), 
      std::istreambuf_iterator<char>());
  in.close();
}

struct path_leaf_string
{
    std::string operator()(const boost::filesystem::directory_entry& entry) const
    {
        return entry.path().leaf().string();
    }
};

std::vector<std::string> listdir(std::string directory) {
  std::vector<std::string> filenames;
  boost::filesystem::path p(directory);
  boost::filesystem::directory_iterator start(p);
  boost::filesystem::directory_iterator end;
  std::transform(start, end, std::back_inserter(filenames), path_leaf_string());
  return filenames;
}

bool exists(std::string filename) {
  struct stat buffer;   
  return (stat (filename.c_str(), &buffer) == 0); 
}

long filesize(std::string filename) {
    struct stat buffer;
    int rc = stat(filename.c_str(), &buffer);
    return rc == 0 ? buffer.st_size : -1;
}

thread_local cudaStream_t current_stream;

void initializeCudaStream(unsigned gpu_id, int priority) {
  CUDA_CALL(cudaSetDevice(gpu_id));
  cudaStream_t stream;  
  CUDA_CALL(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, priority));
  SetStream(stream);

  // int least, greatest;
  // CUDA_CALL(cudaDeviceGetStreamPriorityRange(&least, &greatest));
  // std::cout << "Priority range: " << least << " to " << greatest << std::endl;
}

void SetStream(cudaStream_t stream) {
  current_stream = stream;
}

cudaStream_t Stream() {
  return current_stream;
}

std::string get_clockwork_directory()
{
	int bufsize = 1024;
	char buf[bufsize];
	int len = readlink("/proc/self/exe", buf, bufsize);
	return dirname(dirname(buf));
}

std::string get_example_model_path(std::string model_name)
{
  return get_example_model_path(get_clockwork_directory(), model_name);
}


std::string get_example_model_path(std::string clockwork_directory, std::string model_name) {
  return clockwork_directory + "/resources/" + model_name + "/model";
}

std::string get_controller_log_dir() {
  auto logdir = std::getenv("CLOCKWORK_LOG_DIR");
  if (logdir == nullptr) return "/local";
  std::string logdirs = std::string(logdir);
  if (logdirs == "") return "/local";
  return logdirs;
}

std::string get_modelzoo_dir() {
  auto modelzoo = std::getenv("CLOCKWORK_MODEL_DIR");
  if (modelzoo == nullptr) { return ""; }
  return modelzoo == nullptr ? "" : std::string(modelzoo);
}

std::map<std::string, std::string> get_clockwork_modelzoo() {
  std::string modelzoo = get_modelzoo_dir();
  if (modelzoo == "") {
    std::cout << "CLOCKWORK_MODEL_DIR variable not set, exiting" << std::endl;
    exit(1);
  }

  std::map<std::string, std::string> result;
  for (auto &p : std::filesystem::directory_iterator(modelzoo)) {
    if (exists(p.path() / "model.clockwork_params")) {
      result[p.path().filename()] = p.path() / "model";
    }
  }
  std::cout << "Found " << result.size() << " models in " << modelzoo << std::endl;

  return result;
}

InputGenerator::InputGenerator() {
  std::string basedir = get_clockwork_directory() + "/resources/inputs/processed";
  CHECK(exists(basedir)) << "Could not find Clockwork images directory";

  size_t total_size = 0;
  std::vector<std::string> all_inputs;
  for (auto &p : std::filesystem::directory_iterator(basedir)) {
    if (std::filesystem::is_directory(p)) {
      for (auto &f : std::filesystem::directory_iterator(p)) {
        std::string data;
        readFileAsString(f.path().u8string(), data);
        all_inputs.push_back(data);
        total_size += data.size();
        uncompressed_inputs[data.size()].push_back(data);


        const int max_compress_size = LZ4_compressBound(data.size());
        char* compressed = static_cast<char*>(malloc((size_t) max_compress_size));
        const int compressed_size = LZ4_compress_default(data.data(), compressed, data.size(), max_compress_size);
        std::string compressed_data = std::string(compressed, compressed_size);
        compressed_inputs[data.size()].push_back(compressed_data);
      }
    }
  }

  this->all_inputs = new char[total_size];
  size_t offset = 0;
  for (auto &input : all_inputs) {
    std::memcpy(this->all_inputs+offset, input.data(), input.size());
    offset += input.size();
  }

}

void InputGenerator::generateInput(size_t size, char* buf) {
  auto it = uncompressed_inputs.find(size);
  if (it != uncompressed_inputs.end()) {
    auto &inputs = it->second;
    std::uniform_int_distribution<> d(0, inputs.size()-1);
    auto &input = inputs[d(rng)];
    CHECK(input.size() == size);
    std::memcpy(buf, input.data(), input.size());

  } else if (size < this->all_inputs_size) {
    std::uniform_int_distribution<> d(0, all_inputs_size - size - 1);
    size_t start_offset = d(rng);
    std::memcpy(buf, all_inputs+start_offset, size);

  }
}

void InputGenerator::generateInput(size_t size, char** bufPtr) {
  *bufPtr = new char[size];
  generateInput(size, *bufPtr);
}

void InputGenerator::generateCompressedInput(size_t size, char** bufPtr, size_t* compressed_size) {
  auto it = compressed_inputs.find(size);
  if (it != compressed_inputs.end()) {
    auto &inputs = it->second;
    std::uniform_int_distribution<> d(0, inputs.size()-1);
    auto &input = inputs[d(rng)];
    *compressed_size = input.size();
    *bufPtr = new char[input.size()];
    std::memcpy(*bufPtr, input.data(), input.size());
    return;
  }

  CHECK (size < this->all_inputs_size) << size << " unsupported; too large";

  std::uniform_int_distribution<> d(0, all_inputs_size - size - 1);
  size_t start_offset = d(rng);

  const int max_compress_size = LZ4_compressBound(size);
  *bufPtr = static_cast<char*>(malloc((size_t) max_compress_size));
  *compressed_size = LZ4_compress_default(all_inputs+start_offset, *bufPtr, size, max_compress_size);
}

std::string& InputGenerator::getPrecompressedInput(size_t size) {
  auto it = compressed_inputs.find(size);
  CHECK(it != compressed_inputs.end()) << "Generated inputs not available for input size " << size;
  
  auto &inputs = it->second;
  std::uniform_int_distribution<> d(0, inputs.size()-1);
  return inputs[d(rng)];
}

void InputGenerator::generatePrecompressedInput(size_t size, char** bufPtr, size_t* compressed_size) {
  auto it = compressed_inputs.find(size);
  CHECK(it != compressed_inputs.end()) << "Generated inputs not available for input size " << size;
  
  auto &inputs = it->second;
  std::uniform_int_distribution<> d(0, inputs.size()-1);
  auto &input = inputs[d(rng)];
  *compressed_size = input.size();
  *bufPtr = new char[input.size()];
  std::memcpy(*bufPtr, input.data(), input.size());
  return;
}

bool client_inputs_disabled() {
  auto disable_inputs = std::getenv("CLOCKWORK_DISABLE_INPUTS");
  if (disable_inputs == nullptr) { return false; }
  return std::string(disable_inputs) == "1";
}

std::string get_clockwork_model(std::string shortname) {
  auto modelzoo = get_clockwork_modelzoo();
  auto it = modelzoo.find(shortname);
  if (it == modelzoo.end()) {
    std::cerr << "Unknown model " << shortname << std::endl;
    std::cerr << "Value models:" << std::endl;
    for (auto &p : modelzoo) {
      std::cerr << " " << p.second;
    }
    std::cerr << std::endl;
  }
  return it->second;
}

void printCudaVersion() {
  int driverVersion;
  cudaDriverGetVersion(&driverVersion);
  int runtimeVersion;
  cudaRuntimeGetVersion(&runtimeVersion);
  std::cout << "Using CUDA Driver " << driverVersion << " Runtime " << runtimeVersion << std::endl;
}

GPUClockState::GPUClockState(unsigned num_gpus) : clock(num_gpus), thread(&GPUClockState::run, this) {}

void GPUClockState::run() {
  nvmlReturn_t status;

  status = nvmlInit();
  CHECK(status == NVML_SUCCESS || status == NVML_ERROR_ALREADY_INITIALIZED);

  nvmlEventSet_t eventSet;
  CHECK(NVML_SUCCESS == nvmlEventSetCreate(&eventSet));

  nvmlDevice_t devices[clock.size()];
  for (unsigned i = 0; i < clock.size(); i++) {
    CHECK(NVML_SUCCESS == nvmlDeviceGetHandleByIndex(i, &devices[i]));
    CHECK(NVML_SUCCESS == nvmlDeviceRegisterEvents(devices[i], nvmlEventTypeClock, eventSet));
    CHECK(NVML_SUCCESS == nvmlDeviceGetClockInfo(devices[i], NVML_CLOCK_SM, &clock[i]));
  }

  uint64_t notify_every = 5000000000UL;
  uint64_t last_notify = 0;
  while (alive) {
    nvmlEventData_t data;
    status = nvmlEventSetWait(eventSet, &data, 1000);
    if (status == NVML_ERROR_TIMEOUT) continue;
    CHECK(status == NVML_SUCCESS);

    for (unsigned i = 0; i < clock.size(); i++) {
      CHECK(NVML_SUCCESS == nvmlDeviceGetClockInfo(devices[i], NVML_CLOCK_SM, &clock[i]));
    }

    uint64_t now = util::now();
    if (last_notify + notify_every < now) {
      std::stringstream s;
      s << "GPU Clocks Changed:";
      for (unsigned i = 0; i < clock.size(); i++) {
        s << " GPU" << i << "=" << clock[i];
      }
      s << std::endl;
      std::cout << s.str();
      last_notify = now;
    }
    // usleep(100);
  }
}
void GPUClockState::shutdown() {
  alive = false;
}
void GPUClockState::join() {
  thread.join();
}

unsigned GPUClockState::get(unsigned gpu_id) {
  return clock[gpu_id];
}

std::vector<unsigned> make_batch_lookup(std::vector<unsigned> supported_batch_sizes) {
  std::vector<unsigned> lookup;
  lookup.push_back(0);
  for (auto batch_size : supported_batch_sizes) {
      auto prev = lookup[lookup.size()-1];
      while (lookup.size() < (batch_size + 1)) {
          lookup.push_back(prev);
      }

      for (unsigned i = batch_size; i < lookup.size(); i++) {
          lookup[i] = std::max(lookup[i], batch_size);
      }
  }
  return lookup;
}

std::vector<unsigned> make_reverse_batch_lookup(std::vector<unsigned> supported_batch_sizes) {
  std::sort(supported_batch_sizes.begin(), supported_batch_sizes.end());
  std::vector<unsigned> index_lookup;
  for (int i = 0; i < supported_batch_sizes.size(); i++) {
    while (index_lookup.size() <= supported_batch_sizes[i]) {
      index_lookup.push_back(i);
    }
  }
  return index_lookup;
}

}
}
