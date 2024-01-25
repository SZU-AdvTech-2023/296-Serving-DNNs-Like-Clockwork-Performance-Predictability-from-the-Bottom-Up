#ifndef _CLOCKWORK_TVM_MODEL_H_
#define _CLOCKWORK_TVM_MODEL_H_

#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <dlpack/dlpack.h>
#include <dmlc/memory_io.h>
#include <dmlc/json.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/serializer.h>
#include <clockwork/util.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>


namespace tvm_model {

/**
The origin of this code is TVM's graph_runtime.h and graph_runtime.cc.  This logic deserializes
the TVM Graph for a model and constructs simple structs for the JSON format.
**/

/*! \brief Magic number for NDArray list file  */
constexpr uint64_t kTVMNDArrayListMagic = 0xF7E58D4F05049CB7;

/*! \brief operator attributes about tvm op */
struct TVMOpParam {
  std::string func_name;
  uint32_t num_inputs;
  uint32_t num_outputs;
  uint32_t flatten_data;
};

// Node entry
struct NodeEntry {
	uint32_t node_id;
	uint32_t index;
	uint32_t version;

	// JSON Loader
	void Load(dmlc::JSONReader *reader);
};

// Node
struct Node {
	// operator type in string
	std::string op_type;
	// name of the op
	std::string name;
	// parameters
	TVMOpParam param;
	// inputs
	std::vector<NodeEntry> inputs;
	// control deps
	std::vector<uint32_t> control_deps;
	// JSON Loader
	void LoadAttrs(dmlc::JSONReader *reader, TVMOpParam* param);
	// JSON Loader
	void Load(dmlc::JSONReader *reader);
};

struct GraphAttr {
	size_t storage_num_not_alloctaed{0};
	std::vector<int> storage_id;
	std::vector<int> device_index;
	std::vector<std::string> dltype;
	std::vector<std::vector<int64_t> > shape;
	// The graph attribute fields.
	void Load(dmlc::JSONReader *reader);
};

struct Model {

  /*! \brief The graph nodes. */
  std::vector<Node> nodes_;
  /*! \brief The argument nodes. */
  std::vector<uint32_t> input_nodes_;
  /*! \brief Used for quick entry indexing. */
  std::vector<uint32_t> node_row_ptr_;
  /*! \brief Output entries. */
  std::vector<NodeEntry> outputs_;
  /*! \brief Additional graph attributes. */
  GraphAttr attrs_;

  // The graph attribute fields.
  void Load(dmlc::JSONReader *reader);

  static Model LoadFromFile(std::string filename);
};

struct Params {
	std::unordered_map<std::string, tvm::runtime::NDArray*> data;

	void Load(dmlc::Stream* strm);

	static Params LoadFromFile(std::string filename);
};

struct OpAlloc {
	std::vector<size_t> allocs;
};

struct Allocs {
	std::vector<OpAlloc> ops;

	static Allocs ProfileModel(std::string model_so, std::string model_json, std::string model_params);
};

}

#endif
