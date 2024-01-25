
#include "clockwork-convert/tvm_model.h"
#include <tvm/runtime/device_api.h>

using namespace tvm_model;


void NodeEntry::Load(dmlc::JSONReader *reader) {
	reader->BeginArray();
	CHECK(reader->NextArrayItem()) << "invalid json format";
	reader->Read(&node_id);
	CHECK(reader->NextArrayItem()) << "invalid json format";
	reader->Read(&index);
	if (reader->NextArrayItem()) {
		reader->Read(&version);
		CHECK(!reader->NextArrayItem()) << "invalid json format";
	} else {
		version = 0;
	}
}

void Node::LoadAttrs(dmlc::JSONReader *reader, TVMOpParam* param) {
  int bitmask = 0;
  std::string key, value;
  reader->BeginObject();
  while (reader->NextObjectItem(&key)) {
    reader->Read(&value);
    if (key == "func_name") {
      param->func_name = value;
      bitmask |= 1;
    } else if (key == "num_inputs") {
      param->num_inputs = strtoul(value.c_str(), nullptr, 10);
      bitmask |= 2;
    } else if (key == "num_outputs") {
      param->num_outputs = strtoul(value.c_str(), nullptr, 10);
      bitmask |= 4;
    } else if (key == "flatten_data") {
      param->flatten_data = strtoul(value.c_str(), nullptr, 10);
      bitmask |= 8;
    }
  }
  CHECK_EQ(bitmask, 1|2|4|8) << "invalid format";
}

void Node::Load(dmlc::JSONReader *reader) {
  reader->BeginObject();
  int bitmask = 0;
  std::string key;
  while (reader->NextObjectItem(&key)) {
    if (key == "op") {
      reader->Read(&op_type);
      bitmask |= 1;
    } else if (key == "name") {
      reader->Read(&name);
      bitmask |= 2;
    } else if (key == "inputs") {
      reader->Read(&inputs);
      bitmask |= 4;
    } else if (key == "attr" || key == "attrs") {
      this->LoadAttrs(reader, &param);
    } else if (key == "control_deps") {
      reader->Read(&control_deps);
    } else {
      LOG(FATAL) << "do not support key " << key;
    }
  }
  CHECK_EQ(bitmask, 1|2|4) << "invalid format";
}

void GraphAttr::Load(dmlc::JSONReader *reader) {
  reader->BeginObject();
  int bitmask = 0;
  std::string key, type;
  while (reader->NextObjectItem(&key)) {
    if (key == "dltype") {
      reader->BeginArray();
      CHECK(reader->NextArrayItem());
      reader->Read(&type);
      CHECK_EQ(type, "list_str");
      CHECK(reader->NextArrayItem());
      reader->Read(&dltype);
      CHECK(!reader->NextArrayItem());
      bitmask |= 1;
    } else if (key == "storage_id") {
      reader->BeginArray();
      CHECK(reader->NextArrayItem());
      reader->Read(&type);
      CHECK_EQ(type, "list_int");
      CHECK(reader->NextArrayItem());
      reader->Read(&storage_id);
      CHECK(!reader->NextArrayItem());
      bitmask |= 2;
    } else if (key == "shape") {
      reader->BeginArray();
      CHECK(reader->NextArrayItem());
      reader->Read(&type);
      CHECK_EQ(type, "list_shape");
      CHECK(reader->NextArrayItem());
      reader->Read(&shape);
      CHECK(!reader->NextArrayItem());
      bitmask |= 4;
    } else if (key == "device_index") {
      reader->BeginArray();
      CHECK(reader->NextArrayItem());
      reader->Read(&type);
      CHECK_EQ(type, "list_int");
      CHECK(reader->NextArrayItem());
      reader->Read(&device_index);
      CHECK(!reader->NextArrayItem());
    } else {
      reader->BeginArray();
      CHECK(reader->NextArrayItem());
      reader->Read(&type);
      if (type == "list_int") {
        CHECK(reader->NextArrayItem());
        std::vector<int> temp;
        reader->Read(&temp);
      } else if (type == "size_t") {
        CHECK(reader->NextArrayItem());
        size_t temp;
        reader->Read(&temp);
      } else {
        LOG(FATAL) << "cannot skip graph attr " << key;
      }
      CHECK(!reader->NextArrayItem());
    }
  }
  CHECK_EQ(bitmask, 1|2|4) << "invalid format";
}

void Model::Load(dmlc::JSONReader *reader) {
  reader->BeginObject();
  int bitmask = 0;
  std::string key;
  while (reader->NextObjectItem(&key)) {
    if (key == "nodes") {
      reader->Read(&nodes_);
      bitmask |= 1;
    } else if (key == "arg_nodes") {
      reader->Read(&input_nodes_);
      bitmask |= 2;
    } else if (key == "node_row_ptr") {
      reader->Read(&node_row_ptr_);
      bitmask |= 4;
    } else if (key == "heads") {
      reader->Read(&outputs_);
      bitmask |= 8;
    } else if (key == "attrs") {
      reader->Read(&attrs_);
      bitmask |= 16;
    } else {
      LOG(FATAL) << "key " << key << " is not supported";
    }
  }
  CHECK_EQ(bitmask, 1|2|4|8|16) << "invalid format";
}

Model Model::LoadFromFile(std::string filename) {
	std::ifstream is(filename);
	dmlc::JSONReader reader(&is);
	Model model;
	model.Load(&reader);
	return model;
}

void Params::Load(dmlc::Stream* strm) {
	uint64_t header, reserved;
	CHECK(strm->Read(&header)) << "Invalid parameters file format";
	CHECK(header == kTVMNDArrayListMagic) << "Invalid parameters file format";
	CHECK(strm->Read(&reserved)) << "Invalid parameters file format";

	std::vector<std::string> names;
	CHECK(strm->Read(&names)) << "Invalid parameters file format";

	uint64_t sz;
	strm->Read(&sz);
	size_t size = static_cast<size_t>(sz);
	CHECK(size == names.size()) << "Invalid parameters file format";

	for (std::string name : names) {
    CHECK(data.find(name) == data.end()) << "Duplicate params for " << name;
		data[name] = new tvm::runtime::NDArray();
		data[name]->Load(strm);
	}
}

Params Params::LoadFromFile(std::string filename) {
	std::string paramsData;
	clockwork::util::readFileAsString(filename, paramsData);
	dmlc::MemoryStringStream strm(const_cast<std::string*>(&paramsData));
	Params params;
	params.Load(&strm);
	return params;
}

Allocs Allocs::ProfileModel(std::string model_so, std::string model_json, std::string model_params) {
	const int dtype_code = kDLFloat;
	const int dtype_bits = 32;
	const int dtype_lanes = 1;
	const int device_type = kDLGPU;
	const int device_id = 0;

	const tvm::runtime::PackedFunc load_module(*tvm::runtime::Registry::Get("runtime.module.loadfile_so"));
	tvm::runtime::Module mod_syslib = load_module(model_so, "so");

	// Graph structure
	std::ifstream json_in(model_json, std::ios::in);  // read as text
	std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
	json_in.close();

	// Construct TVM runtime
  const tvm::runtime::PackedFunc create_graph_runtime(*tvm::runtime::Registry::Get("tvm.graph_runtime.create"));
	tvm::runtime::Module mod = create_graph_runtime(json_data, mod_syslib, device_type, device_id);
	// const tvm::runtime::PackedFunc create_graph_runtime(*tvm::runtime::Registry::Get("tvm.decoupled_graph_runtime.create_contiguous"));
	// tvm::runtime::Module mod = create_graph_runtime(json_data, mod_syslib, device_type, device_id);
	
	// tvm::runtime::Module mod = ClockworkGraphRuntimeCreate(json_data, mod_syslib, device_type, device_id);


    // Read params from file
    std::ifstream params_in(model_params, std::ios::binary);  // read as binary
    std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
    params_in.close();

    TVMByteArray params_arr;
    params_arr.data = params_data.c_str();
    params_arr.size = params_data.length();
    tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
    load_params(params_arr);

	  // // Pull out params blob
	  // tvm::runtime::PackedFunc get_const_params = mod.GetFunction("get_const_params");
	  // tvm::runtime::PackedFunc set_const_params = mod.GetFunction("set_const_params");
	  // tvm::runtime::NDArray const_params = get_const_params();


    tvm::runtime::PackedFunc extract_allocs = mod.GetFunction("profile_workspace_allocs");
    std::vector<std::vector<tvm::runtime::WorkspaceAlloc>>* alloc_vector = static_cast<std::vector<std::vector<tvm::runtime::WorkspaceAlloc>>*>((void*) extract_allocs());

    Allocs allocs;
    for (unsigned i = 0; i < alloc_vector->size(); i++) {
    	OpAlloc op;

    	for (unsigned j = 0; j < (*alloc_vector)[i].size(); j++) {
    		tvm::runtime::WorkspaceAlloc alloc = (*alloc_vector)[i][j];
    		if (alloc.isalloc) {
    			op.allocs.push_back(alloc.size);
    		}
    	}

    	allocs.ops.push_back(op);
    }

    return allocs;
}