#ifndef _CLOCKWORK_TVM_ABSTRACT_MODEL_H_
#define _CLOCKWORK_TVM_ABSTRACT_MODEL_H_

#include <vector>
#include <string>
#include "clockwork-convert/tvm_model.h"
#include <dmlc/logging.h>
#include "clockwork/modeldef.h"

#include <dlpack/dlpack.h>

/**
This code is an abstract representation of TVM model with direct pointers between
nodes, inputs, outputs, and storage.  It's a little easier to work with than TVM's
because TVM has additional lookup logic.
**/

namespace clockwork_model {

class StorageLocation;

class Tensor {
public:
	int id;
	DLDataType dltype;
	std::vector<int64_t> shape;
	StorageLocation* storage;

	const size_t Size();
};

class StorageLocation {
public:
	int id;
	std::vector<Tensor*> used_by;

	const size_t Size();
};

class Operation {
public:
	int id;
	std::string op_name;
	std::string func_name;
	std::vector<Tensor*> inputs;
	std::vector<size_t> allocs;
	std::vector<Tensor*> outputs;
};

class LayerWeights {
public:
	std::string name;
	void* data;
	int size;
	Tensor* tensor;
};

/** Essentially the same as a layerweightstensor, but logically distinct */
class Input {
public:
	std::string name;
	Tensor* tensor;
};

class Output {
public:
	int output_ix;
	Tensor* tensor;
};

class Model {
public:
	std::vector<StorageLocation*> storage_locations;
	std::vector<Operation*> operations;
	std::unordered_map<std::string, Input*> inputs;
	std::vector<Output*> outputs;
	std::unordered_map<std::string, LayerWeights*> weights;

	static Model fromTVM(
			tvm_model::Model &model, 
			tvm_model::Params &params, 
			tvm_model::Allocs &workspaceAllocs);
};


class Page {
public:
	std::vector<StorageLocation*> used_by;

	size_t Size();
};

class PreMappedIndices {
public:
	unsigned page_index;
	unsigned location_index;
};

class PageMappedStorage {
public:
	std::vector<Page*> weights;
	std::vector<StorageLocation*> io_storage;
	std::vector<StorageLocation*> workspace_storage;
	size_t transient_memory;

	std::unordered_map<std::string, std::vector<PreMappedIndices>> weights_lookup;

	static PageMappedStorage* calculate(Model &model, size_t weights_page_size, PageMappedStorage* existing_weights_mapping = nullptr);
};

extern void makeModelDef(Model &model, size_t weights_page_size, clockwork::model::PageMappedModelDef &output, char* &weights, int &weightsSize, PageMappedStorage* &mapped, PageMappedStorage* existing_weights_mapping = nullptr);


/* Invariants to check:
- null nodes have no inputs or attrs, only one weights, and the location is not used by others
- only null or tvm_op types
- don't support versions yet but can support multiple outputs
- don't support flatten data
*/

}

#endif
