#include "clockwork-convert/tvm_abstract_model.h"

#include <vector>
#include <algorithm>
#include <string>
#include <cstring>
#include "clockwork-convert/tvm_model.h"
#include <dmlc/logging.h>
#include <tvm/runtime/packed_func.h>

#include <dlpack/dlpack.h>

namespace clockwork_model {


const size_t Tensor::Size() {
	size_t size = 1;
	for (unsigned i = 0; i < shape.size(); i++) {
		size *= static_cast<size_t>(shape[i]);
	}
	size_t bits = dltype.bits * dltype.lanes;
	size_t bytes = ((bits + 7U) / 8U) * size;
	return bytes;
}

const size_t StorageLocation::Size() {
	size_t maxSize = 0;
	for (unsigned i = 0; i < used_by.size(); i++) {
		size_t tensorSize = used_by[i]->Size();
		if (tensorSize > maxSize) {
			maxSize = tensorSize;
		}
	}
	return maxSize;
}

Model Model::fromTVM(tvm_model::Model &model, tvm_model::Params &params, tvm_model::Allocs &allocs) {
	Model out;

	std::unordered_map<int, StorageLocation*> storage_locations;
	for (const int &storage_id : model.attrs_.storage_id) {
		if (storage_locations.find(storage_id) == storage_locations.end()) {
			StorageLocation* storage = new StorageLocation();
			storage->id = storage_id;
			storage_locations[storage_id] = storage;
			out.storage_locations.push_back(storage);
		}
	}

	std::vector<Tensor*> tensors;
	for (unsigned i = 0; i < model.attrs_.storage_id.size(); i++) {
		Tensor* tensor = new Tensor();
		tensor->id = i;
		tensor->dltype = tvm::runtime::String2DLDataType(model.attrs_.dltype[i]);
		tensor->shape = model.attrs_.shape[i];
		tensor->storage = storage_locations[model.attrs_.storage_id[i]];
		tensors.push_back(tensor);
		tensor->storage->used_by.push_back(tensor);
	}

	for (auto &p : params.data) {
		LayerWeights* weights = new LayerWeights();
		weights->name = p.first;
		weights->data = p.second->dataptr();
		weights->size = p.second->Size();
		weights->tensor = nullptr;

		CHECK(out.weights.find(p.first) == out.weights.end()) << "Found duplicate layers with name " << p.first;
		out.weights[p.first] = weights;
	}

	std::vector<std::string> seen_names;

	for (unsigned i = 0; i < model.nodes_.size(); i++) {
		tvm_model::Node &node = model.nodes_[i];

		CHECK(std::find(seen_names.begin(), seen_names.end(), node.name) == seen_names.end()) << "Found duplicate node " << node.name;
		seen_names.push_back(node.name);

		if (node.op_type == "null") {
			int input_index = model.node_row_ptr_[i];
			int input_offset = 0;
			Tensor* tensor = tensors[input_index + input_offset];

			if (out.weights.find(node.name) == out.weights.end()) {
				// If it doesn't have params specified, then it's an input node
				Input* input = new Input();
				input->name = node.name;
				input->tensor = tensor;
				out.inputs[node.name] = input;
			} else {
				// It's a weights node
				out.weights[node.name]->tensor = tensor;
			}
			continue;
		}

		CHECK(node.op_type == "tvm_op") << "Unexpected op type " << node.op_type << " for node " << node.name;

		Operation* op = new Operation();
		op->id = i;
		op->op_name = node.name;
		op->func_name = node.param.func_name;

		CHECK(node.param.flatten_data == 0) << "flatten_data was " << node.param.flatten_data << " but only 0 is currently supported for node " << node.name;

		for (unsigned j = 0; j < node.inputs.size(); j++) {
			int input_node_id = node.inputs[j].node_id;
			int input_index = model.node_row_ptr_[input_node_id];
			int input_offset = node.inputs[j].index;

			CHECK(node.inputs[j].version == 0) << "Encountered version=" << node.inputs[j].version << " for node " << node.name;

			Tensor* input = tensors[input_index + input_offset];
			op->inputs.push_back(input);
		}

		op->allocs = allocs.ops[i].allocs;

		for (unsigned j = 0; j < node.param.num_outputs; j++) {
			int input_index = model.node_row_ptr_[i];
			int input_offset = j;
			Tensor* output = tensors[input_index + input_offset];
			op->outputs.push_back(output);
		}

		out.operations.push_back(op);
	}

	for (unsigned i = 0; i < model.outputs_.size(); i++) {
		int output_node_id = model.outputs_[i].node_id;
		int output_index = model.node_row_ptr_[output_node_id];
		int output_offset = model.outputs_[i].index;

		CHECK(model.outputs_[i].version == 0) << "Encountered version=" << model.outputs_[i].version << " for output " << i;

		Output* output = new Output();
		output->output_ix = i;
		output->tensor = tensors[output_index +output_offset];
		out.outputs.push_back(output);
	}

	return out;
}

size_t Page::Size() {
	size_t size = 0;
	for (StorageLocation* &location : used_by) {
		size += location->Size();
	}
	return size;
}

struct greater_than_storage_location {
	inline bool operator() (StorageLocation* &b1, StorageLocation* &b2) {
		return b1->Size() > b2->Size();
	}
};

struct greater_than_page_size {
	inline bool operator() (Page* &p1, Page* &p2) {
		return p1->Size() > p2->Size();
	}
};

std::vector<Page*> pack(std::vector<StorageLocation*> locations, size_t page_size) {
	// Sort storage locastions in descending order of size
	std::sort(locations.begin(), locations.end(), greater_than_storage_location());

	// Pack each item into page that minimizes remaining space
	std::vector<Page*> pages;
	for (StorageLocation* &location : locations) {
		Page* dst = nullptr;
		for (Page* &page : pages) {
			if (page->Size() + location->Size() <= page_size) {
				dst = page;
				break;
			}
		}

		if (dst == nullptr) {
			dst = new Page();
			pages.push_back(dst);
		}
		dst->used_by.push_back(location);
		std::sort(pages.begin(), pages.end(), greater_than_page_size());
	}

	return pages;
}

void make_weights_lookup_table(Model &model, PageMappedStorage* mapped) {
	unsigned page_number = 0;
	for (Page* page : mapped->weights) {
		unsigned index_in_page = 0;

		for (StorageLocation* location : page->used_by) {

			// Find the LayerWeights that this storage location corresponds to
			LayerWeights* weights = nullptr;
			for (auto &p : model.weights) {
				if (p.second->tensor->storage == location) {
					weights = p.second;
				}
			}

			if (weights == nullptr) {
				throw "Error: page does not correspond to any LayerWeights";
			}

			// Store the actual weights data in the storage lookup table.
			auto data = std::string(static_cast<char*>(weights->data), weights->size);

			mapped->weights_lookup[data].push_back(PreMappedIndices{page_number, index_in_page});

			index_in_page++;
		}

		page_number++;
	}
}

std::vector<Page*> replicate_weights_mapping(Model &model, PageMappedStorage* existing_weights_mapping) {
	std::unordered_map<std::string, unsigned> current_index;

	std::vector<Page*> pages;
	for (unsigned i = 0; i < existing_weights_mapping->weights.size(); i++) {
		Page* page = new Page();
		for (unsigned j = 0; j < existing_weights_mapping->weights[i]->used_by.size(); j++) {
			page->used_by.push_back(nullptr);
		}
		pages.push_back(page);
	}

	for (auto &p : model.weights) {
		LayerWeights* weights = p.second;
		auto data = std::string(static_cast<char*>(weights->data), weights->size);
		CHECK(existing_weights_mapping->weights_lookup.find(data) != existing_weights_mapping->weights_lookup.end())
			<< "Error: " + weights->name + " not found in existing weights mapping";

		unsigned position = current_index[data]++;
		PreMappedIndices indices = existing_weights_mapping->weights_lookup[data][position];
		pages[indices.page_index]->used_by[indices.location_index] = weights->tensor->storage;
	}

	for (unsigned i = 0; i < pages.size(); i++) {
		for (unsigned j = 0; j < pages[i]->used_by.size(); j++) {
			if (pages[i]->used_by[j] == nullptr) {
				std::stringstream err;
				err << "Error: page " << i << "," << j << " is nullptr";
				throw err.str();
			}
		}
	}

	return pages;
}

PageMappedStorage* PageMappedStorage::calculate(Model &model, size_t weights_page_size, PageMappedStorage* existing_weights_mapping) {
	std::vector<StorageLocation*> seen;

	std::vector<StorageLocation*> weights_storage;
	std::vector<StorageLocation*> io_storage;
	std::vector<StorageLocation*> workspace_storage;

	// Pull out storage locations that are model weights
	for (auto &p : model.weights) {
		CHECK(p.second != nullptr);
		CHECK(p.second->tensor != nullptr);
		CHECK(p.second->tensor->storage != nullptr);

		StorageLocation* l = p.second->tensor->storage;

		CHECK(std::find(seen.begin(), seen.end(), l) == seen.end()) << "Found duplicate storage location";

		weights_storage.push_back(l);
		seen.push_back(l);
	}

	// Pull out storage locations that are model inputs
	for (auto &p : model.inputs) {
		CHECK(p.second != nullptr);
		CHECK(p.second->tensor != nullptr);
		CHECK(p.second->tensor->storage != nullptr);

		StorageLocation* l = p.second->tensor->storage;

		CHECK(std::find(seen.begin(), seen.end(), l) == seen.end()) << "Found storage location used for weights and input";

		io_storage.push_back(l);
		seen.push_back(l);
	}

	// Pull out storage locations that are model outputs
	for (auto &output : model.outputs) {
		CHECK(output != nullptr);
		CHECK(output->tensor != nullptr);
		CHECK(output->tensor->storage != nullptr);

		StorageLocation* l = output->tensor->storage;

		CHECK(std::find(seen.begin(), seen.end(), l) == seen.end()) << "Found storage location used for weights and output";

		io_storage.push_back(l);
		seen.push_back(l);
	}

	// The remaining storage locations are intermediate workspace
	for (auto &l : model.storage_locations) {
		if (std::find(seen.begin(), seen.end(), l) == seen.end()) {
			workspace_storage.push_back(l);
			seen.push_back(l);
		}
	}

	// Weights storage locations must fit within the page size
	for (StorageLocation* l : weights_storage) {
		CHECK(l->Size() <= weights_page_size) 
			<< "Weights storage location " << l->id << " has size " << l->Size() << " > weights_page_size=" << weights_page_size;
	}

	// Start constructing return value
	auto mapped = new PageMappedStorage();

	// Pack the weights onto pages, or use existing mapping if provided
	if (existing_weights_mapping == nullptr) {
		mapped->weights = pack(weights_storage, weights_page_size);

		int count = 0;
		for (Page* page : mapped->weights) {
			count += page->used_by.size();
		}

		make_weights_lookup_table(model, mapped);
	} else {
		mapped->weights = replicate_weights_mapping(model, existing_weights_mapping);
	}

	// Calculate the maximum memory needed for transient workspace allocations
	size_t max_transient_memory = 0;
	for (Operation* &op : model.operations) {
		size_t op_transient_memory = 0;
		for (size_t &alloc : op->allocs) op_transient_memory += alloc;
		if (op_transient_memory > max_transient_memory) {
			max_transient_memory = op_transient_memory;
		}
	}

	mapped->io_storage = io_storage;
	mapped->workspace_storage = workspace_storage;
	mapped->transient_memory = max_transient_memory;

	return mapped;
}

void printTensorDef(clockwork::model::PageMappedDLTensorDef def, std::string prefix) {
	std::cout << prefix << def.base_offset << " = [" << def.page << " " << def.page_offset << "] + " << def.size << " shape=[ ";
	for (unsigned i = 0; i < def.shape.size(); i++) {
		std::cout << def.shape[i] << " ";
	}
	std::cout << " ]" << std::endl;
}

void printWorkspaceAlloc(clockwork::model::PageMappedWorkspaceAllocDef def, std::string prefix) {
	std::cout << prefix << "[" << def.page << " " << def.page_offset << "] + " << def.size << std::endl;
}

void printOp(unsigned i, clockwork::model::PageMappedModelDef model, clockwork::model::PageMappedOpDef op, std::string prefix) {
	std::cout << prefix << "Op " << i << " function " << op.so_function;
	std::cout.flush();
	std::cout << " (" << model.so_functions[op.so_function] << "):" << std::endl;
	for (unsigned i = 0; i < op.inputs.size(); i++) {
		printTensorDef(op.inputs[i], prefix+"   ");
	}
	if (op.workspace_allocs.size() > 0) {
		std::cout << prefix << "   " << "Workspace:" << std::endl;
		for (unsigned i = 0; i < op.workspace_allocs.size(); i++) {
			printWorkspaceAlloc(op.workspace_allocs[i], prefix+"    ");
		}
	}
}

void printPageDef(clockwork::model::PageDef def, std::string prefix) {
	std::cout << prefix << "[" << def.base_offset << " +" << def.size << "]" << std::endl;
}

void printNewModel(clockwork::model::PageMappedModelDef model) {
	std::cout << std::endl << "------------------ NEW MODEL ------------------" << std::endl;
	std::cout << model.so_functions.size() << " SO functions" << std::endl;
	std::cout << model.ops.size() << " ops:" << std::endl;
	for (unsigned i = 0; i < model.ops.size(); i++) {
		printOp(i, model, model.ops[i], "  ");
	}
	std::cout << "Inputs:" << std::endl;
	for (unsigned i = 0; i < model.inputs.size(); i++) {
		printTensorDef(model.inputs[i], "   ");
	}
	std::cout << "Outputs:" << std::endl;
	for (unsigned i = 0; i < model.outputs.size(); i++) {
		printTensorDef(model.outputs[i], "   ");
	}
	std::cout << "Weights pages:" << std::endl;
	for (unsigned i = 0; i < model.weights_pages.size(); i++) {
		printPageDef(model.weights_pages[i], "   ");
	}

	std::cout << model.weights_memory_paged << " required in paged-mode" << std::endl;
	std::cout << model.weights_memory << " required memory in non-paged mode" << std::endl;
	std::cout << model.io_memory << " io_memory" << std::endl;
	std::cout << model.workspace_memory << " workspace_memory" << std::endl;
	std::cout << model.weights_pages.size() << " weights pages of size " << model.configured_weights_page_size << std::endl;
}

void makeModelDef(Model &model, size_t weights_page_size, clockwork::model::PageMappedModelDef &output, char* &weights, int &weightsSize, PageMappedStorage* &mapped, PageMappedStorage* existing_weights_mapping) {
	mapped = PageMappedStorage::calculate(model, weights_page_size, existing_weights_mapping);


	// Populate the 'weights_memory' field of the ModelDef
	output.weights_memory = 0;
	for (auto &p : model.weights) {
		output.weights_memory += p.second->tensor->Size();
	}

	output.configured_weights_page_size = weights_page_size;
	output.weights_memory_paged = mapped->weights.size() * weights_page_size;


	// Populate the 'weights_pages' field of the ModelDef
	uint64_t current_offset = 0;
	for (Page* &page : mapped->weights) {
		clockwork::model::PageDef pagedef{current_offset, page->Size()};
		output.weights_pages.push_back(pagedef);
		current_offset += pagedef.size;
	}

	// Populate the 'so_functions' field of the ModelDef
	std::unordered_map<std::string, int> so_functions;
	for (Operation* &operation : model.operations) {
		if (so_functions.find(operation->func_name) == so_functions.end()) {
			so_functions[operation->func_name] = output.so_functions.size();
			output.so_functions.push_back(operation->func_name);
		}
	}
	// TODO: cuda functions not currently done, just extracted from the SO

	struct PagePointer {
		unsigned page;
		uint64_t page_offset;
		uint64_t base_offset;
	};
	std::unordered_map<int, PagePointer> storage_location_pointers;

	// Map the weights to pages
	size_t current_page_offset = 0;
	unsigned current_page = 0;
	current_offset = 0;
	for (Page* page : mapped->weights) {
		current_page_offset = 0;
		for (StorageLocation* &location : page->used_by) {
			CHECK(storage_location_pointers.find(location->id) == storage_location_pointers.end())
				<< "Storage location " << location->id << " assigned to multiple pages";
			storage_location_pointers[location->id] = PagePointer{current_page, current_page_offset, current_offset};
			current_page_offset += location->Size();
			current_offset += location->Size();
		}
		current_page++;
	}

	// The next "page" is all inputs and outputs
	unsigned io_page = current_page;
	current_page_offset = 0;
	for (StorageLocation* location : mapped->io_storage) {
		CHECK(storage_location_pointers.find(location->id) == storage_location_pointers.end())
			<< "IO Storage location " << location->id << " assigned to multiple pages";
		storage_location_pointers[location->id] = PagePointer{current_page, current_page_offset, current_offset};
		current_page_offset += location->Size();
		current_offset += location->Size();
	}
	current_page++;

	// Also save how much io memory is needed
	output.io_memory = current_page_offset;


	// The next "page" is all workspace
	unsigned workspace_page = current_page;
	current_page_offset = 0;
	for (StorageLocation* location : mapped->workspace_storage) {
		CHECK(storage_location_pointers.find(location->id) == storage_location_pointers.end())
			<< "Workspace Storage location " << location->id << " assigned to multiple pages";
		storage_location_pointers[location->id] = PagePointer{current_page, current_page_offset, current_offset};
		current_page_offset += location->Size();
		current_offset += location->Size();
	}

	// Also save how much workspace memory is needed
	size_t transient_memory_begin_offset = current_page_offset;
	output.workspace_memory = current_page_offset + mapped->transient_memory;

	// Now create the op defs
	for (Operation* &operation : model.operations) {
		clockwork::model::PageMappedOpDef opdef;
		opdef.so_function = so_functions[operation->func_name];

		// Both inputs and outputs get passed as arguments to nodes
		for (unsigned i = 0; i < operation->inputs.size() + operation->outputs.size(); i++) {
			Tensor* tensor = i < operation->inputs.size() ? operation->inputs[i] : operation->outputs[i-operation->inputs.size()];
			PagePointer pageptr = storage_location_pointers[tensor->storage->id];

			clockwork::model::PageMappedDLTensorDef tensordef;
			tensordef.base_offset = pageptr.base_offset;
			tensordef.page = pageptr.page;
			tensordef.page_offset = pageptr.page_offset;
			tensordef.size = tensor->Size();
			tensordef.shape = tensor->shape;
			tensordef.code = tensor->dltype.code;
			tensordef.bits = tensor->dltype.bits;
			tensordef.lanes = tensor->dltype.lanes;

			opdef.inputs.push_back(tensordef);
		}

		// Point the transient workspace allocs to the appropriate place
		int current_workspace_offset = 0;
		for (size_t alloc : operation->allocs) {
			clockwork::model::PageMappedWorkspaceAllocDef allocdef;
			allocdef.page = workspace_page;
			allocdef.page_offset = transient_memory_begin_offset + current_workspace_offset;
			allocdef.size = alloc;

			current_workspace_offset += alloc;

			opdef.workspace_allocs.push_back(allocdef);
		}

		output.ops.push_back(opdef);
	}

	// Now save the input locations
	for (auto &p : model.inputs) {
		Tensor* tensor = p.second->tensor;
		PagePointer pageptr = storage_location_pointers[tensor->storage->id];

		clockwork::model::PageMappedDLTensorDef tensordef;
		tensordef.base_offset = pageptr.base_offset;
		tensordef.page = pageptr.page;
		tensordef.page_offset = pageptr.page_offset;
		tensordef.size = tensor->Size();
		tensordef.shape = tensor->shape;

		output.inputs.push_back(tensordef);
	}

	// Now save the output locations
	for (Output* &o : model.outputs) {
		Tensor* tensor = o->tensor;
		PagePointer pageptr = storage_location_pointers[tensor->storage->id];

		clockwork::model::PageMappedDLTensorDef tensordef;
		tensordef.base_offset = pageptr.base_offset;
		tensordef.page = pageptr.page;
		tensordef.page_offset = pageptr.page_offset;
		tensordef.size = tensor->Size();
		tensordef.shape = tensor->shape;

		output.outputs.push_back(tensordef);
	}
	
	printNewModel(output);

	// Now copy the weights
	weightsSize = output.weights_memory;
	weights = static_cast<char*>(malloc(weightsSize));
	for (auto &p : model.weights) {
		void* data = p.second->data;
		size_t size = p.second->size;
		uint64_t offset = storage_location_pointers[p.second->tensor->storage->id].base_offset;
		CHECK(size == p.second->tensor->Size()) << "Mismatched weights sizes " << size << " != " << p.second->tensor->Size();
		std::memcpy(
			weights + offset, // dst
			data, // src
			size // amount
		);
	}
}

}