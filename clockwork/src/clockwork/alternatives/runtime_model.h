#ifndef _CLOCKWORK_ALTERNATIVES_RUNTIME_MODEL_H_
#define _CLOCKWORK_ALTERNATIVES_RUNTIME_MODEL_H_

#include <atomic>
#include <deque>
#include <unordered_map>
#include "clockwork/model/model.h"
#include "clockwork/cache.h"

namespace clockwork {
namespace alternatives {


/* The runtime model hooks into a pagecache and handles page evictions.

Calls to the runtime model aren't thread-safe and callers should call
lock and unlock surrounding any other API calls.  Models can be unlocked
from different threads to where they are locked.  While a model is locked,
its pages will not be evicted (unless explicitly done so by calling evict())
*/
class RuntimeModel {

private:

	std::atomic_flag in_use;	// Only one request can execute at a time for a model

	PageCache* cache;
	model::Model* model;

	bool instantiated_on_device = false;

	std::shared_ptr<Allocation> weights_pages = nullptr;
	std::shared_ptr<Allocation> workspace_pages = nullptr;

public:

	RuntimeModel(PageCache* cache, model::Model* model);

	bool try_lock();
	void lock();
	void unlock();

	/* true if code is already on GPU, false otherwise */
	bool has_code();

	/* sets up the model code on the GPU */
	void instantiate_code();

	/* removes model code from the GPU */
	void uninstantiate_code();

	/* true if weights are already on GPU, false otherwise */
	bool has_weights();

	/* evicts weights from the GPU and frees the associated pages */
	bool evict_weights();

	/* allocate pages on GPU and transfer weights */
	void transfer_weights(cudaStream_t stream);

	/* input size expected by this model */
	unsigned input_size();

	/* send input to GPU */
	void set_input(void* input, cudaStream_t stream);

	/* call model (input is already set with set_input) */
	void call(cudaStream_t stream);

	/* output size to expect from this model */
	unsigned output_size();

	/* get output from GPU (can only be called once per invocation) */
	void get_output(void* output, cudaStream_t stream);
	
};

}
}

#endif