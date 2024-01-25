#ifndef _CLOCKWORK_PRIORITY_QUEUE_H_
#define _CLOCKWORK_PRIORITY_QUEUE_H_

#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include "tbb/concurrent_queue.h"
#include <thread>

namespace clockwork {


/* This is a priority queue with the same semantics as time_release_priority_queue
but only when there is a single thread reading.  It uses a thread-safe concurrent queue
and a non-thread-safe queue maintained by the reader */
template <typename T> class single_reader_priority_queue {
private:
	struct container {
		T* element;
		uint64_t priority;
		uint64_t version;

		friend bool operator < (const container& lhs, const container &rhs) {
			return lhs.priority < rhs.priority || 
			  (lhs.priority == rhs.priority && lhs.version < rhs.version);
		}
		friend bool operator > (const container& lhs, const container &rhs) {
			return lhs.priority > rhs.priority ||
			  (lhs.priority == rhs.priority && lhs.version > rhs.version);
		}
	};

	std::atomic_bool alive;
	tbb::concurrent_queue<container> queue;

	uint64_t version;
	std::priority_queue<container, std::vector<container>, std::greater<container>> reader_queue;

	void pull_new_elements() {
		container next;
		while (queue.try_pop(next)) {
			next.version = version++;
			reader_queue.push(next);
		}
	}

public:

	single_reader_priority_queue() : alive(true) {}

	bool enqueue(T* element, uint64_t priority) {
		if (alive) {
			queue.push(container{element, priority, 0});
		}
		return alive;
	}

	bool try_dequeue(T* &element) {
		pull_new_elements();

		if (!alive || reader_queue.empty() || reader_queue.top().priority > util::now()) {
			return false;
		}

		element = reader_queue.top().element;
		reader_queue.pop();
		return true;
	}

	T* dequeue() {
		// pull_new_elements(); 

		// auto now = util::now();
		// if (alive && !reader_queue.empty() && reader_queue.top().priority > now) {
		// 	std::cout << "Waiting " << (reader_queue.top().priority - now) << std::endl;
		// }

		T* element = nullptr;
		while (alive && !try_dequeue(element)) usleep(1);
		return element;
	}

	std::vector<T*> drain() {
		pull_new_elements();

		std::vector<T*> elements;
		while (!reader_queue.empty()) {
			elements.push_back(reader_queue.top().element);
			reader_queue.pop();
		}

		return elements;
	}

	void shutdown() {
		alive = false;
	}

};

	/* This is a priority queue, but one where priorities also define a minimum
time that an enqueued task is eligible to be dequeued.  The queue will block
if no eligible tasks are available */
template <typename T> class time_release_priority_queue {
private:
	struct container {
		T* element;

		// TODO: priority should be a chrono timepoint not the uint64_t, to avoid
		//       expensive conversions.  Or, a different clock altogether
		uint64_t priority;
		uint64_t version;

		friend bool operator < (const container& lhs, const container &rhs) {
			return lhs.priority < rhs.priority || 
			  (lhs.priority == rhs.priority && lhs.version < rhs.version);
		}
		friend bool operator > (const container& lhs, const container &rhs) {
			return lhs.priority > rhs.priority ||
			  (lhs.priority == rhs.priority && lhs.version > rhs.version);
		}
	};

	std::atomic_bool alive;

	std::atomic_flag in_use;
	std::atomic<uint64_t> version;
	std::priority_queue<container, std::vector<container>, std::greater<container>> queue;

public:

	time_release_priority_queue() : alive(true), in_use(ATOMIC_FLAG_INIT), version(0) {}

	bool enqueue(T* element, uint64_t priority) {
		while (in_use.test_and_set());

		// TODO: will have to convert priority to a chrono::timepoint
		if (alive) {
			queue.push(container{element, priority, version});
			version++;
		}

		in_use.clear();

		return alive;
	}

	bool try_dequeue(T* &element) {
		while (in_use.test_and_set());

		if (!alive || queue.empty() || queue.top().priority > util::now()) {
			in_use.clear();
			return false;
		}

		element = queue.top().element;
		queue.pop();

		in_use.clear();
		return true;
	}

	T* dequeue() {
		while (alive) {
			while (in_use.test_and_set());

			if (queue.empty()) {
				uint64_t version_seen = version.load();
				in_use.clear();

				// Spin until something is enqueued
				while (alive && version.load() == version_seen);

			} else if (queue.top().priority > util::now()) {
				uint64_t next_eligible = queue.top().priority;
				uint64_t version_seen = version.load();
				in_use.clear();

				// Spin until the top element is eligible or something new is enqueued
				while (alive && version.load() == version_seen && next_eligible > util::now());

			} else {
				T* element = queue.top().element;
				queue.pop();
				in_use.clear();
				return element;

			}
		}
		return nullptr;
	}

	std::vector<T*> drain() {
		while (in_use.test_and_set());

		std::vector<T*> elements;
		while (!queue.empty()) {
			elements.push_back(queue.top().element);
			queue.pop();
		}

		in_use.clear();

		return elements;
	}

	void shutdown() {
		while (in_use.test_and_set());

		alive = false;
		version++;

		in_use.clear();
	}
	
};
}

#endif