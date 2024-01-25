#ifndef _CLOCKWORK_CONTROLLER_WORKER_TRACKER_H_
#define _CLOCKWORK_CONTROLLER_WORKER_TRACKER_H_

#include <queue>
#include "clockwork/util.h"


class WorkerTracker {
private:
	struct Work { int id; uint64_t size; uint64_t work_begin; };
	int clock_;
	uint64_t work_begin = 0;
	uint64_t lag; // allowable lag
	uint64_t future; // time into the future to schedule
	std::deque<Work> outstanding;
	uint64_t total_outstanding = 0;

	void update(int id, uint64_t time_of_completion, bool success) {
		if (outstanding.front().id == id) {
			total_outstanding -= outstanding.front().size;
			work_begin = time_of_completion;
			outstanding.pop_front();
		} else {
			// the work was done but we don't know when
			for (auto it = outstanding.begin(); it != outstanding.end(); it++){
				if (it->id == id) {
					total_outstanding -= it->size;
					if (success) work_begin += it->size/clock_;
					outstanding.erase(it);
					break;
				}
			}
		}
		if (outstanding.size() > 0) {
			work_begin = std::max(work_begin, outstanding.front().work_begin);
		}
	}

public:

	WorkerTracker(int clock, uint64_t lag = 100000000UL, uint64_t future=0UL) : clock_(clock), lag(lag), future(future) {}

	// Returns the time outstanding work will complete
	uint64_t available() {
		uint64_t now = util::now();
		uint64_t work_begin = this->work_begin;
		if (outstanding.size() > 0 && (work_begin + (outstanding.front().size / clock_) + lag < now)) {
			// Outstanding work has mysteriously not completed
			work_begin = now - lag - outstanding.front().size / clock_;
		}
		return std::max(work_begin + total_outstanding / clock_, now + future);
	}

	int clock() {
		return clock_;
	}

	void update_clock(int clock) {
		this->clock_ = clock;
	}

	void add(int id, uint64_t work_size, uint64_t work_begin = 0) {
		if (outstanding.empty()) {
			this->work_begin = std::max(work_begin, util::now());
		}
		uint64_t size = work_size * clock_;
		outstanding.push_back({id, size, work_begin});
		total_outstanding += size;
	}

	void success(int id, uint64_t time_of_completion) {
		update(id, time_of_completion, true);
	}

	void error(int id, uint64_t time_of_completion) {
		update(id, time_of_completion, false);
	}

};

#endif