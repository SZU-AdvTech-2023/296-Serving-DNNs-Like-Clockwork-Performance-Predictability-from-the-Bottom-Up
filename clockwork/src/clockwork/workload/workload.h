#ifndef _CLOCKWORK_WORKLOAD_WORKLOAD_H_
#define _CLOCKWORK_WORKLOAD_WORKLOAD_H_

#include <queue>
#include <cstdint>
#include <functional>
#include <vector>
#include "clockwork/util.h"
#include "clockwork/client.h"
#include "tbb/concurrent_queue.h"
#include <random>
#include "dmlc/logging.h"

namespace clockwork {
namespace workload {

class Workload;

typedef std::exponential_distribution<double> Exponential;

class Engine {
private:
	struct element {
		uint64_t ready;
		std::function<void(void)> callback;

		friend bool operator < (const element& lhs, const element &rhs) {
			return lhs.ready < rhs.ready;
		}
		friend bool operator > (const element& lhs, const element &rhs) {
			return lhs.ready > rhs.ready;
		}
	};

	uint64_t now = util::now();
	tbb::concurrent_queue<std::function<void(void)>> runqueue;
	std::priority_queue<element, std::vector<element>, std::greater<element>> queue;
	std::vector<Workload*> workloads;

public:
	std::atomic_int running = 0;
	util::InputGenerator input_generator;

	void AddWorkload(Workload* workload, uint64_t start_after = 0);
	void SetTimeout(uint64_t timeout, std::function<void(void)> callback);
	void InferComplete(Workload* workload, unsigned model_index);
	void InferError(Workload* workload, unsigned model_index, int status);

	void Run(clockwork::Client* client);

};

class Workload {
public:
	std::vector<clockwork::Model*> models;

	int user_id;
	Engine* engine;
	uint64_t start_after = 0;
	uint64_t stop_after = UINT64_MAX;

	Workload(int id);
	Workload(int id, clockwork::Model* model);
	Workload(int id, std::vector<clockwork::Model*> &models);

	// Used during workload initialization
	void AddModel(clockwork::Model* model);
	void AddModels(std::vector<clockwork::Model*> &models);
	void SetEngine(Engine* engine);

	// Methods to be called by subclasses
	void Infer(unsigned model_index = 0);
	void SetTimeout(uint64_t timeout, std::function<void(void)> callback);

	// Methods to be implemented by subclasses
	virtual void Start(uint64_t now) = 0;
	virtual void InferComplete(uint64_t now, unsigned model_index) = 0;
	virtual void InferError(uint64_t now, unsigned model_index, int status) = 0;

	// Optional methods to be overridden by subclasses
	virtual void InferErrorInitializing(uint64_t now, unsigned model_index);

};

// Doesn't run requests, but can perform actions in the Engine
class Timer : public Workload {
public:

	Timer() : Workload(-1) {}

	virtual void Start(uint64_t now) = 0;
	void InferComplete(uint64_t now, unsigned model_index) {
		CHECK(false) << "Timers don't submit infer requests";
	}
	void InferError(uint64_t now, unsigned model_index, int status) {
		CHECK(false) << "Timers don't submit infer requests";
	}
	void InferErrorInitializing(uint64_t now, unsigned model_index) {
		CHECK(false) << "Timers don't submit infer requests";
	}
};

class AdjustSLO : public Timer {
private:
	std::vector<clockwork::Model*> models;
	uint64_t period;
	float current;
	std::function<float(float)> update;
	std::function<bool(float)> terminate;
public:
	AdjustSLO(
		float period_seconds, 
		float initial_slo_factor,
		std::vector<clockwork::Model*> models,
		std::function<float(float)> update,
		std::function<bool(float)> terminate)
			: period(period_seconds * 1000000000UL),
			  current(initial_slo_factor),
			  models(models),
			  update(update),
			  terminate(terminate) {
		for (auto model : models) {
			model->set_slo_factor(initial_slo_factor);
		}
	}
	void UpdateSLO() {
		current = update(current);

		if (terminate(current)) {
			engine->running = 0;
			std::cout << "Terminating engine" << std::endl;
			return;
		}

		std::cout << "Updating slo_factor to " << current << std::endl;
		for (auto model : models) {
			model->set_slo_factor(current);
		}
		SetTimeout(period, [this]() { UpdateSLO(); });
	}
	virtual void Start(uint64_t now) {
		SetTimeout(period, [this]() { UpdateSLO(); });
	}
};

class ClosedLoop : public Workload {
public:
	const unsigned concurrency;
	uint64_t num_requests;
	uint64_t jitter;
	unsigned idx = 0; // for models

	ClosedLoop(int id, clockwork::Model* model, unsigned concurrency);

	ClosedLoop(int id, std::vector<clockwork::Model*> models, unsigned concurrency);

	ClosedLoop(int id, clockwork::Model* model, unsigned concurrency,
		uint64_t num_requests, uint64_t jitter);

	ClosedLoop(int id, std::vector<clockwork::Model*> models,
		unsigned concurrency, uint64_t num_requests, uint64_t jitter);

	virtual void Start(uint64_t now);
	virtual void ActualStart();
	virtual void InferComplete(uint64_t now, unsigned model_index);
	virtual void InferError(uint64_t now, unsigned model_index, int status);
	virtual void InferErrorInitializing(uint64_t now, unsigned model_index);

	unsigned GetAndUpdateIdx();
};

template <typename TDISTRIBUTION> class OpenLoop : public Workload {
public:
	std::minstd_rand rng;
	TDISTRIBUTION distribution;

	OpenLoop(int id, clockwork::Model* model, int rng_seed, TDISTRIBUTION distribution) : 
		Workload(id, model), rng(rng_seed), distribution(distribution) {
	}

	void set_distribution(TDISTRIBUTION new_distribution) {
		distribution = new_distribution;
	}

	void Submit() {
		Infer(0);
		uint64_t timeout = distribution(rng);
		SetTimeout(timeout, [this]() { Submit(); });
	}

	void Start(uint64_t now) {
		SetTimeout(distribution(rng), [this]() { Submit(); });
	}

	void InferComplete(uint64_t now, unsigned model_index) {}
	void InferError(uint64_t now, unsigned model_index, int status) {}

};

class Static {
public:
	uint64_t value;
	Static(uint64_t value) : value(value) {}
	const uint64_t& operator()(std::minstd_rand rng) const { return value; }

	Static& operator= (const Static &other) {
		value = other.value;
		return *this;
	}
};

class FixedRate : public OpenLoop<Static> {
public:
	// mean is provided in seconds
	FixedRate(int id, clockwork::Model* model, int rng_seed, double rate) : 
		OpenLoop(id, model, rng_seed, Static(1000000000.0 / rate)) {
	}

};

class PoissonOpenLoop : public OpenLoop<Exponential> {
public:
	// mean is provided in seconds
	PoissonOpenLoop(int id, clockwork::Model* model, int rng_seed, double rate) : 
		// Exponential requires the rate parameter. Here, we translate the
		// rate parameter, which is in requests/second, into requests/nanosecond
		OpenLoop(id, model, rng_seed, Exponential(rate / 1000000000.0)) {
	}

};

template <typename DBURST, typename DIDLE> 
class BurstyClosedLoop : public Workload {
public:
	unsigned concurrency;
	std::minstd_rand rng;
	DBURST d_burst;
	DIDLE d_idle;

	bool bursting = false;
	int outstanding = 0;

	BurstyClosedLoop(int id, 
		clockwork::Model* model,
		unsigned concurrency,
		int rng_seed, 
		DBURST d_burst,
		DIDLE d_idle) : 
			Workload(id, model), 
			concurrency(concurrency),
			rng(rng_seed),
			d_burst(d_burst),
			d_idle(d_idle)
	{}

	void StartBursting(uint64_t duration) {
		bursting = true;
		for (unsigned i = 0; i < concurrency; i++) {
			outstanding++;
			Infer();
		}
		SetTimeout(duration, [this]() { bursting = false; });
	}

	void ScheduleNextBurst(uint64_t interval) {
		SetTimeout(interval, [this]() { 
			StartBursting(d_burst(rng)); 
		});
	}

	void Start(uint64_t now) {
		double s = static_cast<double>(rng()) / static_cast<double>(UINT64_MAX);
		uint64_t burst = d_burst(rng);
		uint64_t idle = d_idle(rng);
		uint64_t initial_period = burst + idle;
		uint64_t start_at = static_cast<uint64_t>(((double)initial_period) * s);
		if (start_at > burst) {
			ScheduleNextBurst(initial_period - start_at);
		} else {
			StartBursting(burst - start_at);
		}
	}

	void RequestComplete() {
		outstanding--;
		if (bursting) {
			Infer();
			outstanding++;
		} else if (outstanding == 0) {
			ScheduleNextBurst(d_burst(rng));
		}
	}

	void InferComplete(uint64_t now, unsigned model_index) { RequestComplete(); }
	void InferError(uint64_t now, unsigned model_index, int status) { RequestComplete(); }
	void InferErrorInitializing(uint64_t now, unsigned model_index) { RequestComplete(); }

};


class BurstyPoissonClosedLoop : public BurstyClosedLoop<Exponential, Exponential> {
public:
	// mean is provided in seconds
	BurstyPoissonClosedLoop(int id, clockwork::Model* model, unsigned concurrency,
		int rng_seed, double burstIntervalSeconds, double burstDurationSeconds) : 
		BurstyClosedLoop(id, model, concurrency, rng_seed,
			Exponential(1 / (1000000000.0 * burstIntervalSeconds)),
			Exponential(1 / (1000000000.0 * burstDurationSeconds))) {
	}

};

template <typename DREQUEST, typename DBURST, typename DIDLE> 
class BurstyOpenLoop : public Workload {
public:
	int current_burst = 0;
	std::minstd_rand rng_request;
	std::minstd_rand rng_burst;
	DREQUEST d_request;
	DBURST d_burst;
	DIDLE d_idle;

	BurstyOpenLoop(int id, 
		clockwork::Model* model, 
		int rng_seed, 
		DREQUEST d_request,
		DBURST d_burst,
		DIDLE d_idle) : 
			Workload(id, model), 
			rng_request(rng_seed+1), 
			rng_burst(rng_seed), 
			d_request(d_request),
			d_burst(d_burst),
			d_idle(d_idle)
	{}

	void Submit(int burst) {
		if (burst == this->current_burst) {
			Infer(0);
			uint64_t timeout = d_request(rng_request);
			SetTimeout(timeout, [this, burst]() { Submit(burst); });
		}
	}

	void StartBursting() {
		Submit(current_burst);
		uint64_t stop_burst_at = d_burst(rng_burst);
		SetTimeout(stop_burst_at, [this]() { StopBursting(); });
	}

	void StopBursting() {
		current_burst++;
		uint64_t next_burst_at = d_idle(rng_burst);
		SetTimeout(next_burst_at, [this]() { StartBursting(); });
	}

	void Start(uint64_t now) {
		double s = static_cast<double>(rng_burst()) / static_cast<double>(UINT64_MAX);
		uint64_t burst = d_burst(rng_burst);
		uint64_t idle = d_idle(rng_burst);
		uint64_t initial_period = burst + idle;
		uint64_t start_at = static_cast<uint64_t>(((double)initial_period) * s);
		if (start_at > burst) {
			uint64_t next_burst_at = initial_period - start_at;
			SetTimeout(next_burst_at, [this]() { StartBursting(); });
		} else {
			uint64_t stop_burst_at = burst - start_at;
			Submit(current_burst);
			SetTimeout(stop_burst_at, [this]() { StopBursting(); });
		}
	}

	void InferComplete(uint64_t now, unsigned model_index) {}
	void InferError(uint64_t now, unsigned model_index, int status) {}

};

class BurstyPoissonOpenLoop : public BurstyOpenLoop<Exponential, Exponential, Exponential> {
public:
	// mean is provided in seconds
	BurstyPoissonOpenLoop(int id, clockwork::Model* model, int rng_seed, 
			double rate, double burstDurationSeconds, double idleDurationSeconds) : 
		BurstyOpenLoop(id, model, rng_seed, 
			Exponential(rate / 1000000000.0),
			Exponential(1 / (1000000000.0 * burstDurationSeconds)),
			Exponential(1 / (1000000000.0 * idleDurationSeconds))) {
	}

};

template <typename TDISTRIBUTION> 
class TraceReplay : public Workload {
public:
	bool randomise_start;
	unsigned starting_interval;

	std::vector<unsigned> intervals;
	uint64_t interval_duration;
	std::minstd_rand rng;
	TDISTRIBUTION distribution;

	TraceReplay(int id, clockwork::Model* model, int rng_seed,
		std::vector<unsigned> &intervals, // request rate for each interval, specified as requests per minute
		double interval_duration_seconds,
		int start_at, // which interval to start at.  if set to -1, will randomise
		TDISTRIBUTION distribution) : Workload(id, model), 
			rng(rng_seed),
			intervals(intervals),
			distribution(distribution), 
			interval_duration(interval_duration_seconds * 1000000000.0)
	{
		CHECK(intervals.size() > 0) << "Cannot create TraceReplay without intervals";
		if (start_at < 0) {
			starting_interval = start_at % intervals.size();
			randomise_start = true;
		} else {
			starting_interval = 0;
			randomise_start = false;
		}
	}

	void set_distribution(TDISTRIBUTION new_distribution) {
		distribution = new_distribution;
	}

	void Submit(unsigned interval, uint64_t remaining, uint64_t next_arrival) {
		unsigned rate = intervals[interval];
		// std::cout << "Interval " << interval << ", rate=" << rate << ", remaining=" << remaining << ", next=" << next_arrival << std::endl;
		if (rate == 0) {
			SetTimeout(remaining, [this, interval, next_arrival]() {
				Submit(
					(interval + 1) % intervals.size(),
					interval_duration,
					next_arrival
				);
			});
		} else {
			uint64_t timeout = next_arrival / rate;
			if (timeout <= remaining) {
				remaining -= timeout;
				SetTimeout(timeout, [this, interval, remaining]() {
					Infer(0);
					Submit(
						interval, 
						remaining, 
						distribution(rng)
					);
				});
			} else {
				next_arrival = (timeout - remaining) * rate;
				SetTimeout(remaining, [this, interval, next_arrival]() {
					Submit(
						(interval + 1) % intervals.size(), 
						interval_duration, 
						next_arrival
					);
				});
			}
		}
	}

	void Start(uint64_t now) {
		// Determing starting point in interval
		uint64_t remaining = interval_duration;
		if (randomise_start) remaining = rng() % remaining;

		uint64_t next_arrival = distribution(rng);
		Submit(starting_interval, remaining, next_arrival);
	}

	void InferComplete(uint64_t now, unsigned model_index) {}
	void InferError(uint64_t now, unsigned model_index, int status) {}

};

class PoissonTraceReplay : public TraceReplay<Exponential> {
public:

	PoissonTraceReplay(int id, clockwork::Model* model, int rng_seed,
		std::vector<unsigned> &interval_rates, // request rate for each interval, specified as requests per minute
		double scale_factor = 1.0,
		double interval_duration_seconds = 60.0,
		int start_at = 0 // which interval to start at.  if set to -1, will randomise
	) : TraceReplay(id, model, rng_seed, interval_rates, interval_duration_seconds, start_at,
			Exponential(scale_factor / 60000000000.0)) {
	}

};

class AdjustScaleFactor : public Timer {
public:
	std::vector<TraceReplay<Exponential>*> workloads;
	uint64_t period;
	double current;
	std::function<double(double)> update;
	std::function<bool(double)> terminate;

	AdjustScaleFactor(
		unsigned period_seconds,
		double initial_scale_factor,
		std::vector<TraceReplay<Exponential>*> workloads,
		std::function<double(double)> update,
		std::function<bool(double)> terminate)
			: period(period_seconds * 1000000000UL),
			  current(initial_scale_factor),
			  workloads(workloads),
			  update(update),
			  terminate(terminate) {}

	virtual void Start(uint64_t now) {
		SetTimeout(period, [this]() { update_scale_factor(); });
	}

  void update_scale_factor() {
		current = update(current);
		try_termination();

		std::cout << ">>>> Updating scale_factor to " << current << std::endl;
		for (auto workload : workloads) {
			workload->set_distribution(Exponential(current / 60000000000.0));
		}

		SetTimeout(period, [this]() { update_scale_factor(); });
  }

	void try_termination() {
		if (terminate(current)) {
			engine->running = 0;
			std::cout << "Terminating engine" << std::endl;
		}
	}
};

template <typename TDISTRIBUTION> class AdjustRate : public Timer {
public:
	std::vector<OpenLoop<TDISTRIBUTION>*> workloads;
	uint64_t period;
	double current;
	std::function<double(double)> update;
	std::function<bool(double)> terminate;

	AdjustRate(
		unsigned period_seconds,
		double initial_rate,
		std::vector<OpenLoop<TDISTRIBUTION>*> workloads,
		std::function<double(double)> update,
		std::function<bool(double)> terminate)
			: period(period_seconds * 1000000000UL),
			  current(initial_rate),
			  workloads(workloads),
			  update(update),
			  terminate(terminate) {}
	virtual void UpdateRate() = 0;

	virtual void Start(uint64_t now) {
		SetTimeout(period, [this]() { UpdateRate(); });
	}

	void try_termination() {
		if (terminate(current)) {
			engine->running = 0;
			std::cout << "Terminating engine" << std::endl;
		}
	}
};

class AdjustPoissonRate : public AdjustRate<Exponential> {
public:
	AdjustPoissonRate(
		unsigned period_seconds,
		double initial_rate,
		std::vector<OpenLoop<Exponential>*> workloads,
		std::function<double(double)> update,
		std::function<bool(double)> terminate) :
			AdjustRate(
				period_seconds,
				initial_rate,
				workloads,
				update,
				terminate
			) {}

	void UpdateRate() {
		current = update(current);
		try_termination();

		std::cout << "Updating rate to " << current << std::endl;
		for (auto workload : workloads) {
			workload->set_distribution(Exponential(current / 1000000000.0));
		}

		SetTimeout(period, [this]() { UpdateRate(); });
	}
};

class AdjustFixedRate : public AdjustRate<Static> {
public:
	AdjustFixedRate(
		unsigned period_seconds,
		double initial_rate,
		std::vector<OpenLoop<Static>*> workloads,
		std::function<double(double)> update,
		std::function<bool(double)> terminate) :
			AdjustRate(
				period_seconds,
				initial_rate,
				workloads,
				update,
				terminate
			) {}

	void UpdateRate() {
		double previous = current;
		current = update(current);
		try_termination();

		std::cout << "Updating rate to " << current << std::endl;
		for (auto workload : workloads) {
			workload->set_distribution(Static(1000000000.0 / current));
		}

		if (previous == 0) { Start(util::now()); }

		SetTimeout(period, [this]() { UpdateRate(); });
	}
};

}
}

#endif 

