// Copyright 2020 Max Planck Institute for Software Systems

#ifndef SRC_CLOCKWORK_CONTROLLER_INFER5_QUEUES_H_
#define SRC_CLOCKWORK_CONTROLLER_INFER5_QUEUES_H_

#include <atomic>
#include <algorithm>
#include <string>
#include <sstream>
#include <set>
#include "clockwork/controller/scheduler.h"
#include "clockwork/controller/worker_tracker.h"
#include "clockwork/controller/load_tracker.h"
#include "clockwork/telemetry/controller_action_logger.h"
#include "clockwork/thread.h"
#include "clockwork/api/worker_api.h"
#include "clockwork/sliding_window.h"
#include "tbb/mutex.h"
#include "tbb/queuing_mutex.h"
#include "tbb/spin_mutex.h"

namespace clockwork {
namespace scheduler {
namespace infer5 {



}
}
}
#endif // SRC_CLOCKWORK_CONTROLLER_INFER5_QUEUES_H_