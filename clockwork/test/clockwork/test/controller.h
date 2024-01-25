#include <unistd.h>
#include <libgen.h>
#include <fstream>
#include <algorithm>

#include <cuda_runtime.h>
#include "clockwork/api/worker_api.h"
#include "clockwork/test/util.h"
#include "clockwork/model/model.h"
#include "clockwork/worker.h"
#include <catch2/catch.hpp>
#include "clockwork/test/actions.h"
#include "tbb/concurrent_queue.h"

using namespace clockwork;
using namespace clockwork::model;

class TestController : public workerapi::Controller {
public:
    tbb::concurrent_queue<std::shared_ptr<workerapi::Result>> results;

    void sendResult(std::shared_ptr<workerapi::Result> result) {
        results.push(result);
    }

    std::shared_ptr<workerapi::Result> awaitResult() {
        std::shared_ptr<workerapi::Result> result;
        while (!results.try_pop(result));
        return result;
    }

    void expect(int expected_status_code) {
        std::shared_ptr<workerapi::Result> result = awaitResult();
        INFO("id=" << result->id << " type=" << result->action_type << " status=" << result->status);
        if (result->status != actionSuccess) {
            auto error = std::static_pointer_cast<workerapi::ErrorResult>(result);
            INFO(error->message);
            REQUIRE(result->status == expected_status_code);
        } else {
            REQUIRE(result->status == expected_status_code);            
        }
	}
};
