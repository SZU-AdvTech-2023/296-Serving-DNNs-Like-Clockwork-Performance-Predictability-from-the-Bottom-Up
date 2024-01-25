#include <unistd.h>
#include <libgen.h>
#include <fstream>
#include <algorithm>
#include <memory>

#include <cuda_runtime.h>
#include "clockwork/api/worker_api.h"
#include "clockwork/test/util.h"
#include "clockwork/model/model.h"
#include "clockwork/task.h"
#include "clockwork/memory.h"
#include "clockwork/config.h"
#include "clockwork/sliding_window.h"
#include <catch2/catch.hpp>
#include <stdio.h>

using namespace clockwork;
using namespace clockwork::model;

TEST_CASE("Test estimator", "[estimator] [util]") {
    SlidingWindow window(3);

    window.insert(70);
    REQUIRE(window.get_value(0) == 70);
    REQUIRE(window.get_size() == 1);
    REQUIRE(window.get_percentile(0) == 70);
    REQUIRE(window.get_percentile(0.25) == 70);
    REQUIRE(window.get_percentile(0.5) == 70);
    REQUIRE(window.get_percentile(0.75) == 70);
    REQUIRE(window.get_percentile(1) == 70);

    window.insert(50);
    REQUIRE(window.get_value(0) == 50);
    REQUIRE(window.get_value(1) == 70);
    REQUIRE(window.get_size() == 2);
    REQUIRE(window.get_percentile(0) == 50);
    REQUIRE(window.get_percentile(0.25) == 55);
    REQUIRE(window.get_percentile(0.5) == 60);
    REQUIRE(window.get_percentile(0.75) == 65);
    REQUIRE(window.get_percentile(1) == 70);

    window.insert(90);
    REQUIRE(window.get_value(0) == 50);
    REQUIRE(window.get_value(1) == 70);
    REQUIRE(window.get_value(2) == 90);
    REQUIRE(window.get_size() == 3);
    REQUIRE(window.get_percentile(0) == 50);
    REQUIRE(window.get_percentile(0.25) == 60);
    REQUIRE(window.get_percentile(0.5) == 70);
    REQUIRE(window.get_percentile(0.75) == 80);
    REQUIRE(window.get_percentile(1) == 90);

    window.insert(110);
    REQUIRE(window.get_value(0) == 50);
    REQUIRE(window.get_value(1) == 90);
    REQUIRE(window.get_value(2) == 110);
    REQUIRE(window.get_size() == 3);
    REQUIRE(window.get_percentile(0) == 50);
    REQUIRE(window.get_percentile(0.25) == 70);
    REQUIRE(window.get_percentile(0.5) == 90);
    REQUIRE(window.get_percentile(0.75) == 100);
    REQUIRE(window.get_percentile(1) == 110);

    window.insert(20);
    REQUIRE(window.get_value(0) == 20);
    REQUIRE(window.get_value(1) == 90);
    REQUIRE(window.get_value(2) == 110);
    REQUIRE(window.get_percentile(0) == 20);
    REQUIRE(window.get_percentile(0.25) == 55);
    REQUIRE(window.get_percentile(0.5) == 90);
    REQUIRE(window.get_percentile(0.75) == 100);
    REQUIRE(window.get_percentile(1) == 110);
    REQUIRE(window.get_size() == 3);

    window.insert(20);
    REQUIRE(window.get_value(0) == 20);
    REQUIRE(window.get_value(1) == 20);
    REQUIRE(window.get_value(2) == 110);
    REQUIRE(window.get_percentile(0) == 20);
    REQUIRE(window.get_percentile(0.25) == 20);
    REQUIRE(window.get_percentile(0.5) == 20);
    REQUIRE(window.get_percentile(0.75) == 65);
    REQUIRE(window.get_percentile(1) == 110);
    REQUIRE(window.get_size() == 3);

}


TEST_CASE("Test batch lookup", "[batchlookup] [util]") {
    {
        auto lookup = util::make_batch_lookup({0});
        REQUIRE(lookup.size() == 1);
        REQUIRE(lookup[0] == 0);
    }
    {
        auto lookup = util::make_batch_lookup({1});
        REQUIRE(lookup.size() == 2);
        REQUIRE(lookup[0] == 0);
        REQUIRE(lookup[1] == 1);
    }
    {
        auto lookup = util::make_batch_lookup({2});
        REQUIRE(lookup.size() == 3);
        REQUIRE(lookup[0] == 0);
        REQUIRE(lookup[1] == 0);
        REQUIRE(lookup[2] == 2);
    }
    {
        auto lookup = util::make_batch_lookup({4});
        REQUIRE(lookup.size() == 5);
        REQUIRE(lookup[0] == 0);
        REQUIRE(lookup[1] == 0);
        REQUIRE(lookup[2] == 0);
        REQUIRE(lookup[3] == 0);
        REQUIRE(lookup[4] == 4);
    }
    {
        auto lookup = util::make_batch_lookup({1,2,4});
        REQUIRE(lookup.size() == 5);
        REQUIRE(lookup[0] == 0);
        REQUIRE(lookup[1] == 1);
        REQUIRE(lookup[2] == 2);
        REQUIRE(lookup[3] == 2);
        REQUIRE(lookup[4] == 4);
    }
    {
        auto lookup = util::make_batch_lookup({1,2,4,8});
        REQUIRE(lookup.size() == 9);
        REQUIRE(lookup[0] == 0);
        REQUIRE(lookup[1] == 1);
        REQUIRE(lookup[2] == 2);
        REQUIRE(lookup[3] == 2);
        REQUIRE(lookup[4] == 4);
        REQUIRE(lookup[5] == 4);
        REQUIRE(lookup[6] == 4);
        REQUIRE(lookup[7] == 4);
        REQUIRE(lookup[8] == 8);
    }
    {
        auto lookup = util::make_batch_lookup({1,2,4,8,16});
        REQUIRE(lookup.size() == 17);
        REQUIRE(lookup[0] == 0);
        REQUIRE(lookup[1] == 1);
        REQUIRE(lookup[2] == 2);
        REQUIRE(lookup[3] == 2);
        REQUIRE(lookup[4] == 4);
        REQUIRE(lookup[5] == 4);
        REQUIRE(lookup[6] == 4);
        REQUIRE(lookup[7] == 4);
        REQUIRE(lookup[8] == 8);
        REQUIRE(lookup[9] == 8);
        REQUIRE(lookup[10] == 8);
        REQUIRE(lookup[11] == 8);
        REQUIRE(lookup[12] == 8);
        REQUIRE(lookup[13] == 8);
        REQUIRE(lookup[14] == 8);
        REQUIRE(lookup[15] == 8);
        REQUIRE(lookup[16] == 16);
    }
}


// TEST_CASE("Test scheduler work tracker", "[scheduler] [worktracker]") {
//     clockwork::scheduler::infer3::Scheduler::WorkTracker2 tracker(4, 4);

//     tracker.addRequest(1, 70000000UL, 100000000UL);
//     tracker.addRequest(2, 40000000UL, 100000000UL);
//     tracker.addRequest(3, 50000000UL, 100000000UL);

//     REQUIRE(tracker.loadModel(0) == 1);
//     REQUIRE(tracker.loadModel(0) == 3);
//     REQUIRE(tracker.loadModel(0) == 2);
//     REQUIRE(tracker.loadModel(0) == 0);
// }


// TEST_CASE("Test scheduler work tracker 2", "[scheduler] [worktracker]") {
//     clockwork::scheduler::infer3::Scheduler::WorkTracker2 tracker(4, 4);

//     tracker.addRequest(1, 30000000UL, 100000000UL);
//     REQUIRE(tracker.loadModel(0) == 1);

//     for (unsigned i = 0; i < 100; i++) {
//         tracker.addRequest(1, 30000000UL, 100000000UL);
//     }
//     tracker.addRequest(2, 30000000UL, 100000000UL);
//     tracker.addRequest(3, 20000000UL, 100000000UL);
//     tracker.addRequest(0, 40000000UL, 100000000UL);

//     REQUIRE(tracker.loadModel(1) == 1);
//     REQUIRE(tracker.loadModel(2) == 1);
//     REQUIRE(tracker.loadModel(3) == 1);
//     REQUIRE(tracker.loadModel(0) == 0);
//     REQUIRE(tracker.loadModel(0) == 2);
//     REQUIRE(tracker.loadModel(0) == 3);
// }


// TEST_CASE("Test scheduler work tracker 3", "[scheduler] [worktracker]") {
//     clockwork::scheduler::infer3::Scheduler::WorkTracker2 tracker(4, 4);

//     tracker.loadModelComplete(0, tracker.loadModel(0), true);
//     tracker.loadModelComplete(0, tracker.loadModel(0), true);
//     tracker.loadModelComplete(0, tracker.loadModel(0), true);
//     tracker.loadModelComplete(0, tracker.loadModel(0), true);


//     tracker.addRequest(1, 30000000UL, 100000000UL);

//     for (unsigned i = 0; i < 10; i++) {
//         tracker.addRequest(1, 30000000UL, 100000000UL);
//     }
//     tracker.addRequest(2, 30000000UL, 100000000UL);
//     tracker.addRequest(3, 20000000UL, 100000000UL);
//     tracker.addRequest(0, 40000000UL, 100000000UL);

//     REQUIRE(tracker.evictModel(0) == 3);
//     REQUIRE(tracker.evictModel(0) == 2);
//     REQUIRE(tracker.evictModel(0) == 0);
//     REQUIRE(tracker.evictModel(0) == 1);
// }
