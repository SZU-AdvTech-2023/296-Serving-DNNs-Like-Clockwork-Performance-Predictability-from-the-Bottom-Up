#include <catch2/catch.hpp>

#include <unistd.h>
#include <thread>
#include <unordered_map>
#include <algorithm>
#include <cstdlib>
#include <iostream>

#include "clockwork/util.h"
#include "clockwork/priority_queue.h"

TEST_CASE("Priority Queue Simple Dequeue Order", "[queue]") {
    using namespace clockwork;

    time_release_priority_queue<int> q;

    std::vector<int*> elements;
    for (unsigned i = 0; i < 10; i++) {
        int* element = new int();
        q.enqueue(element, i);
        elements.push_back(element);
    }

    for (unsigned i = 0; i < 10; i++) {
        int* element = q.dequeue();
        REQUIRE(element == elements[i]);
    }
}

TEST_CASE("Priority Queue Reverse Dequeue Order", "[queue]") {
    using namespace clockwork;

    time_release_priority_queue<int> q;

    std::vector<int*> elements;
    for (unsigned i = 0; i < 10; i++) {
        int* element = new int();
        q.enqueue(element, 9-i);
        elements.push_back(element);
    }

    for (unsigned i = 0; i < 10; i++) {
        int* element = q.dequeue();
        REQUIRE(element == elements[9-i]);
    }
}

TEST_CASE("Priority Queue ZigZag Dequeue Order", "[queue]") {
    using namespace clockwork;

    time_release_priority_queue<int> q;


    std::vector<int> priorities = { 10, 0, 5, 8, 3, 7, 11, 1};
    std::unordered_map<int, int*> elems;

    for (int &priority : priorities) {
        int* element = new int();
        q.enqueue(element, priority);
        elems[priority] = element;
    }

    std::sort(priorities.begin(), priorities.end());

    for (int &priority : priorities) {
        int* element = q.dequeue();
        REQUIRE(element == elems[priority]);
    }
}

TEST_CASE("Priority Queue Multiple Identical Priorities", "[queue]") {
    using namespace clockwork;

    time_release_priority_queue<int> q;

    std::vector<int*> low;
    std::vector<int*> high;

    for (unsigned i = 0; i < 10; i++) {
        int* elow = new int();
        q.enqueue(elow, 10);
        low.push_back(elow);

        int* ehigh = new int();
        q.enqueue(ehigh, 20);
        high.push_back(ehigh);
    }

    for (unsigned i = 0; i < low.size(); i++) {
        int* e = q.dequeue();
        REQUIRE(std::find(low.begin(), low.end(), e) != low.end());
        REQUIRE(std::find(high.begin(), high.end(), e) == high.end());
    }

    for (unsigned i = 0; i < high.size(); i++) {
        int* e = q.dequeue();
        REQUIRE(std::find(low.begin(), low.end(), e) == low.end());
        REQUIRE(std::find(high.begin(), high.end(), e) != high.end());
    }
}

TEST_CASE("Priority Queue Eligible Time", "[queue]") {
    using namespace clockwork;

    time_release_priority_queue<int> q;

    uint64_t now = clockwork::util::now();
    std::vector<int*> elements;
    std::vector<uint64_t> priorities;


    for (int i = -10; i < 10; i++) {
        int* e = new int();
        uint64_t priority = now + i * 200000000L; // 200ms * i
        q.enqueue(e, priority);
        elements.push_back(e);
        priorities.push_back(priority);
    }

    for (int i = 0; i < elements.size(); i++) {
        int* e = q.dequeue();
        REQUIRE(e == elements[i]);
        REQUIRE(clockwork::util::now() >= priorities[i]);
    }
}

class ShutdownSignaller {
public:
    std::atomic_bool signalled_shutdown;
    std::thread thread;
    clockwork::time_release_priority_queue<int> &q;
    ShutdownSignaller(clockwork::time_release_priority_queue<int> &q) : 
            q(q), signalled_shutdown(false), thread(&ShutdownSignaller::run, this) {
    }
    ~ShutdownSignaller() {
        while (!signalled_shutdown);
        thread.join();
    }
    void run() {
        using namespace clockwork;
        uint64_t start = util::now();
        while (util::now() < start + 100000000UL) {
            usleep(1000);
        }
        signalled_shutdown = true;
        q.shutdown();
    }
};

TEST_CASE("Priority Queue Blocking Dequeue", "[queue] [shutdown]") {
    using namespace clockwork;

    time_release_priority_queue<int> q;
    ShutdownSignaller s(q);

    int* e = q.dequeue();

    {
        INFO("Blocking dequeue returned before shutdown signaller completed");
        REQUIRE(s.signalled_shutdown);
    }

    {
        INFO("Blocking dequeue should have returned a nullptr");
        REQUIRE(e == nullptr);
    }
}

class Dequeuer {
public:
    std::thread thread;
    clockwork::time_release_priority_queue<int> &q;
    std::atomic_bool complete;
    std::vector<int*> dequeued;
    Dequeuer(clockwork::time_release_priority_queue<int> &q) : 
            q(q), complete(false), thread(&Dequeuer::run, this) {
    }
    ~Dequeuer() {
        while (!complete);
        thread.join();
    }
    void run() {
        using namespace clockwork;
        uint64_t start = util::now();
        while (true) {
            int* element = q.dequeue();
            if (element == nullptr) {
                break;
            } else {
                dequeued.push_back(element);
            }
        }
        complete = true;
    }
};

TEST_CASE("Priority Queue Shutdown", "[queue] [shutdown]") {
    using namespace clockwork;

    time_release_priority_queue<int> q;

    Dequeuer d(q);

    std::vector<int*> elements;
    for (unsigned i = 0; i < 10; i++) {
        int* element = new int();
        q.enqueue(element, i);
        elements.push_back(element);
    }

    uint64_t now = util::now();
    while (util::now() < now + 100000000) { // Wait 100ms
        INFO("Dequeuer thread completed before it was signalled");
        REQUIRE(!d.complete.load());
        usleep(1000);
    }

    INFO("Dequeuer thread dequeued " << d.dequeued.size() << " elements");
    REQUIRE(d.dequeued.size() == 10);

    INFO("Dequeuer thread completed before it was signalled");
    REQUIRE(!d.complete.load());

    q.shutdown();

    now = util::now();
    while (util::now() < now + 1000000000) { // Max 1s
        if (d.complete.load()) {
            break;
        }
        usleep(1000);
    }
    INFO("Dequeuer thread never unblocked");
    REQUIRE(d.complete.load());
}

TEST_CASE("Priority Queue Enqueue after Shutdown", "[queue] [shutdown]") {
    using namespace clockwork;

    time_release_priority_queue<int> q;

    REQUIRE(q.enqueue(new int(), 0));

    q.shutdown();

    INFO("Should not be able to enqueue new elements after shutdown");
    REQUIRE(!q.enqueue(new int(), 0));
}

TEST_CASE("Priority Queue Drain after Shutdown", "[queue] [shutdown]") {
    using namespace clockwork;

    time_release_priority_queue<int> q;

    uint64_t now = util::now();

    REQUIRE(q.enqueue(new int(), now + 1000000000UL));
    REQUIRE(q.enqueue(new int(), now + 1000000000UL));
    REQUIRE(q.enqueue(new int(), now + 1000000000UL));

    q.shutdown();

    int* dequeued = q.dequeue();
    INFO("Shouldn't be able to dequeue after queue shutdown");
    REQUIRE(dequeued == nullptr);


    std::vector<int*> drained = q.drain();

    INFO("Unable to drain pending elements from queue after shutdown");
    REQUIRE(drained.size() == 3);
}

TEST_CASE("Single Reader Priority Queue Simple Dequeue Order", "[queue]") {
    using namespace clockwork;

    single_reader_priority_queue<int> q;

    std::vector<int*> elements;
    for (unsigned i = 0; i < 10; i++) {
        int* element = new int();
        q.enqueue(element, i);
        elements.push_back(element);
    }

    for (unsigned i = 0; i < 10; i++) {
        int* element = q.dequeue();
        REQUIRE(element == elements[i]);
    }
}

TEST_CASE("Single Reader Priority Queue Reverse Dequeue Order", "[queue]") {
    using namespace clockwork;

    single_reader_priority_queue<int> q;

    std::vector<int*> elements;
    for (unsigned i = 0; i < 10; i++) {
        int* element = new int();
        q.enqueue(element, 9-i);
        elements.push_back(element);
    }

    for (unsigned i = 0; i < 10; i++) {
        int* element = q.dequeue();
        REQUIRE(element == elements[9-i]);
    }
}



TEST_CASE("Single Reader Priority Queue ZigZag Dequeue Order", "[queue]") {
    using namespace clockwork;

    single_reader_priority_queue<int> q;


    std::vector<int> priorities = { 10, 0, 5, 8, 3, 7, 11, 1};
    std::unordered_map<int, int*> elems;

    for (int &priority : priorities) {
        int* element = new int();
        q.enqueue(element, priority);
        elems[priority] = element;
    }

    std::sort(priorities.begin(), priorities.end());

    for (int &priority : priorities) {
        int* element = q.dequeue();
        REQUIRE(element == elems[priority]);
    }
}

TEST_CASE("Single Reader Priority Queue Multiple Identical Priorities", "[queue]") {
    using namespace clockwork;

    single_reader_priority_queue<int> q;

    std::vector<int*> low;
    std::vector<int*> high;

    for (unsigned i = 0; i < 10; i++) {
        int* elow = new int();
        q.enqueue(elow, 10);
        low.push_back(elow);

        int* ehigh = new int();
        q.enqueue(ehigh, 20);
        high.push_back(ehigh);
    }

    for (unsigned i = 0; i < low.size(); i++) {
        int* e = q.dequeue();
        REQUIRE(std::find(low.begin(), low.end(), e) != low.end());
        REQUIRE(std::find(high.begin(), high.end(), e) == high.end());
    }

    for (unsigned i = 0; i < high.size(); i++) {
        int* e = q.dequeue();
        REQUIRE(std::find(low.begin(), low.end(), e) == low.end());
        REQUIRE(std::find(high.begin(), high.end(), e) != high.end());
    }
}

TEST_CASE("Single Reader Priority Queue Eligible Time", "[queue]") {
    using namespace clockwork;

    single_reader_priority_queue<int> q;

    uint64_t now = clockwork::util::now();
    std::vector<int*> elements;
    std::vector<uint64_t> priorities;


    for (int i = -10; i < 10; i++) {
        int* e = new int();
        uint64_t priority = now + i * 200000000L; // 200ms * i
        q.enqueue(e, priority);
        elements.push_back(e);
        priorities.push_back(priority);
    }

    for (int i = 0; i < elements.size(); i++) {
        int* e = q.dequeue();
        REQUIRE(e == elements[i]);
        REQUIRE(clockwork::util::now() >= priorities[i]);
    }
}

class ShutdownSignaller2 {
public:
    std::atomic_bool signalled_shutdown;
    std::thread thread;
    clockwork::single_reader_priority_queue<int> &q;
    ShutdownSignaller2(clockwork::single_reader_priority_queue<int> &q) : 
            q(q), signalled_shutdown(false), thread(&ShutdownSignaller2::run, this) {
    }
    ~ShutdownSignaller2() {
        while (!signalled_shutdown);
        thread.join();
    }
    void run() {
        using namespace clockwork;
        uint64_t start = util::now();
        while (util::now() < start + 100000000UL) {
            usleep(1000);
        }
        signalled_shutdown = true;
        q.shutdown();
    }
};

TEST_CASE("Single Reader Priority Queue Blocking Dequeue", "[queue] [shutdown]") {
    using namespace clockwork;

    single_reader_priority_queue<int> q;
    ShutdownSignaller2 s(q);

    int* e = q.dequeue();

    {
        INFO("Blocking dequeue returned before shutdown signaller completed");
        REQUIRE(s.signalled_shutdown);
    }

    {
        INFO("Blocking dequeue should have returned a nullptr");
        REQUIRE(e == nullptr);
    }
}

class Dequeuer2 {
public:
    std::thread thread;
    clockwork::single_reader_priority_queue<int> &q;
    std::atomic_bool complete;
    std::vector<int*> dequeued;
    Dequeuer2(clockwork::single_reader_priority_queue<int> &q) : 
            q(q), complete(false), thread(&Dequeuer2::run, this) {
    }
    ~Dequeuer2() {
        while (!complete);
        thread.join();
    }
    void run() {
        using namespace clockwork;
        uint64_t start = util::now();
        while (true) {
            int* element = q.dequeue();
            if (element == nullptr) {
                break;
            } else {
                dequeued.push_back(element);
            }
        }
        complete = true;
    }
};

TEST_CASE("Single Reader Priority Queue Shutdown", "[queue] [shutdown]") {
    using namespace clockwork;

    single_reader_priority_queue<int> q;

    Dequeuer2 d(q);

    std::vector<int*> elements;
    for (unsigned i = 0; i < 10; i++) {
        int* element = new int();
        q.enqueue(element, i);
        elements.push_back(element);
    }

    uint64_t now = util::now();
    while (util::now() < now + 100000000) { // Wait 100ms
        INFO("Dequeuer thread completed before it was signalled");
        REQUIRE(!d.complete.load());
        usleep(1000);
    }

    INFO("Dequeuer thread dequeued " << d.dequeued.size() << " elements");
    REQUIRE(d.dequeued.size() == 10);

    INFO("Dequeuer thread completed before it was signalled");
    REQUIRE(!d.complete.load());

    q.shutdown();

    now = util::now();
    while (util::now() < now + 1000000000) { // Max 1s
        if (d.complete.load()) {
            break;
        }
        usleep(1000);
    }
    INFO("Dequeuer thread never unblocked");
    REQUIRE(d.complete.load());
}

TEST_CASE("Single Reader Priority Queue Enqueue after Shutdown", "[queue] [shutdown]") {
    using namespace clockwork;

    single_reader_priority_queue<int> q;

    REQUIRE(q.enqueue(new int(), 0));

    q.shutdown();

    INFO("Should not be able to enqueue new elements after shutdown");
    REQUIRE(!q.enqueue(new int(), 0));
}

TEST_CASE("Single Reader Priority Queue Drain after Shutdown", "[queue] [shutdown]") {
    using namespace clockwork;

    single_reader_priority_queue<int> q;

    uint64_t now = util::now();

    REQUIRE(q.enqueue(new int(), now + 1000000000UL));
    REQUIRE(q.enqueue(new int(), now + 1000000000UL));
    REQUIRE(q.enqueue(new int(), now + 1000000000UL));

    q.shutdown();

    int* dequeued = q.dequeue();
    INFO("Shouldn't be able to dequeue after queue shutdown");
    REQUIRE(dequeued == nullptr);


    std::vector<int*> drained = q.drain();

    INFO("Unable to drain pending elements from queue after shutdown");
    REQUIRE(drained.size() == 3);
}