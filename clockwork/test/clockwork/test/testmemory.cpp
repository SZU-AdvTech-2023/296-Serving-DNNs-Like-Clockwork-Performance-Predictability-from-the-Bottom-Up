#include <catch2/catch.hpp>

#include <cstdlib>
#include <queue>

#include "clockwork/memory.h"

TEST_CASE("MemoryPool Alloc", "[mempool]") {

    using namespace clockwork;

    size_t size = 1000;
    char* baseptr = static_cast<char*>(malloc(size));
    MemoryPool pool(baseptr, size);

    auto alloc1 = pool.alloc(1000);
    REQUIRE(alloc1 != nullptr);
    REQUIRE(alloc1 == baseptr);
}

TEST_CASE("MemoryPool Free", "[mempool]") {

    using namespace clockwork;

    size_t size = 1000;
    char* baseptr = static_cast<char*>(malloc(size));
    MemoryPool pool(baseptr, size);

    auto alloc = pool.alloc(1000);
    pool.free(alloc);

    REQUIRE(pool.alloc(1000) != nullptr);
}

TEST_CASE("MemoryPool Free Multiple", "[mempool]") {

    using namespace clockwork;

    size_t size = 1000;
    char* baseptr = static_cast<char*>(malloc(size));
    MemoryPool pool(baseptr, size);

    auto alloc = pool.alloc(1000);
    pool.free(alloc);

    alloc = pool.alloc(1000);
    REQUIRE(alloc != nullptr);
    REQUIRE(pool.alloc(1000) == nullptr);

    pool.free(alloc);
    alloc = pool.alloc(1000);
    REQUIRE(alloc != nullptr);
    REQUIRE(pool.alloc(1000) == nullptr);
}

TEST_CASE("MemoryPool Exhaust", "[mempool]") {

    using namespace clockwork;

    size_t size = 1000;
    char* baseptr = static_cast<char*>(malloc(size));
    MemoryPool pool(baseptr, size);

    auto alloc1 = pool.alloc(1000);

    REQUIRE(pool.alloc(1) == nullptr);
    REQUIRE(pool.alloc(1) == nullptr);    
}

TEST_CASE("MemoryPool Multiple Alloc", "[mempool]") {

    using namespace clockwork;

    size_t size = 1000;
    char* baseptr = static_cast<char*>(malloc(size));
    MemoryPool pool(baseptr, size);

    for (unsigned i = 0; i < 10; i++) {
        auto alloc = pool.alloc(100);
        REQUIRE(alloc != nullptr);
    }
    REQUIRE(pool.alloc(1) == nullptr);
}

TEST_CASE("MemoryPool Multiple Alloc 2", "[mempool]") {

    using namespace clockwork;

    size_t size = 1000;
    char* baseptr = static_cast<char*>(malloc(size));
    MemoryPool pool(baseptr, size);

    std::queue<char*> allocs;

    for (unsigned i = 0; i < 5; i++) {
        auto alloc = pool.alloc(7);
        REQUIRE(alloc != nullptr);
        allocs.push(alloc);
    }

    for (unsigned i = 0; i < 10000; i++) {
        pool.free(allocs.front());
        allocs.pop();
        pool.free(allocs.front());
        allocs.pop();
        auto alloc = pool.alloc(7);
        REQUIRE(alloc != nullptr);
        allocs.push(alloc);
        alloc = pool.alloc(7);
        REQUIRE(alloc != nullptr);
        allocs.push(alloc);
    }

}

TEST_CASE("MemoryPool Indivisible Limit", "[mempool]") {

    using namespace clockwork;

    size_t size = 1000;
    char* baseptr = static_cast<char*>(malloc(size));
    MemoryPool pool(baseptr, size);

    auto alloc1 = pool.alloc(800);
    REQUIRE(alloc1 != nullptr);

    REQUIRE(pool.alloc(300) == nullptr);

    auto alloc2 = pool.alloc(200);
    REQUIRE(alloc2 != nullptr);
}

TEST_CASE("MemoryPool Wrap", "[mempool]") {

    using namespace clockwork;

    size_t size = 1000;
    char* baseptr = static_cast<char*>(malloc(size));
    MemoryPool pool(baseptr, size);

    auto alloc1 = pool.alloc(700);
    REQUIRE(alloc1 != nullptr);

    REQUIRE(pool.alloc(500) == nullptr);

    auto alloc2 = pool.alloc(100);
    REQUIRE(alloc2 != nullptr);

    pool.free(alloc1);

    REQUIRE(pool.alloc(800) == nullptr);

    auto alloc3 = pool.alloc(700);
    REQUIRE(alloc3 != nullptr);
}

TEST_CASE("MemoryPool Wrap 2", "[mempool]") {

    using namespace clockwork;

    size_t size = 1000;
    char* baseptr = static_cast<char*>(malloc(size));
    MemoryPool pool(baseptr, size);

    auto alloc1 = pool.alloc(700);
    REQUIRE(alloc1 != nullptr);

    auto alloc2 = pool.alloc(300);
    REQUIRE(alloc2 != nullptr);

    pool.free(alloc1);

    auto alloc3 = pool.alloc(300);
    REQUIRE(alloc3 != nullptr);

    auto alloc4 = pool.alloc(400);
    REQUIRE(alloc4 != nullptr);
}

TEST_CASE("MemoryPool Inverted Free Order", "[mempool]") {

    using namespace clockwork;

    size_t size = 1000;
    char* baseptr = static_cast<char*>(malloc(size));
    MemoryPool pool(baseptr, size);

    auto alloc1 = pool.alloc(500);
    REQUIRE(alloc1 != nullptr);

    auto alloc2 = pool.alloc(500);
    REQUIRE(alloc2 != nullptr);

    REQUIRE(pool.alloc(500) == nullptr);

    pool.free(alloc2);

    REQUIRE(pool.alloc(500) == nullptr);

    pool.free(alloc1);

    REQUIRE(pool.alloc(500) != nullptr);
    REQUIRE(pool.alloc(500) != nullptr);
    REQUIRE(pool.alloc(500) == nullptr);
}

TEST_CASE("MemoryPool Staggered Free Order", "[mempool]") {

    using namespace clockwork;

    size_t size = 1000;
    char* baseptr = static_cast<char*>(malloc(size));
    MemoryPool pool(baseptr, size);

    auto alloc1 = pool.alloc(300);
    REQUIRE(alloc1 != nullptr);

    auto alloc2 = pool.alloc(300);
    REQUIRE(alloc2 != nullptr);

    REQUIRE(pool.alloc(500) == nullptr);

    pool.free(alloc2);

    REQUIRE(pool.alloc(500) == nullptr);

    auto alloc3 = pool.alloc(100);
    REQUIRE(alloc3 != nullptr);
    auto alloc4 = pool.alloc(100);
    REQUIRE(alloc4 != nullptr);
    auto alloc5 = pool.alloc(100);
    REQUIRE(alloc5 != nullptr);

    REQUIRE(pool.alloc(500) == nullptr);

    pool.free(alloc1);

    REQUIRE(pool.alloc(700) == nullptr);
    REQUIRE(pool.alloc(600) != nullptr);
    REQUIRE(pool.alloc(100) == nullptr);
}

TEST_CASE("MemoryPool Large alloc aligns with end of mempool Part 1", "[mempool]") {

    using namespace clockwork;

    size_t size = 1000;
    char* baseptr = static_cast<char*>(malloc(size));
    MemoryPool pool(baseptr, size);

    auto alloc1 = pool.alloc(300);
    REQUIRE(alloc1 != nullptr);

    auto alloc2 = pool.alloc(400);
    REQUIRE(alloc2 != nullptr);

    pool.free(alloc1);

    REQUIRE(pool.alloc(600) != nullptr);
}

TEST_CASE("MemoryPool Large alloc aligns with end of mempool Part 2", "[mempool]") {

    using namespace clockwork;

    size_t size = 1000;
    char* baseptr = static_cast<char*>(malloc(size));
    MemoryPool pool(baseptr, size);

    auto alloc1 = pool.alloc(300);
    REQUIRE(alloc1 != nullptr);

    for (unsigned i = 0; i < 4; i++) {
        auto alloc2 = pool.alloc(100);
        REQUIRE(alloc2 != nullptr);
    }

    pool.free(alloc1);

    REQUIRE(pool.alloc(600) == nullptr);
}

TEST_CASE("CUDAMemoryPool", "[mempool]") {
    using namespace clockwork;

    CUDAMemoryPool* pool = CUDAMemoryPool::create(1000, GPU_ID_0);
    delete pool;
}

TEST_CASE("CUDAHostMemoryPool", "[mempool]") {
    using namespace clockwork;

    CUDAHostMemoryPool* pool = CUDAHostMemoryPool::create(1000);
    delete pool;
}