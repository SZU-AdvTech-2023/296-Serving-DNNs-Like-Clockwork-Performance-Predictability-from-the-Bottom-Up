#include <catch2/catch.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <queue>
#include "clockwork/dummy/memory_dummy.h"

TEST_CASE("Simple Page alloc and free Dummy", "[cache] [dummy]") {

    using namespace clockwork;
    
    size_t total_size = 100;
    size_t page_size = 100;
    size_t total_pages  = total_size/page_size;
    size_t n_free_pages = total_pages;
    
    PageCacheDummy* cache = new PageCacheDummy(total_size, page_size);

    REQUIRE( cache->total_pages == total_pages);
    REQUIRE( cache->n_free_pages == n_free_pages);

    bool alloc1 = cache->alloc(1);
    n_free_pages--;
    
    REQUIRE( alloc1);
    REQUIRE( cache->total_pages == total_pages);
    REQUIRE( cache->n_free_pages == n_free_pages);

    cache->free(1);
    n_free_pages++;
    
    REQUIRE( cache->total_pages == total_pages);
    REQUIRE( cache->n_free_pages == n_free_pages);

    bool alloc2 = cache->alloc(total_pages);
    n_free_pages-=total_pages;
    
    REQUIRE( alloc2);
    REQUIRE( cache->total_pages == total_pages);
    REQUIRE( cache->n_free_pages == n_free_pages);

    bool alloc3 = cache->alloc(1);
    
    REQUIRE( !alloc3);
    REQUIRE( cache->total_pages == total_pages);
    REQUIRE( cache->n_free_pages == n_free_pages);

    cache->free(total_pages);
    n_free_pages+=total_pages;
    
    REQUIRE( cache->total_pages == total_pages);
    REQUIRE( cache->n_free_pages == n_free_pages);

    cache->free(1);
    
    REQUIRE( cache->total_pages == total_pages);
    REQUIRE( cache->n_free_pages == n_free_pages);
}

TEST_CASE("Simple Page clear Dummy", "[cache] [dummy]") {

    using namespace clockwork;
    
    size_t total_size = 100;
    size_t page_size = 100;
    size_t total_pages  = total_size/page_size;
    size_t n_free_pages = total_pages;
    
    PageCacheDummy* cache = new PageCacheDummy(total_size, page_size);

    REQUIRE( cache->total_pages == total_pages);
    REQUIRE( cache->n_free_pages == n_free_pages);

    bool alloc = cache->alloc(total_pages);
    n_free_pages-=total_pages;
    
    REQUIRE( alloc);
    REQUIRE( cache->total_pages == total_pages);
    REQUIRE( cache->n_free_pages == n_free_pages);

    cache->clear();
    n_free_pages = total_pages;
    REQUIRE( cache->total_pages == total_pages);
    REQUIRE( cache->n_free_pages == n_free_pages);

    cache->free(1);
    
    REQUIRE( cache->total_pages == total_pages);
    REQUIRE( cache->n_free_pages == n_free_pages);
}

