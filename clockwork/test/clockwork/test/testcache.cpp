#include <catch2/catch.hpp>

#include <cstdlib>

#include "clockwork/memory.h"
#include "clockwork/cache.h"

TEST_CASE("Create Page Cache with bad sizes", "[cache]") {

    using namespace clockwork;
    
    size_t total_size = 100;
    size_t page_size = 11;
    void* baseptr = malloc(total_size);
    
    PageCache* cache;
    REQUIRE_THROWS(cache = new PageCache(static_cast<char*>(baseptr), total_size, page_size));
}

TEST_CASE("Evictions disabled", "[cache]") {
    using namespace clockwork;
    
    size_t total_size = 100;
    size_t page_size = 10;
    void* baseptr = malloc(total_size);
    bool allow_evictions = false;

    PageCache* cache = new PageCache(static_cast<char*>(baseptr), total_size, page_size, allow_evictions);


    std::shared_ptr<Allocation> alloc1 = cache->alloc(5, []{});
    REQUIRE(alloc1 != nullptr);
    cache->unlock(alloc1);

    std::shared_ptr<Allocation> alloc2 = cache->alloc(3, []{});
    REQUIRE(alloc2 != nullptr);
    cache->unlock(alloc2);

    std::shared_ptr<Allocation> alloc3 = cache->alloc(3, []{});
    REQUIRE(alloc3 == nullptr);

    std::shared_ptr<Allocation> alloc4 = cache->alloc(2, []{});
    REQUIRE(alloc2 != nullptr);
    cache->unlock(alloc4);

    std::shared_ptr<Allocation> alloc5 = cache->alloc(1, []{});
    REQUIRE(alloc5 == nullptr);

    cache->free(alloc1);

    std::shared_ptr<Allocation> alloc6 = cache->alloc(5, []{});
    REQUIRE(alloc6 != nullptr);
}

TEST_CASE("Cache lock and unlock", "[cache]") {

    using namespace clockwork;
    
    size_t total_size = 100;
    size_t page_size = 100;
    void* baseptr = malloc(total_size);
    
    PageCache* cache = new PageCache(static_cast<char*>(baseptr), total_size, page_size);

    REQUIRE( !cache->freePages.isEmpty() );
    REQUIRE( cache->lockedAllocations.isEmpty() );
    REQUIRE( cache->unlockedAllocations.isEmpty() );



    std::shared_ptr<Allocation> alloc1 = cache->alloc(1, []{});
    REQUIRE( alloc1 != nullptr);
    REQUIRE( alloc1->usage_count == 1 );
    REQUIRE( !cache->lockedAllocations.isEmpty() );
    REQUIRE( cache->unlockedAllocations.isEmpty() );

    REQUIRE_NOTHROW( cache->unlock(alloc1) );
    REQUIRE( alloc1->usage_count == 0 );
    REQUIRE( cache->lockedAllocations.isEmpty() );
    REQUIRE( !cache->unlockedAllocations.isEmpty() );

    REQUIRE( cache->trylock(alloc1) );
    REQUIRE( alloc1->usage_count == 1 );
    REQUIRE( !cache->lockedAllocations.isEmpty() );
    REQUIRE( cache->unlockedAllocations.isEmpty() );

    REQUIRE( cache->trylock(alloc1) );
    REQUIRE( alloc1->usage_count == 2 );
    REQUIRE( !cache->lockedAllocations.isEmpty() );
    REQUIRE( cache->unlockedAllocations.isEmpty() );  

    REQUIRE( cache->trylock(alloc1) );
    REQUIRE( alloc1->usage_count == 3 );
    REQUIRE( !cache->lockedAllocations.isEmpty() );
    REQUIRE( cache->unlockedAllocations.isEmpty() );

    REQUIRE_NOTHROW( cache->unlock(alloc1) );
    REQUIRE( alloc1->usage_count == 2 );
    REQUIRE( !cache->lockedAllocations.isEmpty() );
    REQUIRE( cache->unlockedAllocations.isEmpty() );  

    REQUIRE_NOTHROW( cache->unlock(alloc1) );
    REQUIRE( alloc1->usage_count == 1 );
    REQUIRE( !cache->lockedAllocations.isEmpty() );
    REQUIRE( cache->unlockedAllocations.isEmpty() );  

    REQUIRE_NOTHROW( cache->unlock(alloc1) );
    REQUIRE( alloc1->usage_count == 0 );
    REQUIRE( cache->lockedAllocations.isEmpty() );
    REQUIRE( !cache->unlockedAllocations.isEmpty() );  
}

TEST_CASE("Simple Page alloc", "[cache]") {

    using namespace clockwork;
    
    size_t total_size = 100;
    size_t page_size = 100;
    void* baseptr = malloc(total_size);
    
    PageCache* cache = new PageCache(static_cast<char*>(baseptr), total_size, page_size);

    REQUIRE( !cache->freePages.isEmpty() );
    REQUIRE( cache->lockedAllocations.isEmpty() );
    REQUIRE( cache->unlockedAllocations.isEmpty() );


    std::shared_ptr<Allocation> alloc1 = cache->alloc(1, []{});
    REQUIRE( alloc1 != nullptr);
    REQUIRE( alloc1->evicted == false );
    REQUIRE( alloc1->usage_count == 1 );
    REQUIRE( alloc1->pages.size() == 1 );
    REQUIRE( cache->freePages.isEmpty() );
    REQUIRE( !cache->lockedAllocations.isEmpty() );
    REQUIRE( cache->unlockedAllocations.isEmpty() );

    std::shared_ptr<Allocation> alloc2 = nullptr;
    alloc2 = cache->alloc(1, []{});
    REQUIRE(alloc2 == nullptr);
    REQUIRE( alloc1->evicted == false );
    REQUIRE( alloc1->usage_count == 1 );
    REQUIRE( alloc1->pages.size() == 1 );
    REQUIRE( cache->freePages.isEmpty() );
    REQUIRE( !cache->lockedAllocations.isEmpty() );
    REQUIRE( cache->unlockedAllocations.isEmpty() );    
}

TEST_CASE("Alloc too much", "[cache]") {

    using namespace clockwork;
    
    size_t total_size = 100;
    size_t page_size = 100;
    void* baseptr = malloc(total_size);
    
    PageCache* cache = new PageCache(static_cast<char*>(baseptr), total_size, page_size);

    REQUIRE( !cache->freePages.isEmpty() );
    REQUIRE( cache->lockedAllocations.isEmpty() );
    REQUIRE( cache->unlockedAllocations.isEmpty() );


    std::shared_ptr<Allocation> alloc1 = nullptr;
    REQUIRE(cache->alloc(2, []{}) == nullptr);
    REQUIRE( alloc1 == nullptr);
    REQUIRE( !cache->freePages.isEmpty() );
    REQUIRE( cache->lockedAllocations.isEmpty() );
    REQUIRE( cache->unlockedAllocations.isEmpty() );  
}

TEST_CASE("Simple Page eviction", "[cache]") {

    using namespace clockwork;
    
    size_t total_size = 100;
    size_t page_size = 100;
    void* baseptr = malloc(total_size);
    
    PageCache* cache = new PageCache(static_cast<char*>(baseptr), total_size, page_size);

    REQUIRE( !cache->freePages.isEmpty() );
    REQUIRE( cache->lockedAllocations.isEmpty() );
    REQUIRE( cache->unlockedAllocations.isEmpty() );

    int evictionCount = 0;
    std::function<void(void)> callback1 = [&evictionCount]{ evictionCount++; };

    std::shared_ptr<Allocation> alloc1 = cache->alloc(1, callback1);
    REQUIRE( alloc1 != nullptr);
    REQUIRE( alloc1->evicted == false );

    std::shared_ptr<Allocation> alloc2 = nullptr;
    REQUIRE(cache->alloc(1, []{}) == nullptr);
    REQUIRE (alloc2 == nullptr);
    REQUIRE( alloc1->evicted == false );

    REQUIRE_NOTHROW(cache->unlock(alloc1));

    std::shared_ptr<Allocation> alloc3 = cache->alloc(1, []{});
    REQUIRE (alloc3 != nullptr);
    REQUIRE( alloc3->evicted == false );
    REQUIRE( alloc1->evicted == true );
    REQUIRE(evictionCount == 1);
    
}

TEST_CASE("Simple Page free", "[cache]") {

    using namespace clockwork;
    
    size_t total_size = 100;
    size_t page_size = 100;
    void* baseptr = malloc(total_size);
    
    PageCache* cache = new PageCache(static_cast<char*>(baseptr), total_size, page_size);

    REQUIRE( !cache->freePages.isEmpty() );
    REQUIRE( cache->lockedAllocations.isEmpty() );
    REQUIRE( cache->unlockedAllocations.isEmpty() );

    int evictionCount = 0;
    std::function<void(void)> callback1 = [&evictionCount]{ evictionCount++; };

    std::shared_ptr<Allocation> alloc1 = cache->alloc(1, callback1);
    REQUIRE( alloc1 != nullptr);
    REQUIRE( alloc1->evicted == false );
    REQUIRE( cache->freePages.isEmpty() );
    REQUIRE( !cache->lockedAllocations.isEmpty() );
    REQUIRE( cache->unlockedAllocations.isEmpty() );

    REQUIRE_NOTHROW(cache->unlock(alloc1));
    REQUIRE( alloc1->evicted == false );
    REQUIRE( cache->freePages.isEmpty() );
    REQUIRE( cache->lockedAllocations.isEmpty() );
    REQUIRE( !cache->unlockedAllocations.isEmpty() );

    REQUIRE_NOTHROW(cache->free(alloc1));
    REQUIRE( alloc1->evicted == true );
    REQUIRE( !cache->freePages.isEmpty() );
    REQUIRE( cache->lockedAllocations.isEmpty() );
    REQUIRE( cache->unlockedAllocations.isEmpty() );
    REQUIRE(evictionCount == 1);
    
}

TEST_CASE("LRU page eviction", "[cache]") {

    using namespace clockwork;
    
    size_t total_size = 100;
    size_t page_size = 10;
    void* baseptr = malloc(total_size);
    
    PageCache* cache = new PageCache(static_cast<char*>(baseptr), total_size, page_size);

    REQUIRE( !cache->freePages.isEmpty() );
    REQUIRE( cache->lockedAllocations.isEmpty() );
    REQUIRE( cache->unlockedAllocations.isEmpty() );


    std::shared_ptr<Allocation> alloc1 = cache->alloc(5, []{});
    REQUIRE( alloc1 != nullptr);
    
    std::shared_ptr<Allocation> alloc2 = cache->alloc(5, []{});
    REQUIRE( alloc2 != nullptr);

    REQUIRE_NOTHROW( cache->unlock(alloc2) );
    REQUIRE_NOTHROW( cache->unlock(alloc1) );
    
    std::shared_ptr<Allocation> alloc3 = cache->alloc(1, []{});
    REQUIRE( alloc3 != nullptr);
    REQUIRE( alloc1->evicted == false );
    REQUIRE( alloc2->evicted == true );
    REQUIRE( alloc3->evicted == false );
    REQUIRE( !cache->freePages.isEmpty() );

    REQUIRE_NOTHROW( cache->unlock(alloc3) );

    std::shared_ptr<Allocation> alloc4 = cache->alloc(4, []{});
    REQUIRE( alloc4 != nullptr);
    REQUIRE( alloc1->evicted == false );
    REQUIRE( alloc2->evicted == true );
    REQUIRE( alloc3->evicted == false );
    REQUIRE( alloc4->evicted == false );
    REQUIRE( cache->freePages.isEmpty() );

    REQUIRE( cache->trylock(alloc1) );

    std::shared_ptr<Allocation> alloc5 = cache->alloc(1, []{});
    REQUIRE( alloc5 != nullptr);
    REQUIRE( alloc1->evicted == false );
    REQUIRE( alloc2->evicted == true );
    REQUIRE( alloc3->evicted == true );
    REQUIRE( alloc4->evicted == false );
    REQUIRE( alloc5->evicted == false );
    REQUIRE( cache->freePages.isEmpty() );
    
}

TEST_CASE("Linked List AddRemove", "[linkedlist]") {

    using namespace clockwork;

    LinkedList<int*> list;

    REQUIRE( list.isEmpty() );
    REQUIRE( list.popHead() == nullptr );
    REQUIRE( list.popTail() == nullptr );
    REQUIRE( list.remove(nullptr) == false );
    REQUIRE( list.head == nullptr );
    REQUIRE( list.tail == nullptr );

    int* a = new int();
    LinkedListElement<int*>* e = list.pushBack(a);
    REQUIRE( !list.isEmpty() );
    REQUIRE( e != nullptr );
    REQUIRE( e->container == &list );
    REQUIRE( e->data == a );
    REQUIRE( list.head == e );
    REQUIRE( list.tail == e );

    REQUIRE( list.remove(e) );
    REQUIRE( list.head == nullptr );
    REQUIRE( list.tail == nullptr );
}

TEST_CASE("Linked List MultiAdd", "[linkedlist]") {

    using namespace clockwork;

    LinkedList<int*> list;
    LinkedListElement<int*>* e1 = list.pushBack(nullptr);
    REQUIRE( list.head == e1 );
    REQUIRE( e1->prev == nullptr );
    REQUIRE( e1->next == nullptr );
    REQUIRE( list.tail == e1 );

    LinkedListElement<int*>* e2 = list.pushBack(nullptr);
    REQUIRE( list.head == e1 );
    REQUIRE( e1->prev == nullptr );
    REQUIRE( e1->next == e2 );
    REQUIRE( e2->prev == e1 );
    REQUIRE( e2->next == nullptr );
    REQUIRE( list.tail == e2 );

    LinkedListElement<int*>* e3 = list.pushBack(nullptr);

    REQUIRE( list.head == e1 );
    REQUIRE( e1->prev == nullptr );
    REQUIRE( e1->next == e2 );
    REQUIRE( e2->prev == e1 );
    REQUIRE( e2->next == e3 );
    REQUIRE( e3->prev == e2 );
    REQUIRE( e3->next == nullptr );
    REQUIRE( list.tail == e3 );
}

TEST_CASE("Linked List PopHead", "[linkedlist]") {

    using namespace clockwork;

    LinkedList<int*> list;

    int* x = new int();
    LinkedListElement<int*>* e1 = list.pushBack(x);

    REQUIRE( list.head == e1 );
    REQUIRE( e1->prev == nullptr );
    REQUIRE( e1->next == nullptr );
    REQUIRE( list.tail == e1 );

    REQUIRE( list.popHead() == x );

    REQUIRE( list.head == nullptr );
    REQUIRE( list.tail == nullptr );
}

TEST_CASE("Linked List PopHead2", "[linkedlist]") {

    using namespace clockwork;

    LinkedList<int*> list;

    int* x = new int();
    int* y = new int();
    LinkedListElement<int*>* e1 = list.pushBack(x);
    LinkedListElement<int*>* e2 = list.pushBack(y);

    REQUIRE( x != y );
    REQUIRE( list.popHead() == x );

    REQUIRE( list.head == e2 );
    REQUIRE( e2->prev == nullptr );
    REQUIRE( e2->next == nullptr );
    REQUIRE( list.tail == e2 );

    REQUIRE( list.popHead() == y );

    REQUIRE( list.head == nullptr );
    REQUIRE( list.tail == nullptr );
}

TEST_CASE("Linked List PopTail", "[linkedlist]") {

    using namespace clockwork;

    LinkedList<int*> list;

    int* x = new int();
    LinkedListElement<int*>* e1 = list.pushBack(x);

    REQUIRE( list.head == e1 );
    REQUIRE( e1->prev == nullptr );
    REQUIRE( e1->next == nullptr );
    REQUIRE( list.tail == e1 );

    REQUIRE( list.popTail() == x );

    REQUIRE( list.head == nullptr );
    REQUIRE( list.tail == nullptr );
}

TEST_CASE("Linked List PopTail2", "[linkedlist]") {

    using namespace clockwork;

    LinkedList<int*> list;

    int* x = new int();
    int* y = new int();
    LinkedListElement<int*>* e1 = list.pushBack(x);
    LinkedListElement<int*>* e2 = list.pushBack(y);

    REQUIRE( x != y );
    REQUIRE( list.popTail() == y );

    REQUIRE( list.head == e1 );
    REQUIRE( e1->prev == nullptr );
    REQUIRE( e1->next == nullptr );
    REQUIRE( list.tail == e1 );

    REQUIRE( list.popTail() == x );

    REQUIRE( list.head == nullptr );
    REQUIRE( list.tail == nullptr );
}

TEST_CASE("Linked List RemoveMiddle", "[linkedlist]") {

    using namespace clockwork;

    LinkedList<int*> list;
    LinkedListElement<int*>* e1 = list.pushBack(nullptr);
    LinkedListElement<int*>* e2 = list.pushBack(nullptr);
    LinkedListElement<int*>* e3 = list.pushBack(nullptr);

    REQUIRE( list.head == e1 );
    REQUIRE( e1->prev == nullptr );
    REQUIRE( e1->next == e2 );
    REQUIRE( e2->prev == e1 );
    REQUIRE( e2->next == e3 );
    REQUIRE( e3->prev == e2 );
    REQUIRE( e3->next == nullptr );
    REQUIRE( list.tail == e3 );


    REQUIRE( list.remove(e2) );


    REQUIRE( list.head == e1 );
    REQUIRE( e1->prev == nullptr );
    REQUIRE( e1->next == e3 );
    REQUIRE( e3->prev == e1 );
    REQUIRE( e3->next == nullptr );
    REQUIRE( list.tail == e3 );

}


TEST_CASE("Multiple Baseptrs", "[cache]") {

    using namespace clockwork;

    size_t ptr_size = 100;

    char* baseptr_1 = static_cast<char*>(malloc(ptr_size));
    char* baseptr_2 = static_cast<char*>(malloc(ptr_size));

    std::vector<std::pair<char*, size_t>> baseptrs = {{{baseptr_1, ptr_size}, {baseptr_2, ptr_size}}};
    
    size_t page_size = 100;
    
    PageCache* cache = new PageCache(baseptrs, ptr_size * baseptrs.size(), page_size);

    REQUIRE( !cache->freePages.isEmpty() );
    REQUIRE(cache->freePages.size() == 2);
    REQUIRE( cache->lockedAllocations.isEmpty() );
    REQUIRE( cache->unlockedAllocations.isEmpty() );


    std::shared_ptr<Allocation> alloc1 = cache->alloc(2, []{});
    REQUIRE( alloc1 != nullptr);
    REQUIRE( alloc1->evicted == false );
    REQUIRE( alloc1->usage_count == 1 );
    REQUIRE( alloc1->pages.size() == 2 );
    REQUIRE( cache->freePages.isEmpty() );
    REQUIRE( !cache->lockedAllocations.isEmpty() );
    REQUIRE( cache->unlockedAllocations.isEmpty() );
    REQUIRE( alloc1->page_pointers[0] == baseptr_1 );
    REQUIRE( alloc1->page_pointers[1] == baseptr_2 );

    std::shared_ptr<Allocation> alloc2 = nullptr;
    alloc2 = cache->alloc(1, []{});
    REQUIRE(alloc2 == nullptr);
    REQUIRE( alloc1->evicted == false );
    REQUIRE( alloc1->usage_count == 1 );
    REQUIRE( alloc1->pages.size() == 2 );
    REQUIRE( cache->freePages.isEmpty() );
    REQUIRE( !cache->lockedAllocations.isEmpty() );
    REQUIRE( cache->unlockedAllocations.isEmpty() );

    cache->unlock(alloc1);
    cache->free(alloc1);
    REQUIRE(cache->freePages.size() == 2);

    free(baseptr_1);
    free(baseptr_2);
}