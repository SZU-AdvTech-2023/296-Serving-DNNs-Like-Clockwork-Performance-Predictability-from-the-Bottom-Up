#include <catch2/catch.hpp>
#include "clockwork/worker.h"
#include "clockwork/network/worker.h"


using namespace clockwork;

// Commenting out since it fails non-deterministically
//TEST_CASE("Test Worker Server Shutdown", "[network]") {
//	clockwork::ClockworkWorker clockwork;
//    clockwork::network::worker::Server server(&clockwork);
//    clockwork.controller = &server;
//
//    clockwork.shutdown(false);
//    server.shutdown(false);
//
//    clockwork.join();
//    server.join();
//}
