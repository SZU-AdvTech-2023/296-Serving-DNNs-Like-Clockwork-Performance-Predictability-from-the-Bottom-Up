cmake_minimum_required(VERSION 3.4.1)

project(clockwork)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O2")
set(CXX "g++-8")
set(CC "gcc-8")
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Threads REQUIRED)
find_package(Boost COMPONENTS system filesystem REQUIRED)

# Introduce variables:
# * CMAKE_INSTALL_LIBDIR
# * CMAKE_INSTALL_BINDIR
# * CMAKE_INSTALL_INCLUDEDIR
include(GNUInstallDirs)

include_directories(
	include
	src
	external
	$ENV{TVM_HOME}/include
	$ENV{TVM_HOME}/3rdparty/dmlc-core/include
	$ENV{TVM_HOME}/3rdparty/dlpack/include
	/usr/local/cuda/include
	${Boost_INCLUDE_DIR}
	${CMAKE_CURRENT_BINARY_DIR}
)

link_directories(
	$ENV{TVM_HOME}/build
	$ENV{TVM_HOME}/3rdparty/dmlc-core/build
	/usr/local/cuda/lib64
	/usr/lib/x86_64-linux-gnu/nvidia/current
)


# Build a clockwork_proto lib containing the protobuf stuff
find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS proto/clockwork.proto)
add_library(clockwork_proto ${PROTO_HDRS} ${PROTO_SRCS})
target_link_libraries(clockwork_proto INTERFACE ${Protobuf_LIBRARIES})

add_library( clockwork
	external/lz4.c
	external/lz4hc.c
	src/clockwork/client.cpp
	src/clockwork/task.cpp
	src/clockwork/modeldef.cpp
	src/clockwork/common.cpp
	src/clockwork/cache.cpp
	src/clockwork/util.cpp
	src/clockwork/action.cpp
	src/clockwork/runtime.cpp
	src/clockwork/memory.cpp
	src/clockwork/worker.cpp
	src/clockwork/thread.cpp
	src/clockwork/model/memfile.cpp
	src/clockwork/model/model.cpp
	src/clockwork/model/batched.cpp
	src/clockwork/model/cuda.cpp
	src/clockwork/model/so.cpp
	src/clockwork/api/client_api.cpp
	src/clockwork/api/worker_api.cpp
	src/clockwork/network/network.cpp
	src/clockwork/network/client_api.cpp
	src/clockwork/network/worker.cpp
	src/clockwork/network/controller.cpp
	src/clockwork/controller/controller.cpp
	src/clockwork/controller/load_tracker.cpp
	src/clockwork/controller/direct_controller.cpp
	src/clockwork/controller/scheduler.cpp
	src/clockwork/controller/smart_scheduler.cpp
	src/clockwork/controller/concurrent_infer_and_load_scheduler.cpp
	src/clockwork/controller/infer5/load_tracker.cpp
	src/clockwork/controller/infer5/infer5_scheduler.cpp
	src/clockwork/config.cpp
	src/clockwork/network/client.cpp
	src/clockwork/workload/workload.cpp
	src/clockwork/telemetry/telemetry.cpp
    src/clockwork/dummy/memory_dummy.cpp
    src/clockwork/dummy/action_dummy.cpp
    src/clockwork/dummy/worker_dummy.cpp
    src/clockwork/dummy/network/worker_dummy.cpp

	$ENV{TVM_HOME}/src/runtime/meta_data.h

### No point maintaining the below any more
#	src/clockwork/alternatives/runtime_model.cpp
#	src/clockwork/alternatives/model_manager.cpp
#	src/clockwork/alternatives/threadpoolruntime.cpp
#	src/clockwork/alternatives/greedyruntime.cpp
#	src/clockwork/alternatives/worker.cpp
)


target_link_libraries( clockwork PUBLIC
	clockwork_proto
	cuda
	cudart
	tvm_runtime
	tbb
	nvidia-ml
	config++
	rt
	stdc++fs	
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_FILESYSTEM_LIBRARY}
)



# Converts from TVM models to clockwork models
include_directories(clockwork-convert)
add_executable (convert
	src/clockwork-convert/tvm_model.cpp
	src/clockwork-convert/tvm_abstract_model.cpp
	src/clockwork-convert/convert.cpp
)

target_link_libraries( convert
	clockwork
	clockwork_proto
	Threads::Threads
	dl
	cuda
	cudart
	tvm_runtime
	tbb
	nvidia-ml
	stdc++fs
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_FILESYSTEM_LIBRARY}
)

# cudafatbin is temporary / hacky 
add_executable (cudafatbin src/clockwork-convert/cudafatbin.cc )
target_link_libraries( cudafatbin
	clockwork
	clockwork_proto
	Threads::Threads
	dl
	cuda
	cudart
	tvm_runtime
	tbb
	nvidia-ml
	stdc++fs
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_FILESYSTEM_LIBRARY}
)

add_executable (check_model
	src/check_model.cpp
)

target_link_libraries( check_model
	clockwork
	clockwork_proto
	Threads::Threads
	dl
	cuda
	cudart
	tvm_runtime
	tbb
	nvidia-ml
	stdc++fs
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_FILESYSTEM_LIBRARY}
)


# Inflate telemetry files from binary to human-readable
add_executable (inflate src/inflate.cpp )
target_link_libraries( inflate
	clockwork
	clockwork_proto
	Threads::Threads
	dl
	cuda
	cudart
	tvm_runtime
	tbb
	nvidia-ml
	stdc++fs
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_FILESYSTEM_LIBRARY}
)


# All tests
include_directories(test)
add_executable (tests
	test/tests.cpp
	test/clockwork/test/util.cpp
	test/clockwork/test/actions.cpp
	test/clockwork/test/testtask.cpp
	test/clockwork/test/testaction.cpp
	test/clockwork/test/testworker.cpp
	test/clockwork/test/testcache.cpp
	test/clockwork/test/testmemory.cpp
	test/clockwork/test/testpriorityqueue.cpp
	test/clockwork/test/testclient.cpp
	test/clockwork/test/testtelemetry.cpp
	test/clockwork/test/testnetwork.cpp
	test/clockwork/test/testconfig.cpp
	test/clockwork/test/testutil.cpp
	test/clockwork/test/model/testmodel.cpp
	test/clockwork/test/model/testbatched.cpp
    test/clockwork/test_dummy/actions.cpp
    test/clockwork/test_dummy/testaction.cpp
    test/clockwork/test_dummy/testworker.cpp
    test/clockwork/test_dummy/testmemory.cpp
)
target_link_libraries( tests
	clockwork
	clockwork_proto
	Threads::Threads
	dl
	cuda
	cudart
	tvm_runtime
	tbb
	nvidia-ml
	stdc++fs
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_FILESYSTEM_LIBRARY}
)

# Like tests, but does profiling
add_executable ( profile
	profile/profile.cpp
	test/clockwork/test/util.cpp
	profile/clockwork/profile/check.cpp
	profile/clockwork/profile/compression.cpp
	profile/clockwork/profile/model/profilecuda.cpp
	profile/clockwork/profile/model/profilemodel.cpp
	
### No point maintaining the below any more
#	profile/clockwork/profile/alternatives/profilegreedy.cpp
)

target_link_libraries( profile
	clockwork
	clockwork_proto
	Threads::Threads
	dl
	cuda
	cudart
	tvm_runtime
	tbb
	nvidia-ml
	stdc++fs
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_FILESYSTEM_LIBRARY}
)


# Standalone clockwork worker
add_executable (worker src/worker.cpp )
target_link_libraries(
    worker
	clockwork
	clockwork_proto
    Threads::Threads
    dl
    cuda
    cudart
    tvm_runtime
    stdc++fs
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_FILESYSTEM_LIBRARY}
)

add_executable (worker_dummy src/worker_dummy.cpp )
target_link_libraries(
    worker_dummy
    clockwork
    clockwork_proto
    Threads::Threads
    dl
    cuda
    cudart
    tvm_runtime
    stdc++fs
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_FILESYSTEM_LIBRARY}
)


# Standalone clockwork controller 
add_executable (controller src/controller.cpp )
target_link_libraries(
    controller
	clockwork
	clockwork_proto
    Threads::Threads
    dl
    cuda
    cudart
    tvm_runtime
    stdc++fs
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_FILESYSTEM_LIBRARY}
)

# Standalone clockwork workload generating client
add_executable (client src/client.cpp )
target_link_libraries(
    client
	clockwork
	clockwork_proto
    Threads::Threads
    dl
    cuda
    cudart
    tvm_runtime
    stdc++fs
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_FILESYSTEM_LIBRARY}
)

# Simple Clockwork client demonstrating usage
add_executable (simpleclient src/simpleclient.cpp )
target_link_libraries(
    simpleclient
	clockwork
	clockwork_proto
    Threads::Threads
    dl
    cuda
    cudart
    tvm_runtime
    stdc++fs
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_FILESYSTEM_LIBRARY}
)

# Spams the client API
add_executable (netclient src/network/netclient.cpp )
target_link_libraries(
    netclient
	clockwork
	clockwork_proto
    Threads::Threads
    dl
    cuda
    cudart
    tvm_runtime
    stdc++fs
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_FILESYSTEM_LIBRARY}
)

# Echo server for netclient
add_executable (netcontroller src/network/netcontroller.cpp )
target_link_libraries(
    netcontroller
	clockwork
	clockwork_proto
    Threads::Threads
    dl
    cuda
    cudart
    tvm_runtime
    stdc++fs
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_FILESYSTEM_LIBRARY}
)

