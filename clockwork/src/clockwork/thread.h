#ifndef _CLOCKWORK_THREAD_H_
#define _CLOCKWORK_THREAD_H_

#include <vector>
#include <thread>

namespace clockwork {
namespace threading {

// Initializes a clockwork process
void initProcess();

// Initializes a network thread
void initNetworkThread(std::thread &thread);

// Initializes a logger thread
void initLoggerThread(std::thread &thread);

// Initializes a GPU thread
void initGPUThread(int gpu_id, std::thread &thread);

// Initializes a high priority CPU thread
void initHighPriorityThread(std::thread &thread);
void initHighPriorityThread(std::thread &thread, int num_cores);

// Initializes a low priority CPU thread
void initLowPriorityThread(std::thread &thread);

unsigned coreCount();
void setCore(unsigned core);
void setCores(std::vector<unsigned> cores, pthread_t thread);
void setAllCores();
void addCore(unsigned core);
void removeCore(unsigned core);
std::vector<unsigned> currentCores();
unsigned getCurrentCore();

std::vector<unsigned> getGPUCoreAffinity(unsigned gpu_id);

int minPriority(int scheduler);
int maxPriority(int scheduler);

void setDefaultPriority();
void setDefaultPriority(pthread_t thread);
void setMaxPriority();
void setPriority(int scheduler, int priority, pthread_t thread);

int currentScheduler();
int currentPriority();

}
}

#endif
