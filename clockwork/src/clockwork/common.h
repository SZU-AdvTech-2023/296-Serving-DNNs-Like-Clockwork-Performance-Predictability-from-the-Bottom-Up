#ifndef _CLOCKWORK_COMMON_H_
#define _CLOCKWORK_COMMON_H_

#include <functional>
#include <array>
#include "clockwork/telemetry.h"
#include <cuda_runtime.h>

namespace clockwork {
	
enum TaskType {
	CPU, PCIe_H2D_Weights, PCIe_H2D_Inputs, GPU, PCIe_D2H_Output
};
extern std::array<TaskType, 5> TaskTypes;

// enum TaskType {
// 	Disk, CPU, PCIe_H2D_Weights, PCIe_H2D_Inputs, GPU, PCIe_D2H_Output, Sync, ModuleLoad
// };
// extern std::array<TaskType, 8> TaskTypes;

std::string TaskTypeName(TaskType type);




}

#endif
