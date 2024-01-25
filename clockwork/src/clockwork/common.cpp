#include "clockwork/common.h"

namespace clockwork {

std::array<TaskType, 5> TaskTypes = {
	CPU, PCIe_H2D_Weights, PCIe_H2D_Inputs, GPU, PCIe_D2H_Output
};

std::string TaskTypeName(TaskType type) {
	switch(type) {
		case CPU: return "CPU";
		case PCIe_H2D_Weights: return "PCIe_H2D_Weights";
		case PCIe_H2D_Inputs: return "PCIe_H2D_Inputs";
		case GPU: return "GPU";
		case PCIe_D2H_Output: return "PCIe_D2H_Output";
	};
}

// std::array<TaskType, 8> TaskTypes = {
// 	Disk, CPU, PCIe_H2D_Weights, PCIe_H2D_Inputs, GPU, PCIe_D2H_Output, Sync, ModuleLoad
// };

// std::string TaskTypeName(TaskType type) {
// 	switch(type) {
// 		case Disk: return "Disk";
// 		case CPU: return "CPU";
// 		case PCIe_H2D_Weights: return "PCIe_H2D_Weights";
// 		case PCIe_H2D_Inputs: return "PCIe_H2D_Inputs";
// 		case GPU: return "GPU";
// 		case PCIe_D2H_Output: return "PCIe_D2H_Output";
// 		case Sync: return "Sync";
// 		case ModuleLoad: return "cuModuleLoad";
// 	};
// }

}
