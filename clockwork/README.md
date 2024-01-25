# Clockwork

A multi-tenant managed inference server, backed by a modified version of TVM.  Read about Clockwork in our [OSDI 2020 Preprint](https://arxiv.org/pdf/2006.02464.pdf)

This README file describes the pre-requisites and steps required to build and run Clockwork.  If you follow these steps but encounter errors, please e-mail the mailing list.

Clockwork is not feature complete, but we welcome contributions from others!

Mailing List: clockwork-users@googlegroups.com

# Resources

### Other Repositories

The following other repositories are relevant and will be referenced here and there.

* [`clockwork-results`](https://gitlab.mpi-sws.org/cld/ml/clockwork-results) contains experiment scripts and documentation for reproducing results from the OSDI 2020 Clockwork paper.
* [`clockwork-modelzoo-volta`](https://gitlab.mpi-sws.org/cld/ml/clockwork-modelzoo-volta) contains pre-compiled models that can be used for experimentation
* [`azure-functions`](https://gitlab.mpi-sws.org/cld/trace-datasets/azure-functions) contains workload traces from Microsoft Azure that can be used for experimentation
* [`azure-functions` (deprecated)](https://gitlab.mpi-sws.org/cld-private/datasets/azure-functions) contains the "preview" traces from Microsoft Azure.  This repository is only available internally.  Credentials will be provided to OSDI 2020 evaluators.

### Getting Started

The following pages step through the things required to build and run Clockwork

* [Installation Pre-Requisites](docs/prerequisites.md)
* [Building Clockwork](docs/building.md)
* [Environment Setup](docs/environment.md)
* [Clockwork Configuration](docs/configuration.md)
* [Running Clockwork for the first time](docs/firstrun.md)

### Next Steps
* [Clockwork Workflow](docs/workflow.md) An overview of Clockwork's current workflow
* [Customizing Your Environment](docs/customizing.md) Tweaks needed if you have different machines and GPUs
* [Running Without GPUs](docs/withoutgpus.md) Instructions for running without GPUs

### Additional Information
* [Telemetry](docs/telemetry.md) Description of telemetry logged by Clockwork
* [Troubleshooting Guide](docs/troubleshooting.md) Common error messages
* Experiment documentation in the [`clockwork-results`](https://gitlab.mpi-sws.org/cld/ml/clockwork-results) repository.
* [Workloads](docs/workloads.md) Available client workloads
* [Controller](docs/controller.md) Controller options

# Contacts

#### Mailing List

clockwork-users@googlegroups.com

#### People
Arpan Gujarati, Reza Karimi, Safya Alzayat, Wei Hao, Antoine Kaufmann, Ymir Vigfusson, Jonathan Mace

#### Organizations

Max Planck Institute for Software Systems

Emory University
