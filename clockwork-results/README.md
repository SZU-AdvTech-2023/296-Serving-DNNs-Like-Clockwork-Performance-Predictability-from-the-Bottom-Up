# Clockwork Experiments

This repository contains scripts and instructions for running the experiments from ["Serving DNNs like Clockwork: Performance Predictability from the Bottom Up"](https://arxiv.org/pdf/2006.02464.pdf)

Important other repositories:
* [`clockwork`](https://gitlab.mpi-sws.org/cld/ml/clockwork) The main Clockwork repository containing source code, build instructions, environment, troubleshooting, and more.
* [`clockwork-modelzoo-volta`](https://gitlab.mpi-sws.org/cld/ml/clockwork-modelzoo-volta) Pre-compiled models used for experiments
* [`azure-functions` (new)](https://gitlab.mpi-sws.org/cld/trace-datasets/azure-functions) contains workload traces from Microsoft Azure that can be used for experimentation
* [`azure-functions` (private,deprecated)](https://gitlab.mpi-sws.org/cld-private/datasets/azure-functions) contains the "preview" traces from Microsoft Azure.  This repository is only available with credentials that will be provided to OSDI 2020 evaluators.  These are the traces that should be used to reproduce experiments.

# Abstract

Clockwork is a distributed system for serving DNNs.  Clockwork is open-source and available at [`clockwork`](https://gitlab.mpi-sws.org/cld/ml/clockwork); our artifact evaluation plan will reference that repository numerous times.

This repository contains the scripts needed to run the experiments presented in the [paper](https://arxiv.org/pdf/2006.02464.pdf).  We give an overview of how to evaluate Clockwork, describe the workflow of Clockwork experiments, and provide descriptions of how to run each experiment.

The primary software artifact is the system itself, located at [`clockwork`](https://gitlab.mpi-sws.org/cld/ml/clockwork).

# Roadmap

## Machines

The biggest pain point for evaluation will be system setup for the worker machines.  To reproduce the experiment results exactly, you will require worker machines each with: 768GB RAM or higher; 16 CPU cores or more; at least 1 32GB Tesla v100 GPU; 10GBit network.  The "big" experiment is the Azure experiment from section 6.5, which requires 6 worker machines.  Several experiments require fewer worker machines.

All is not lost!  It is possible to approximate the experiment results with different worker configurations.  In particular, most cloud providers tend to offer GPUs with 16GB memory rather than 32GB.  With a couple of tweaks, it's possible to use these GPUs.  Likewise, with other tweaks, you can account for less RAM.  There are workarounds for slower networks, though 10GBit is strongly recommended.  16 CPU cores is a hard constraint.

Two additional non-GPU machines are required - one to run the controller, and the other to run the client.  The controller requires at least 18 CPU cores, but otherwise has no significant resource requirements.  The client has no significant resource requirements.  However, it is strongly recommended that all nodes have at least 10GBit network.

## Setup

You could set up Clockwork from scratch by following the instructions in the [`clockwork`](https://gitlab.mpi-sws.org/cld/ml/clockwork) repository.  We have verified these instructions on Google Cloud VMs (though, per the previous section, they do not have identical resource offerings)

We have also provided a Docker image and scripts which may shortcut some of the steps.

If access is possible to MPI machines, then we will pre-configure everything ahead of time on your behalf.

## Experiments

Experiments can be run using the scripts provided in this repository.  We have also provided descriptions of how to run the experiments manually.

To get started with Clockwork, we recommend getting the system running manually, in order to understand the pieces involved, and to ensure the system has been configured appropriately for your machines.  Afterwards, you might then choose to run the experiments using the provided scripts or manually.

Concretely, we suggest the following "warmup" steps before running a full experiment:

1. Follow the "Getting Started" instructions for Clockwork in the [`clockwork`](https://gitlab.mpi-sws.org/cld/ml/clockwork) repository.  All documentation in the main Clockwork repository is part of this artifact evaluation process.
1. Get the system building and running without GPUs, with 3 machines (client, controller, worker_dummy), using the `simple` workload.
1. Get the system building and running using 1 real GPU, with 3 machines (client, controller, worker).  You can test any workload with only 1 worker (if it's a heavy workload it'll just drop lots of requests).  At this step, you want to try to trigger any configuration errors that could happen (e.g. due to differences between our setup and yours).
1. You can reproduce larger-scale experiments using only 1 worker, by using the `worker_dummy`.  This is an emulated Clockwork worker that does not require a GPU; it can emulate a large number of GPUs from a single machine.

The repository is structured based on the Evaluation section in the paper. The following table summarizes different experiments in the paper and the directory containing the respective experiment scripts. The READMEs in the respective experiment directories explain the experiment in detail.

| Experiment Name / Section / Paragraph | Related Figures | Experiment Directory  |
| :------------- | :---------- | :----------- |
| 6.1. How Does Clockwork Compare? | Figure 5 | [sec61_fig5](sec61_fig5) |
| 6.2. Can Clockwork Serve Thousands? | Figure 6 | [sec62_fig6](sec62_fig6) |
| 6.3. How Low Can Clockwork Go? | Figure 7 | [sec63_fig7](sec63_fig7) |
| 6.4. Can Clockwork Isolate Performance? | Figure 8 | [sec64_fig8](sec64_fig8) |
| 6.5. Are Realistic Workloads Predictable? | Figures 9 and 10 | [sec65\_fig9\_fig10](sec65_fig9_fig10) |
| 6.6. Can Clockwork Scale? | Figure 11 | [sec66_fig11](sec66_fig11) |

<!--| 6.5. Is Clockwork Predictable? Paragraph *Tighter SLOs at large scale*. (deprecated) | Table | [sec65_table](sec65_table) |-->

# Additional Setup Information for Experiment Scripts

### Environment Variables

The experiment scripts assume the following environment variables:

* `CLOCKWORK_BUILD` points to the location of the Clockwork build directory
* `CLOCKWORK_MODEL_DIR` points to the location of the modelzoo (e.g. `clockwork-modelzoo-volta`)

### Connections

The scripts require that a password-free SSH connection can be set up to execute commands on each machine remotely. For example, if your username is `alice`, and `cluster01` is one of the worker machines, then `ssh -o StrictHostKeyChecking=no -l alice cluster01 "${CLOCKWORK_BUILD}/worker"` should seamlessly start the `worker` process on machine `cluster01`.
When you are working with a Docker process (see below), it will open an SSH connection on port 2200 where it will recognize the same authorized SSH hosts (it copies the .ssh/authorized_keys file in the home directory of your user). The username inside a Docker environment is `clockwork', and the `CLOCKWORK_BUILD' directory is `/clockwork/build'. 

### Script Modifications

All scripts require modification before running, in order to set variables such as `client`, `controller`, and `worker` hostnames/addresses.  These are described in more detail with each experiment page.

### Setting up a Clockwork Environment on a Machine via Docker

We have provided a Docker container with source code, environment and dependencies for each of the Clockwork main processes.

We assume that the image runs Ubuntu 20.04, and has at least 50GB of storage. 
Your instance should make TCP ports 2200, 12345 and 12346 accessible to other Clockwork nodes.

First, type: `git clone https://gitlab.mpi-sws.org/cld/ml/clockwork-results'.

Now, run `clockwork-results/setup/host_setup.sh` on the machine. 
This will download the latest kernel drivers (v450.51.06) for NVIDIA Tesla v100 in case the machine has such a GPU.
This script will also set up a variant of Docker with native GPU support, and configure GPU parameters.
It will also resize `/dev/shm' on the host to use 8GB of RAM, to be used internally. Some experiments may require more
shared memory through `mount -o remount,rw,exec,size=???G'.

Next, ensure that the SSH public keys you intend to use for experiments exist in the `.ssh/authorized_keys' file under your user's home directory. This file will be copied.

Finally, set up the Docker environment using the following script on your machine:
`clockwork-results/setup/host_docker.sh`
When first run, it will grab and run the Clockwork Docker image from Docker Hub, for which credentials have been provided.
The command will launch a Docker environment reachable via SSH on port 2200 on the machine with the same authorized ssh keys
as your user had on the host instance.
You can test this by typing `ssh -p 2200 clockwork@localhost' on the machine. The Clockwork binaries live under `/clockwork/build'.

_OPTIONAL:_

Many of the experiments rely on precompiled models and the Azure functions trace. 
You can download each of them using the following commands:
`clockwork-results/setup/host_get_models.sh`
and
`clockwork-results/setup/host_get_trace.sh`
Please note that the models are large, and thus make take a while to download.
The trace is private and so the repository is password-protected. You have been provided with credentials.
The generated directories will be visible within the Docker environment when you rerun the `host_docker.sh` script.

### Additional Software Dependencies

The data analysis and plotting scripts are written in Python. They require `Python 3.x` and depend on `numpy`, `pandas`, `matplotlib`, and `seaborn` libraries.
