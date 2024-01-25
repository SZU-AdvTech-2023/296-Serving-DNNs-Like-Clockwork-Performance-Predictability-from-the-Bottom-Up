# Section 6.4 Can Clockwork Isolate Performance?

This experiment reproduces **Figure 8** from Section 6.4.
INFER5
The experiment has a similar setup to [sec63_fig7](../sec63_fig7). It uses 8 machines in total (6 with GPUs), and requires approximately 60 minutes.

For this experiment, we set `CLOCKWORK_DISABLE_INPUTS=0`. This will cause inputs to be generated at the clients.

## Overview

In this experiment, `N + M` independent copies (i.e., separate model instances) of ResNet50 are deployed on 6 workers, with 1 GPU being on each worker.

Inference requests for the `N` model instances are sent at a request rate of `R` requests per second per model instance. These requests have low-latency SLOs.

Inference requests for the `M` model instances are sent in a closed loop. At most `C` requests can be sent per model instance (denoting the concurrency factor). These requests have batch (relatively higher) SLOs.

The experiment goal is to evaluate if Clockwork can isolate the performance of low-latency and batch requests.

The low-latency SLOs are varied as follows. For the first 30 seconds, requests are sent to the controller with an SLO target of 2.9 ms (which is 1x times a single inference latency on the GPU, without batching). Thereafter, every 30 seconds, the SLO target is increased by 50% (i.e., increased by 1.5x), until the SLO target exceeds 100x times the single inference latency.

To execute this experiment for `N=6`, `R = 200`, `M = 12` and `C = 16`, the Clockwork client can be invoked using the `slo-exp-2` workload and appropriate arguments, i.e., using `./client [address] slo-exp-2 resnet50_v2 6 poisson 200 1 100 1.5 mul 30 12 16 0`.

For reference, here is the relevant workload description obtained by running `./client -h`:

```
Usage: client [address] [workload] [workload parameters (if required)]
Available workloads with parameters:
	...
	slo-exp-2 model copies-fg dist-fg rate-fg slo-start-fg slo-end-fg slo-factor-fg slo-op-fg period-fg copies-bg concurrency-bg slo-bg
		 Description: Running latency-sensitive (foreground or FG) and batch (background or BG) workloads simultaneously
		 Workload parameters:
			 model: model name (e.g., "resnet50_v2")
			 copies-fg: number of FG models
			 dist-fg: arrival distribution ("poisson"/"fixed-rate") for open loop clients for FG models
			 rate-fg: total arrival rate (in requests/second) for FG models
			 slo-start-fg: starting slo multiplier for FG models
			 slo-end-fg: ending slo multiplier for FG models
			 slo-factor-fg: factor by which the slo multiplier should change for FG models
			 slo-op-fg: operator ("add"/"mul") for applying param slo-factor-fg
			 period-fg: number of seconds before changing FG models' slo
			 copies-bg: number of BG models (for which requests arrive in closed loop)
			 concurrency-bg: number of concurrent requests for BG model' closed loop clients
			 slo-bg: slo multiplier for BG moels (ideally, should be a relaxed slo)
		 Examples:
			 client volta04:12346 slo-exp-2 resnet50_v2    2 poisson 200  2 32 2 mul 7    4 1 100
				 (2 FG models with PoissonOpenLoop clients sending requests at 200 rps)
				 (the SLO factor of each FG model is updated every 7 seconds as follows: 2 4 8 16 32)
				 (4 BG models with a relaxed SLO factor of 100 and respective ClosedLoop clients configured with a concurrency factor of 1)
	...
```

The `slo-exp-2` workload assigns low-latency SLOs to each request for the `N` model instances on the client-side itself.

However, the batch SLO on the client side is configured as 0. In this case, the default SLO on the controller side is used. Thus, when invoking the controller, we configure it with a very high default SLO that acts as the batch SLO,
along with configuring the `generate_inputs`, `max_gpus`, and `schedule_ahead` options.

For reference, here are the relevant controller options for the default INFER5 scheduler, obtained by running `./controller -h`:

```
USAGE:
  controller [TYPE] [WORKERS] [OPTIONS]
	...
	INFER5    The Clockwork Scheduler.  You should usually be using this.  Options:
       generate_inputs    (bool, default false)  Should inputs and outputs be generated if not present.  Set to true to test network capacity
       max_gpus           (int, default 100)  Set to a lower number to limit the number of GPUs.
       schedule_ahead     (int, default 10000000)  How far ahead, in nanoseconds, should the scheduler schedule.
       default_slo        (int, default 100000000)  The default SLO to use if client's don't specify slo_factor.  Default 100ms
       max_exec        (int, default 25000000)  Don't use batch sizes >1, whose exec time exceeds this number.  Default 25ms
       max_batch        (int, default 16)  Don't use batch sizes that exceed this number.  Default 16.
	...
```

## Running the Experiment Manually

To manually run this experiment on machines `cluster01` (client), `cluster02` (controller), `cluster03`-`cluster08` (workers), run the following commands in order. Suppose that Clockwork binaries are located at `${CLOCKWORK_BUILD}` on each machine.

First, on each `cluster03`-`cluster08`, run

```
${CLOCKWORK_BUILD}/worker
```

Second, on `cluster02`, run

```
${CLOCKWORK_BUILD}/controller INFER5 cluster03:12345,cluster04:12345,cluster05:12345,cluster06:12345,cluster07:12345,cluster08:12345 0 6 5000000 86400000000000
```

Third, on `cluster01`, run

```
${CLOCKWORK_BUILD}/client cluster02:12346 slo-exp-2 resnet50_v2 6 poisson 200 1 100 1.5 mul 30 12 16 0
```


The client will terminate automatically after around 10 minutes. Terminate other processes afterwards.

The telemetry files `clockwork_request_log.tsv` and `clockwork_action_log.tsv` are located on `cluster02`, either at `${CLOCKWORK_LOG_DIR}` (if defined and already created) or at `/local/`.

For reproducing Figure 8, run the experiment with three different configurations:

1. `M = 0` and `C = 0` (the value of `C` is irrelevant here)
2. `M = 12` and `C = 16`
3. `M = 48` and `C = 4`

Make sure the telemetry files for each experiment are either stored in separate directories, or renamed after each experiment.

In the end, copy all telemetry files to the master node, and run the `plotter.py` script to process them and produce graphs corresponding to Figure 8. For details, see [Processing Telemetry Data](#processing-telemetry-data) below.

We also provide scripts to automate the aforementioned workflow. 

## Running the Experiment Using Scripts

The automated experiment takes approximately 1 hour to run.

## Requirements

The experiment is executed from a *master* node. This may or may not be one of the machines on which the Clockwork processes run.

The experiment will be initiated remotely over SSH by the master node, assuming that a password-free SSH connection can be used to execute commands remotely.

### Configuring the scripts

All machines should have Clockwork checked out under the same path.

On the master node, set `CLOCKWORK_BUILD` to the path to Clockwork's build directory (e.g. such that `${CLOCKWORK_BUILD}/worker` can be invoked).

Modify the following variables in `run.sh`:

* `client` hostname of machine that will run the client
* `controller` hostname of machine that will run the controller
* `workers` hostnames of machines that will run the workers
* `username` username to use when SSH'ing

In addition, `run.sh` uses `logdir` as the path to send process outputs and logs.

* This directory will be created on all machines
* At the end of the experiment, outputs will be copied back from machines to the master node from which the experiment was initiated
* Currently, `logdir` is set to `/local/clockwork/slo-exp-2/log/[TIMESTAMP]`, based on the timestamp at which the experiment starts
* Modify `logdir`, especially if the default path is not writable

### Running the scripts

1. Ensure you have followed the basic Clockwork build, setup, and test steps described in the main Clockwork repository
1. Configure the experiment as described above
2. From this directory `sh ./run_in_background.sh` will execute `run.sh` in the background
3. On any machines, you can `tail` the respective logfiles `{logdir}/*.log`
4. On the master node form which the experiment was initiated, you can check experiment progress by tailing `{logdir}/run.log`
5. The experiment is over once `run.log` has output `Exiting`

The experiment duration is roughly 1 hour.

Upon completion, all necessary logs from remote machines will be copied back to the master node that initiated the experiment.

In particular, all telemetry files from the controller machine are copied back to the master node at `{logdir}/`.

## Processing Telemetry Data

The plotting script is `plotter.py`. Run `python3 plotter.py -h` to see its usage.

```
usage: plotter.py [-h] -l LOGDIR

optional arguments:
  -h, --help            show this help message and exit
  -l LOGDIR, --logdir LOGDIR
                        Path to the log files
```

To generate the graph corresponding to **Figure 8**, run `python3 plotter.py -l {logdir}`. If you are running the experiment manually, ensure that all telemetry files are copied to `{logdir}` on the master node.

The graph is output to `./graphs/fig8_dist=poisson.pdf`.

Plotting takes approximately 10 minutes. 

## Customizing Your Environment

This experiment does not fully utilize GPU memory, so it is possible to use GPUs with less memory and reproduce the same results.  To do this, Clockwork must be configured to expect less GPU memory.

On all worker machines, edit `config/default.cfg` and change `weights_cache_size` to `10737418240L`.

For details, refer to the [Customizing Your Environemnt](https://gitlab.mpi-sws.org/cld/ml/clockwork/-/blob/master/docs/customizing.md) page in the Clockwork source repository.

<!--## Experiment Customization

### Workers with less than 32GB GPU Memory

Clockwork's experiments used 32GB Tesla v100 GPUs.  Some cloud providers offer v100 GPUs with less memory (e.g. 16GB).

This experiment does not fully utilize GPU memory, so it is possible to use GPUs with less memory and reproduce the same results.  To do this, Clockwork must be configured to expect less GPU memory.

On all worker machines, edit `config/default.cfg` and change `weights_cache_size` to `10737418240L`.

Common error messages that can happen when using different GPU configurations with different memory limits include:

```
  what():  [20:58:09] /home/jcmace/clockwork/src/clockwork/cache.cpp:237: Check failed: e == cudaSuccess || e == cudaErrorCudartUnloading: CUDA: out of memory
```
```
  what():  [21:03:51] /home/jcmace/clockwork/src/clockwork/model/cuda.cpp:80: cuModuleLoadData Error: CUDA_ERROR_OUT_OF_MEMORY
```

If the above errors occur, then consult the main README for how to adjust your configuration.

-->