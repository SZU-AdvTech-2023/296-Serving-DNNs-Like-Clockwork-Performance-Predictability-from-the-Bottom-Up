# Section 6.3 How Low Can Clockwork Go?

This experiment reproduces **Figure 7** from Section 6.3.

This experiment uses 8 machines in total (6 with GPUs), and requires approximately 60 minutes.

For this INFER5experiment, set `CLOCKWORK_DISABLE_INPUTS=0`. This will cause inputs to be generated at the clients.

## Overview

In this experiment, `N` independent copies (i.e., separate model instances) of ResNet50 are deployed on 6 workers, with 1 GPU being on each worker.

Inference requests for the `N` model instances are sent at a cumulative request rate of `R` requets per second.

The experiment goal is to evaluate how Clockwork reacts to low-latency SLOs.

For the first 30 seconds, requests are sent to the controller with an SLO target of 2.9 ms (which is 1x times a single inference latency on the GPU, without batching). Thereafter, every 30 seconds, the SLO target is increased by 50% (i.e., increased by 1.5x), until the SLO target exceeds 100x times the single inference latency.

To execute this experiment for `N = 12` and `R = 600`, the Clockwork client can be invoked using the `slo-exp-1` workload and appropriate arguments, i.e., using `./client [address] slo-exp-1 resnet50_v2  12 poisson 600 1 100 1.5 mul 30`.

For reference, here is the relevant workload description obtained by running `./client -h`:

```
Usage: client [address] [workload] [workload parameters (if required)]
Available workloads with parameters:
	...
	slo-exp-1 model copies dist rate slo-start slo-end slo-factor slo-op period
		 Workload parameters:
			 model: model name (e.g., "resnet50_v2")
			 copies: number of model instances
			 dist: arrival distribution ("poisson"/"fixed-rate")
			 rate: arrival rate (in requests/second)
			 slo-start: starting slo multiplier
			 slo-end: ending slo multiplier
			 slo-factor: factor by which the slo multiplier should change
			 slo-op: operator ("add"/"mul") for incrementing slo
			 period: number of seconds before changing slo
		 Examples:
			 client volta04:12346 slo-exp-1 resnet50_v2 4 poisson 100 2 32 2 mul 7
				 (increases slo every 7s as follows: 2 4 8 16 32)
			 client volta04:12346 slo-exp-1 resnet50_v2 4 poisson 100 10 100 10 add 3
				 (increases slo every 3s as follows: 10 20 30 ... 100)
		 In each case, an open loop client is used
	...
```

<!--In order to generate Figure 7, we run six different experiments where `N = 12` or `N = 48`, and where `R = 600`, `R = 1200`, or `R = 2400`.-->

Since the `slo-exp-1` workload assigns low-latency SLOs to each request on the client-side itself, the controller default SLO value is not relevant. Thus, when invoking the controller, we only configure its `generate_inputs`, `max_gpus`, and `schedule_ahead` options. 

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
${CLOCKWORK_BUILD}/controller INFER5 cluster03:12345,cluster04:12345,cluster05:12345,cluster06:12345,cluster07:12345,cluster08:12345 0 6 5000000
```

Third, on `cluster01`, run

```
${CLOCKWORK_BUILD}/client cluster02:12346 slo-exp-1 resnet50_v2  12 poisson 600 1 100 1.5 mul 30
```

The client will terminate automatically after around 10 minutes. Terminate other processes afterwards.

The telemetry files `clockwork_request_log.tsv` and `clockwork_action_log.tsv` are located on `cluster02`, either at `${CLOCKWORK_LOG_DIR}` (if defined and already created) or at `/local/`.

For reproducing Figure 7, run the experiment with six different configurations, where `N = 12` or `N = 48`, and where `R = 600`, `R = 1200`, or `R = 2400`.

Make sure the telemetry files for each experiment are either stored in separate directories, or renamed after each experiment.

In the end, copy all telemetry files to the master node, and run the `process.py` script to process them and produce graphs corresponding to Figure 7. For details, see [Processing Telemetry Data](#processing-telemetry-data) below.

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
* Currently, `logdir` is set to `/local/clockwork/slo-exp-1/log/[TIMESTAMP]`, based on the timestamp at which the experiment starts
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

To generate the graph corresponding to **Figure 7**, run `python3 plotter.py -l {logdir}`. If you are running the experiment manually, ensure that all telemetry files are copied to `{logdir}` on the master node.

The graph is output to `./graphs/fig7_dist=poisson.pdf`.

Plotting takes approximately 10 minutes. 

## Customizing Your Environment

This experiment does not fully utilize GPU memory, so it is possible to use GPUs with less memory and reproduce the same results.  To do this, Clockwork must be configured to expect less GPU memory.

On all worker machines, edit `config/default.cfg` and change `weights_cache_size` to `10737418240L`.

For details, refer to the [Customizing Your Environemnt](https://gitlab.mpi-sws.org/cld/ml/clockwork/-/blob/master/docs/customizing.md) page in the Clockwork source repository.