# Section 6.5 Is Clockwork Predictable?

This experiment reproduces **Figures 9 and 10** from Section 6.5.

It requires 6 worker machines with GPUs, a non-GPU machine for the controller, and a non-GPU machine for the client.

For this experiment, we set `CLOCKWORK_DISABLE_INPUTS=1`, since we want to generate inputs on the controller.

## Overview

In this experiment, we simply replay a workload trace of Microsoft Azure Functions (MAF) for 8 hours in real-time.

In particular, we use 61 different model varieties, duplicate each 66 times, resulting in a total of 4026 model instances (which saturates the main memory capacity of MPI worker machines). We then reply four or five function traces for each model instance.

To execute this experiment, the Clockwork client can be invoked using the `azure` workload. Although we use 6 workers, we configure a workload corresponding to `num_workers = 8`. All other configurations are left as default.

For reference, here is the relevant workload description obtained by running `./client -h`:

```
Usage: client [address] [workload] [workload parameters (if required)]
Available workloads with parameters:
	...
	azure
		 Description: replay an azure workload trace.  Can be run with no arguments, in which case default values are used.  The defaults will load 3100 models and replay a trace that will give approximately the total load the system can handle.
		 Workload parameters:
			 num_workers: (int, default 1) the number of workers you're using
			 use_all_models: (bool, default 1) load all models or just resnet50_v2
			 load_factor: (float, default 1.0) the workload will generate approximately this much load to the system.  e.g. 0.5 will load by approximately 1/2; 2.0 will overload by a factor of 2
			 memory_load_factor: (1, 2, 3, or 4; default 4):
				 1: loads approx. 200 models
				 2: loads approx. 800 models
				 3: loads approx. 1800 models
				 4: loads approx. 4000 models
			 interval: (int, default 60) interval duration in seconds
			 trace: (int, 1 to 13 inclusive, default 1) trace ID to replay
			 randomise: (bool, default false) randomize each client's starting point in the trace
	bursty_experiment
			 num_models: (int, default 3600) number of 'major' workload models
	...
```

We configure the Clockwork controller to generate inputs on the controller side, i.e., `generate_inputs = 1`, and to use one GPU per worker, i.e., `max_gpus = 6`. All other configurations are left as default.

For reference, here are the relevant controller options for the default INFER4 scheduler, obtained by running `./controller -h`:

```
USAGE:
  controller [TYPE] [WORKERS] [OPTIONS]
	...
	INFER4    The Clockwork Scheduler.  You should usually be using this.  Options:
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
${CLOCKWORK_BUILD}/controller INFER4 cluster03:12345,cluster04:12345,cluster05:12345,cluster06:12345,cluster07:12345,cluster08:12345 1 6
```

Third, on `cluster01`, run

```
${CLOCKWORK_BUILD}/client cluster02:12346 azure 8
```

Terminate all process after 8 hours. To simply observe the trends in Figures 9 and 10, 2 hours experiment time is sufficient.

The telemetry files `clockwork_request_log.tsv` and `clockwork_action_log.tsv` are located on `cluster02`, either at `${CLOCKWORK_LOG_DIR}` (if defined and already created) or at `/local/`.

In the end, copy the telemetry files to the master node, and run the `process.py` script to process them and produce graphs corresponding to Figures 9 and 10. For details, see [Processing Telemetry Data](#processing-telemetry-data) below.

We also provide scripts to automate the aforementioned workflow. 

## Running the Experiment Using Scripts

The automated experiment takes approximately 8 hours to run.

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
* Currently, `logdir` is set to `/local/clockwork/maf-exp-1-and-2/log/[TIMESTAMP]`, based on the timestamp at which the experiment starts
* Modify `logdir`, especially if the default path is not writable

Running the experiment for 2 hours is sufficient to infer the trends reported in the paper. In order to do so, update the `timeout_duration` variable in `run.sh` to slightly more than 2 hours, e.g., `145m`, so that the controller gets around 15 minutes to load the models in the beginning on each worker.

### Running the scripts

1. Ensure you have followed the basic Clockwork build, setup, and test steps described in the main Clockwork repository
1. Configure the experiment as described above
2. From this directory `sh ./run_in_background.sh` will execute `run.sh` in the background
3. On any machines, you can `tail` the respective logfiles `{logdir}/*.log`
4. On the master node form which the experiment was initiated, you can check experiment progress by tailing `{logdir}/run.log`
5. The experiment is over once `run.log` has output `Exiting`

Upon completion, all necessary logs from remote machines will be copied back to the master node that initiated the experiment.

In particular, all telemetry files from the controller machine are copied back to the master node at `{logdir}/`.

## Processing Telemetry Data

The plotting script is `process.py`. Run `python3 process.py -h` to see its usage.

```
usage: process.py [-h] [-i INPUTDIR] [-o OUTPUTDIR] [-l LEADIN] [-d DURATION] [-b BUCKETSIZE]

Process clockwork_request_log.tsv and clockwork_action_log.tsv

optional arguments:
  -h, --help            show this help message and exit
  -i INPUTDIR, --inputdir INPUTDIR
                        Path to a directory containing experiment output files. Files should be in format "file=controller_1_request.tsv" etc.
  -o OUTPUTDIR, --outputdir OUTPUTDIR
                        Directory to put the processed output. Directory will be created if it does not exist.
  -l LEADIN, --leadin LEADIN
                        Exclude the first LEADIN seconds of data. Default 600 (10 minutes).
  -d DURATION, --duration DURATION
                        Include the first DURATION seconds of data. Set to -1 to include all data. Default -1.
  -b BUCKETSIZE, --bucketsize BUCKETSIZE
                        Interval size for time series data. Default 60 (1 minute).
```

To generate the graph corresponding to Figures 9 and 10, run `python3 process.py --inputdir {logdir} --outputdir {logdir}`. If you are running the experiment manually, ensure that all telemetry files are copied to `{logdir}` on the master node.

A total of 9 graphs are produced corresponding to Figures 9 and 10:

 * `9a_throughput.pdf`
 * `9b_timeseries_latency.pdf`
 * `9c_timeseries_batchsize.pdf`
 * `9d_timeseries_coldmodels.pdf`
 * `9e_timeseries_coldstarts.pdf`
 * `10_bl_infer_completion_error.pdf`
 * `10_br_load_completion_error.pdf`
 * `10_tl_infer_prediction_error.pdf`
 * `10_tr_load_prediction_error.pdf`

 Plotting takes approximately 10 minutes. 

## Experiment Customization

It is possible to use machines with less RAM and reproduce the same results. To do this, the workload must be configured to provision less models.

In particular, the `azure` workload can be configured with a `memory_load_factor` of `1`, `2`, or `3`, instead of the default value `4`, since 

```
	1: loads approx. 200 models
	2: loads approx. 800 models
	3: loads approx. 1800 models
	4: loads approx. 4000 models
```

and 4000 models reach close to the main-memory capacity of our MPI worker machines.

For details, refer to the [Customizing Your Environemnt](https://gitlab.mpi-sws.org/cld/ml/clockwork/-/blob/master/docs/customizing.md) page in the Clockwork source repository.

