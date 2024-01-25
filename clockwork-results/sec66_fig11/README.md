# Section 6.6 Can Clockwork Scale?

This experiment reproduces **Figure 11** from Section 6.6.

This experiment uses 3 machines in total and requires approximately 2 hours.

For this experiment, we set `CLOCKWORK_DISABLE_INPUTS=1`. This will cause inputs to be generated at the workers.

## Overview

In this experiment, we evaluate the scalability of Clockwork's controller using the Microsoft Azure Functions (MAF) workload trace, and using [Clockwork's emulated workers](https://gitlab.mpi-sws.org/cld/ml/clockwork/-/blob/master/docs/withoutgpus.md) (hence, this experiment does not require machines with GPUs).

In particular, we run 15 experiments. In each experiment, we evaluate the peak goodput for `N` emulated workers. Across the 15 expeirments, `N` is increased from 10 to 150 in steps of 10, i.e., 10, 20, 30, and so on.

Given `N`, the peak goodput is evaluated using the `azure_scalability_exp` workload generator. Here is the relevant workload description obtained by running `./client -h`:

```
Usage: client [address] [workload] [workload parameters (if required)]
Available workloads with parameters:
	...
	azure_scalability_exp
		 Description: Same as the azure workload above, but with an added mechanism to periodically increase the load factor. Only the necessary workload parameters have been retained.
		 Workload parameters:
			 num_workers: (int, default 1) the number of workers you're using
			 load_factor_min: (float, default 0.1) the minimum load factor
			 load_factor_max: (float, default 2.0) the maximum load factor
			 load_factor_inc: (float, default 2.0) the factor by which load factor is incremented
			 load_factor_period: (float, default 1.0) the period in seconds after which the load factor is incremented
			 memory_load_factor: (1, 2, 3, or 4; default 4):
				 1: loads approx. 200 models
				 2: loads approx. 800 models
				 3: loads approx. 1800 models
				 4: loads approx. 4000 models
	...
```

The workload generator increases the load factor periodically every `load_factor_period` seconds. Starting with `load_factor_min`, the load factor increased by a factor of `load_factor_inc` in every step, until it exceeds `load_factor_max`. Thereafter, we measure the maximum goodput achieved for this configuration from the telemetry files. We repeat the experiment for each `N`, three times, and report the median.

The Clockwork controller for each experiment is configured to use `N` GPUs (respectively) and a 100ms default SLO. For reference, here are the relevant controller options for the default INFER5 scheduler, obtained by running `./controller -h`:

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

### Running the Experiment with 10 Emulated Workers

To manually run the experiment on machines `cluster01` (client), `cluster02` (controller), and `cluster03` (worker) with 10 emulated workers, run the following commands in order. Suppose that Clockwork binaries are located at `${CLOCKWORK_BUILD}` on each machine. 

First, on `cluster03`, run

```
${CLOCKWORK_BUILD}/worker_dummy -n 10
```

Second, on `cluster02`, run

```
${CLOCKWORK_BUILD}/controller INFER5 cluster03:12345 0 10 5000000 100000000
```

Finally, on `cluster01`, run

```
${CLOCKWORK_BUILD}/client cluster02:12346 azure_scalability_exp 0.1 4 1.5 60
```

The client will terminate automatically after around 10 minutes.

The telemetry files `clockwork_request_log.tsv` and `clockwork_action_log.tsv` are located on `cluster02`, either at `${CLOCKWORK_LOG_DIR}` (if defined and already created) or at `/local/`.
Copy them to a designated folder or rename them, e.g., to `clockwork_request_log_N=10_trial=1.tsv` and `clockwork_action_log_N=10_trial=1.tsv`, respectively; otherwise, they will be overwritten when you run the second trial of the experiment or run the experiment with a different configuration.

### Running the Experiment for Other Configurations

Repeat the above experiment for `N=10`, `N=20`, `N=30`, up to `N=150`. For each `N`, repeat the experiment three times.

The graph in Figure 11 (left) in the paper reports aggregated measurements across three trials for `N=40`, and the graph in Figure 11 (right) in the paper reports aggregated measurement across all 45 experiments (i.e., 3 trials x 15 different values of `N`). For reproducing these graphs, see [Processing Telemetry Data](#processing-telemetry-data) below.

<!--For reproducing Figure 11 graphs, run the `process_request_log.py` and `process_action_log.py` scripts on the respective telemetry files. For details, see [Processing Telemetry Data](#processing-telemetry-data) below.
-->

We also provide scripts to automate the aforementioned workflow.

## Running the Experiment Using Scripts

<!--* This experiment requires 1 worker with GPUs, a non-GPU machine for the controller, and a non-GPU machine for the client
* ==TODO:== This experiment requires 10Gbit networking between controller and worker machines
* ==TODO==: **Suitability for Workers with 16GB GPU Memory**
* ==TODO==: **Suitability for Workers with low RAM**-->

This experiment takes approximately 2 hours to run.

The experiment will be initiated remotely over SSH by the master node, assuming that a password-free SSH connection can be used to execute commands remotely.

### Configuring the scripts

All machines should have Clockwork checked out under the same path.

On the master node, set `CLOCKWORK_BUILD` to the path to Clockwork's build directory (e.g. such that `${CLOCKWORK_BUILD}/worker` can be invoked).

Modify the following variables in `run.sh`:

* `client` hostname of machine that will run the client
* `controller` hostname of machine that will run the controller
* `worker` hostname of machine that will run the workers
* `username` username to use when SSH'ing

In addition, `run.sh` uses `logdir` as the path to send process outputs and logs.

* This directory will be created on all machines
* At the end of the experiment, outputs will be copied back from machines to the master node from which the experiment was initiated
* Currently, `logdir` is set to `/local/clockwork/azure_scalability_exp_b/log/[TIMESTAMP]`, based on the timestamp at which the experiment starts
* Modify `logdir`, especially if the default path is not writable


### Running the scripts

1. Ensure you have followed the basic Clockwork build, setup, and test steps described in the main Clockwork repository
1. Configure the experiment as described above
2. From this directory `sh ./run_in_background.sh` will execute `run.sh` in the background
3. On any machines, you can `tail` the respective logfiles `{logdir}/*.log`
4. On the master node form which the experiment was initiated, you can check experiment progress by tailing `{logdir}/run.log`
5. The experiment is over once `run.log` has output `Exiting`

The experiment duration is roughly 2 hours.

Upon completion, all necessary logs from remote machines will be copied back to the master node that initiated the experiment.

In particular, the telemetry files from the controller machine are copied back to the master node at `{logdir}/file=controller_i_request.tsv` and `{logdir}/file=controller_i_action.tsv`, where `i` denotes the config

### Repeating the epxeriment

Rerun the above experiment scripts thrice, so that we can report the median across three trials.

## Processing Telemetry Data

To process the telemetry data and to generate the graphs as provided in Figure 11 in the paper, we use the `process.py` and `data/plotter.py` scripts.

### Step 1

In the first step, we process the telemetry files using the `process.py` script. The processing script generates different types of timeseries data from the telemetry output.

If running the experiment using scripts, assuming that the output of the three trials is stored at `{logdir-trial-1}`, `{logdir-trial-2}`, and `{logdir-trial-3}`, we run the following:

```
python3 process.py -i {logdir-trial-1} -o ./data/processed-data-trial-1
python3 process.py -i {logdir-trial-2} -o ./data/processed-data-trial-2
python3 process.py -i {logdir-trial-3} -o ./data/processed-data-trial-3
```

The `process.py` script assumes that the telemetry files are named as `file=controller_i_request.tsv` and `file=controller_i_action.tsv`, where `i` denotes the config number. In this case, config number 1 corresponds to `N=10`, 2 corresponds to `N=20`, and so on. Thus, if running the experiment manually, make sure the telemetry files for each trials are copied in one fold with appropriate filenames.

### Step 2

Once the telemetry data has been processed, simply run the plotting script as follows.

```
cd data
python3 plotter.py -i . -o .
```

The script will output the following graphs.

* `11a_goodput_median_40_workers.pdf` (Figure 11 left)
* `11b_goodput_median_aggregate.pdf` (Figure 11 right)