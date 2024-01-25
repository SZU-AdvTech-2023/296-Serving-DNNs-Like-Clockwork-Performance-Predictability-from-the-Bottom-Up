# Section 6.2 Can Clockwork Serve Thousands?

This experiment reproduces **Figure 6** from Section 6.2.

This experiment uses 3 machines in total (1 with GPUs) and requires approximately 1.5 hours.

For this experiment, set `CLOCKWORK_DISABLE_INPUTS=0`. This will cause inputs to be generated at the clients.

## Overview

<!--This experiment uses 3 machines in total (1 with GPUs). The workload runs for approximately 1.5 hours.  Afterwards, logs are collected from the machines and processed to produce figures presented in the paper.
-->

In this experiment, `N + 1` independent copies (i.e., separate model instances) of ResNet50 are deployed on a single worker using a single GPU, and set a SLO of 100ms while configuring the controller.

For 1 model instances, a steady request rate of 200r/s is maintained throughout the experiment (we denote this workload as the *minor* workload).

For the remaining `N` model instances, we vary the number of instances that are active at any point in time, and evenly distribute a workload of 1000r/s across all active models (we denote this workload as the *major* workload).

In the first 15 minutes, only the minor workload is active. Thereafter, an additional model instance belonging to the major workload is activated every 1 second (the total major workload request rate remains at 1000r/s).

The experiment tests Clockwork when workload demand changes.

To execute this experiment, the Clockwork client must be invoked using the `bursty_experiment` workload. Here is the relevant workload description obtained by running `./client -h`:

```
Usage: client [address] [workload] [workload parameters (if required)]
Available workloads with parameters:
	...
	bursty_experiment
			 num_models: (int, default 3600) number of 'major' workload models
	...
```

The default value of `N` is 3600, which works well for workers provisioned on the MPI cluster machines. For workers with lower RAM, `N` can be reduced by half.

The Clockwork controller must be configured to use 1 GPU and a 100ms default SLO. For reference, here are the relevant controller options for the default INFER5 scheduler, obtained by running `./controller -h`:

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

To manually run this experiment on machines `cluster01` (client), `cluster02` (controller), `cluster03` (worker), run the following commands in order. Suppose that Clockwork binaries are located at `${CLOCKWORK_BUILD}` on each machine. 

1. On `cluster03`, run `${CLOCKWORK_BUILD}/worker`
2. On `cluster02`, run `${CLOCKWORK_BUILD}/controller INFER5 cluster03:12345 0 1 100000000`
3. On `cluster01`, run `${CLOCKWORK_BUILD}/client cluster02:12346 bursty_experiment`

After 75 minutes, terminate all processes.

The telemetry files `clockwork_request_log.tsv` and `clockwork_action_log.tsv` are located on `cluster02`, either at `${CLOCKWORK_LOG_DIR}` (if defined and already created) or at `/local/`.

For reproducing Figure 6 graphs, run the `process_request_log.py` and `process_action_log.py` scripts on the respective telemetry files. For details, see [Processing Telemetry Data](#processing-telemetry-data) below.

We also provide scripts to automate the aforementioned workflow.

## Running the Experiment Using Scripts

<!--* This experiment requires 1 worker with GPUs, a non-GPU machine for the controller, and a non-GPU machine for the client
* ==TODO:== This experiment requires 10Gbit networking between controller and worker machines
* ==TODO==: **Suitability for Workers with 16GB GPU Memory**
* ==TODO==: **Suitability for Workers with low RAM**-->

This experiment takes approximately 1.5 hours to run.

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
* Currently, `logdir` is set to `/local/clockwork/bursty_experiment/log/[TIMESTAMP]`, based on the timestamp at which the experiment starts
* Modify `logdir`, especially if the default path is not writable


### Running the scripts

1. Ensure you have followed the basic Clockwork build, setup, and test steps described in the main Clockwork repository
1. Configure the experiment as described above
2. From this directory `sh ./run_in_background.sh` will execute `run.sh` in the background
3. On any machines, you can `tail` the respective logfiles `{logdir}/*.log`
4. On the master node form which the experiment was initiated, you can check experiment progress by tailing `{logdir}/run.log`
5. The experiment is over once `run.log` has output `Exiting`

The experiment duration is roughly 1.5 hours.

Upon completion, all necessary logs from remote machines will be copied back to the master node that initiated the experiment.

In particular, the telemetry files from the controller machine are copied back to the master node at `{logdir}/file=controller_request.tsv` and `{logdir}/file=controller_action.tsv`.

## Processing Telemetry Data

To generate graphs, either after running the experiment scripts or after running the experiments manually, run the following two commands:

```
python3 process_request_log.py -i {logdir}/file=controller_request.tsv -o {logdir}
python3 process_action_log.py -i {logdir}/file=controller_action.tsv  -o {logdir}
```

This will output all graphs to `{logdir}`.  A total of 5 graphs are produced corresponding to Figure 6:

 * `6a-goodput.pdf`
 * `6b-latency.pdf`
 * `6c-coldstarts.pdf`
 * `6d-pci.pdf`
 * `6e-gpu.pdf`

Plotting takes approximately 10 minutes.

## Customizing Your Environment

The default workload configuration provisions 3601 model instances (since `N = 3600`, by default).

For workers with lower RAM, a smaller value of `N` can be used when invoking the Clockwork client, e.g., using `${CLOCKWORK_BUILD}/client cluster02:12346 bursty_experiment 1800`.

A GPU device memory of 32 GB (like in MPI machines) is saturated once 201 model instances have been activated. For other configurations, such as for GPUs with only 16 GB memory, the saturation point arrives slightly earlier.

For details, refer to the [Customizing Your Environemnt](https://gitlab.mpi-sws.org/cld/ml/clockwork/-/blob/master/docs/customizing.md) page in the Clockwork source repository.
