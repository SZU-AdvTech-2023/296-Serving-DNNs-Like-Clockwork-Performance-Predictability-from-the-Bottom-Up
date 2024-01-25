# Section 6.1 How Does Clockwork Compare?

This experiment reproduces the throughput and latency data for Clockwork, which is used in **Figure 5** from Section 6.1.

The experiment requires 1 worker with GPUs, a non-GPU machine for the controller, and a non-GPU machine for the client.

<!--* ==TODO:== This experiment requires 10Gbit networking between controller and worker machines
* ==TODO==: **Suitability for Workers with 16GB GPU Memory**
* ==TODO==: **Suitability for Workers with low RAM**-->

For this experiment, set `CLOCKWORK_DISABLE_INPUTS=0`. This will cause inputs to be generated at the clients. <!--In this case, the `generate_inputs_opt` variable in `run.sh` does not matter.-->


## Overview

In this experiment, 15 independent copies of ResNet50 (i.e., 15 separate model instances) are provisioned on a single worker and one GPU.

For each model instance, 16 closed-loop clients are run, which send inference requests to the controller in a closed-loop fashion. That is, a second request is sent immediately and only after a response to the first request is returned.

The Clockwork client can be invoked using the `comparison_experiment2` workload and default arguments to execute this experiment. Here is the relevant workload description obtained by running `./client -h`:
	
```
Usage: client [address] [workload] [workload parameters (if required)]
Available workloads with parameters:
	...
	comparison_experiment2
                 Description: closed-loop version of comparison experiment
                 Workload parameters:
                         num_models: (int, default 15) the number of models you're using
                         concurrency: (int, default 16) closed loop workload concurrency
	...
```

In order to generate statistics for Figure 5, we run multiple experiments, varying the default controller SLO from 10 ms to 500 ms.

Since the `comparison_experiment2` workload generator does not assign individual SLOs to each request, the controller assigns a default SLO to each incoming request.
The default controller SLO is configured by passing it the appropriate SLO value (in nanoseconds) during invocation.

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

To manually run this experiment on machines `cluster01` (client), `cluster02` (controller), `cluster03` (worker) for, say, a target SLO of 250ms, run the following commands in order. Suppose that Clockwork binaries are located at `${CLOCKWORK_BUILD}` on each machine. 

1. On `cluster03`, run `${CLOCKWORK_BUILD}/worker`
2. On `cluster02`, run `${CLOCKWORK_BUILD}/controller INFER5 cluster03:12345 0 1 250000000`
3. On `cluster01`, run `${CLOCKWORK_BUILD}/client cluster02:12346 comparison_experiment2`

After 30 minutes, terminate all processes.

The telemetry files `clockwork_request_log.tsv` and `clockwork_action_log.tsv` are located on `cluster02`, either at `${CLOCKWORK_LOG_DIR}` (if defined and already created) or at `/local/`.

For reproducing Figure 5, run the experiment with SLO values of 10ms, 25ms, 50ms, 100ms, 250ms, 500ms (3 hours of experiment time). Make sure the telemetry files for each experiment are either stored in separate directories, or renamed after each experiment run.

In the end, run the `process.py` script to process the telemtry files and produce graphs corresponding to Figure 5. For details, see [Processing Telemetry Data](#processing-telemetry-data) below.

We also provide scripts to automate the aforementioned workflow. 

<!--This experiment uses 3 machines in total (1 with GPUs). The workload runs for approximately 3 hours.  Afterwards, logs are collected from the machines and processed to produce figures presented in the paper.
-->

## Running the Experiment Using Scripts

<!--* This experiment requires 1 worker with GPUs, a non-GPU machine for the controller, and a non-GPU machine for the client
* ==TODO:== This experiment requires 10Gbit networking between controller and worker machines
* ==TODO==: **Suitability for Workers with 16GB GPU Memory**
* ==TODO==: **Suitability for Workers with low RAM**-->

The experiment takes approximately 3 hours to run.

### Requirements

The experiment is executed from a *master* node. This may or may not be one of the machines on which the Clockwork processes run.

The experiment will be initiated remotely over SSH by the master node, assuming that a password-free SSH connection can be used to execute commands remotely.

### Configuring the scripts

All machines should have Clockwork checked out under the same path.

On the master node, set `CLOCKWORK_BUILD` to the path to Clockwork's build directory (e.g. such that `${CLOCKWORK_BUILD}/worker` can be invoked).

Modify the following variables in `run.sh`:

* `client` hostname of machine that will run the client
* `controller` hostname of machine that will run the controller
* `worker` hostname of machine that will run the worker
* `username` username to use when SSH'ing

In addition, `run.sh` uses `logdir` as the path to send process outputs and logs.

* This directory will be created on all machines
* At the end of the experiment, outputs will be copied back from machines to the master node from which the experiment was initiated
* Currently, `logdir` is set to `/local/clockwork/comparison_experiment2/log/[TIMESTAMP]`, based on the timestamp at which the experiment starts
* Modify `logdir`, especially if the default path is not writable

<!--* `logdir` is a path to send process outputs and logs
* This directory will be created on all machines
* At the end of the experiment, outputs will be copied back from machines to the master node from which the experiment was initiated
* Currently, `logdir` is set to `/local/clockwork/comparison_experiment2/log/[TIMESTAMP]`, based on the timestamp at which the experiment starts
-->

<!--Set `CLOCKWORK_BUILD` to Clockwork's build directory.  All nodes must check out Clockwork at the same location.-->

<!--Set `CLOCKWORK_DISABLE_INPUTS=0`. This will cause inputs to be generated at the clients. In this case, the `generate_inputs_opt` variable in `run.sh` does not matter.-->

<!--## Experiment Workflow-->

### Running the scripts

1. Ensure you have followed the basic Clockwork build, setup, and test steps described in the main Clockwork repository
1. Configure the experiment as described above
2. From this directory `sh ./run_in_background.sh` will execute `run.sh` in the background
3. On any machines, you can `tail` the respective logfiles `{logdir}/*.log`
4. On the master node form which the experiment was initiated, you can check experiment progress by tailing `{logdir}/run.log`
5. The experiment is over once `run.log` has output `Exiting`

The experiment duration is roughly 3 hours.

Upon completion, all necessary logs from remote machines will be copied back to the master node that initiated the experiment.

For exmaple, for the experiment run with 25ms SLO, the telemetry files from the controller machine are copied back to the master node at `{logdir}/file=controller_request_25.tsv` and `{logdir}/file=controller_action_25.tsv`.

## Processing Telemetry Data

To reproduce Figure 5, only request telemetry files are needed. The plotting script is `process.py`. Run `python3 process.py -h` to see its usage.

```
usage: process.py [-h] [-o OUTPUTDIR] SLO:FILENAME [SLO:FILENAME ...]

Process multiple request log files for different SLOs

positional arguments:
  SLO:FILENAME          An SLO and corresponding filename, separated with a colon, e.g. 25:/local/clockwork_request_log.tsv

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUTDIR, --outputdir OUTPUTDIR
                        Directory to put the processed output. Directory will be created if it does not exist.
```

To generate graphs, either after running the experiment scripts or after running the experiments manually, run

```
python3 process.py 10:{request_telemetry_for_10ms} 25:{request_telemetry_for_25ms} 50:{request_telemetry_for_50ms} 100:{request_telemetry_for_100ms} 250:{request_telemetry_for_250ms} 500:{request_telemetry_for_500ms} -o {logdir}
```

This will output all graphs to `{logdir}`. A total of 7 graphs are produced corresponding to Figure 5:

 * `fig5_goodput.pdf`
 * `fig5_10ms_SLO.pdf`
 * `fig5_25ms_SLO.pdf`
 * `fig5_50ms_SLO.pdf`
 * `fig5_250ms_SLO.pdf`
 * `fig5_100ms_SLO.pdf`
 * `fig5_500ms_SLO.pdf`

Plotting takes approximately 10 minutes.

## Customizing Your Environment

Since this experiment deals with only 16 ResNet50 model instances, it is suitable for workers with 16GB GPU Memory as well as workers with low RAM.

For details, refer to the [Customizing Your Environemnt](https://gitlab.mpi-sws.org/cld/ml/clockwork/-/blob/master/docs/customizing.md) page in the Clockwork source repository.


<!--On all worker machines, edit config/default.cfg and change weights_cache_size to 10737418240L.-->

<!--* ==TODO==: **Suitability for Workers with 16GB GPU Memory**
* ==TODO==: **Suitability for Workers with low RAM**-->

