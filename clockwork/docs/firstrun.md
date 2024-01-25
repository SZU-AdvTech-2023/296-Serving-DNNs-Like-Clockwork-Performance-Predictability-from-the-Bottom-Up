# Running Clockwork for the First Time

This page describes how to run Clockwork for the first time.  Make sure you have followed the previous build and configuration steps.

**It is possible to run Clockwork without any GPUs**.  This will be mentioned in-line in these instructions.

This tutorial will require three machines - 1 worker, 1 controller, and 1 client.

## Check the environment is OK

From Clockwork's `build` directory on each machine, run

```
./profile [check]
```

This will check your current environment settings.

## Test a model

On the worker machine, from Clockwork's `build` directory, run:

```
./check_model ${CLOCKWORK_MODEL_DIR}/resnet50_v2/model
```

This command will only work if the worker machine has a GPU.  You should see output similar to the following:

```
Checking /home/jcmace/clockwork-modelzoo-volta/resnet50_v2/model using random input
  input_size:  602112
  output_size: 4000
  workspace:   9834496
  weights size paged (non-paged) [num_pages]: 117440512 (102160300) [7]
  weights transfer latency: 8.31 ms
  execution latency:
    b1: 2.73 ms
    b2: 4.05 ms
    b4: 5.88 ms
    b8: 9.93 ms
    b16: 17.33 ms
```

You can replace `resnet50_v2` with other models in the above command.

## Start a Worker

### Option 1: Start a Worker with a GPU

On the worker machine, from Clockwork's `build` directory, run:
```
./worker
```

You should see output similar to the following:

```
Using CUDA Driver 10020 Runtime 9020
Starting Clockwork Worker
Loading Clockwork worker default config from /home/jcmace/clockwork/config/default.cfg
GPU0-GPU-0 started
GPU0-PCIe_H2D_Inputs-0 started
PCIe_H2D_Weights-0 started
GPU0-PCIe_D2H_Output-0 started
AsyncTaskChecker-0 started
IO service thread listening on 0.0.0.0:12345
```

Workers use all available GPUs.

By default the worker will listen for incoming controller connections on port 12345.  Run `./worker -h` for more options.


### Option 2: Start a Worker without a GPU

On the worker machine, from Clockwork's `build` directory, run:
```
./worker_dummy -n 1
```

*Note*: the `-n 1` specifies it should simulate 1 GPU.  You can simulate many GPUs by increasing this number.

You should see output similar to the following:

```
Starting Clockwork Worker
Loading Clockwork worker default config from /home/jcmace/clockwork/config/default.cfg
IO service thread listening on 0.0.0.0:12345
```

The simulated worker behaves and responds similar to the real worker.  However, instead of performing actual inferences, it simply waits for the profiled execution time before responding.  It is useful for development and environment preparation before running experiments.

By default the worker will listen for incoming controller connections on port 12345.  Run `./worker_dummy -h` for more options.

## Start the controller

The workers started in the previous step will wait for an incoming connection from Clockwork's controller.

On the controller machine, from Clockwork's `build` directory, run:
```
./controller INFER5 volta01:12345
```

Here, `volta01:12345` is the hostname:port of the worker started in the previous step.  `INFER5` is the name of the default Clockwork scheduler.

You should see output similar to the following:
```
Starting Clockwork Controller
Logging requests to /local/clockwork_request_log.tsv
Logging actions to /local/clockwork_action_log.tsv
ConcurrentInferAndLoadScheduler using:
         default_slo=100000000
         latest_delta=10000000
         schedule_ahead=10000000
         max_allowable_exec_time=250000000
         max_batch_size=8
         generate_inputs=0
         max_gpus=100
Connecting to worker volta01:12345
IO service thread listening for clients on 0.0.0.0:12346
Connection established
(Startup) Running ControllerStartup
(Startup-1) Bouncing LS and Infer requests until startup is complete
(Startup-2) Querying current worker state
Clockwork page_size=16777216
Worker 0 1 GPUs 0 models
GPU 0 29.8819 GB (1912 pages) 0 loaded models

(Startup-3) Awaiting LoadModel requests from clients
```

By default, the controller will listen for incoming client connections on port 12346.  Run `./controller -h` for more options.

### Start a client

The controller started in the previous step will wait for incoming client connections.

On the client machine, from Clockwork's `build` directory, run:

```
./client volta02:12346 simple
```

Here, `volta02:12346` is the address of the controller started in step 2.  `simple` is the name of a workload.  Run `./client -h` to list available workloads.

You should see output similar to the following:

```
Running workload `simple` on volta02:12346
Client is sending inputs with requests.  Set CLOCKWORK_DISABLE_INPUTS=1 to disable inputs.
Connecting to clockwork @ volta02:12346
Connection established
Found 61 models in /home/jcmace/clockwork-modelzoo-volta
Clockwork initializing, retrying Controller initializing
throughput=300.20 min=3.82 max=22.49 mean=4.97
throughput=401.83 min=3.82 max=7.74 mean=4.96
throughput=398.53 min=3.82 max=7.86 mean=5.00
```

### Outputs

The above processes will run until terminated.  Each process will output summary statistics every 10 seconds.  For the above example, after a period of time, the outputs look as follows:

#### Worker
```
Starting Clockwork Worker
Loading Clockwork worker default config from /home/jcmace/clockwork/config/default.cfg
IO service thread listening on 0.0.0.0:12345
Received A0:GetWorkerState
Sending R0:GetWorkerState:
 page_size=16777216
gpus=
 GPU-0 weights_cache=21.5GB (1376 pages) io_pool=512.0MB workspace_pool=512.0MB 0 models currently on GPU
 GPU-1 weights_cache=21.5GB (1376 pages) io_pool=512.0MB workspace_pool=512.0MB 0 models currently on GPU
models=

Clock Skew=0  RTT=0  LdWts=0  Inf=0  Evct=0  || Total Pending=0  Errors=0
Received A0:LoadModelFromDisk model=0 [0.0, 16847613635838.0] /home/jcmace/clockwork-modelzoo-volta/resnet50_v2/model
Sending R0:LoadModelFromDisk input=602112 output=4000 weights=112.0 MB (7 pages) xfer=8.4 b1=3.3 b2=5.3 b4=7.5 b8=12.4 duration=14.9
Clock Skew=673306  RTT=53853  LdWts=4  Inf=2885  Evct=0  || Total Pending=2  Errors=0
Clock Skew=668080  RTT=55526  LdWts=0  Inf=4020  Evct=0  || Total Pending=2  Errors=0
Clock Skew=665855  RTT=54409  LdWts=0  Inf=3986  Evct=0  || Total Pending=1  Errors=0
```

#### Controller
```
(Startup-3) Awaiting LoadModel requests from clients
Client  --> Req0:LoadModel path=/home/jcmace/clockwork-modelzoo-volta/resnet50_v2/model
(Startup-4) LoadModelStage has begun
Worker <--  A0:LoadModelFromDisk model=0 [0.0, 16847613635839.0] /home/jcmace/clockwork-modelzoo-volta/resnet50_v2/model
Worker  --> R0:LoadModelFromDisk input=602112 output=4000 weights=112.0 MB (7 pages) xfer=8.4 b1=3.3 b2=5.3 b4=7.5 b8=12.4 duration=14.9
Client <--  Rsp0:LoadModel model_id=[0->1] input=602112 output=4000
Client  --> Req1:LS
(Startup-6) LoadModelStage complete.  Printing loaded models:
Clockwork page_size=16777216
Worker 0 2 GPUs 2 models
GPU 0 21.5 GB (1376 pages) 0 loaded models
GPU 1 21.5 GB (1376 pages) 0 loaded models
M-0 src=/home/jcmace/clockwork-modelzoo-volta/resnet50_v2/model input=602112 output=4000 weights=112.0 MB (7 pages) xfer=8.4 b1=3.3 b2=5.3 b4=7.5 b8=12.4
M-1 src=/home/jcmace/clockwork-modelzoo-volta/resnet50_v2/model input=602112 output=4000 weights=112.0 MB (7 pages) xfer=8.4 b1=3.3 b2=5.3 b4=7.5 b8=12.4

(Startup-end) Transitioning to scheduler
Client <--  Rsp1:LS error 5: Controller initializing
Created 2 models
Created 2 GPUs on 1 Workers
Total GPU capacity 2752 pages (1376 per GPU).
Total model pages 14 (0% oversubscription).
 * Admitting inference requests
GPU handler [ 0 ] started
GPU handler [ 1 ] started
Network Status:  Client ✔✔✔✔✔ Controller ✔✔✔✔✔ Workers (inputs generated by client)
W0-GPU0 LoadW min=8.37 max=8.37 mean=8.37 e2emean=9.67 e2emax=9.67 throughput=0.0 utilization=0.00 clock=[0-0] norm_max=0.00 norm_mean=0.00
Client throughput=0.0 success=100.00% min=13.9 max=13.9 mean=13.9
Network->Workers: 31.4MB/s (361 msgs) snd, 1.4MB/s (361 msgs) rcv,
W0-GPU0 LoadW min=8.37 max=8.37 mean=8.37 e2emean=17.80 e2emax=17.80 throughput=0.1 utilization=0.00 clock=[0-0] norm_max=0.00 norm_mean=0.00
W0-GPU0 Infer min=3.32 max=3.32 mean=3.32 e2emean=4.47 e2emax=7.25 throughput=200.6 utilization=0.67 clock=[1380-1380] norm_max=3.32 norm_mean=3.32
Client throughput=400.8 success=100.00% min=3.6 max=21.8 mean=4.7
Network->Workers: 34.8MB/s (402 msgs) snd, 1.6MB/s (402 msgs) rcv,
W0-GPU1 LoadW min=8.37 max=8.37 mean=8.37 e2emean=13.73 e2emax=17.79 throughput=0.2 utilization=0.00 clock=[0-0] norm_max=0.00 norm_mean=0.00
W0-GPU1 Infer min=3.32 max=3.32 mean=3.32 e2emean=4.47 e2emax=7.13 throughput=200.1 utilization=0.66 clock=[1380-1380] norm_max=3.32 norm_mean=3.32
```

#### Client

```
Clockwork initializing, retrying Controller initializing
throughput=300.20 min=3.82 max=22.49 mean=4.97
throughput=401.83 min=3.82 max=7.74 mean=4.96
throughput=398.53 min=3.82 max=7.86 mean=5.00
```
