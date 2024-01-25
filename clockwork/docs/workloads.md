# Workloads

Run `./client -h` to list the available workloads.

Note: descriptions below may be slightly out of date.

```
Usage: client [address] [workload] [workload parameters (if required)]
Available workloads with parameters:
         example
         fill_memory
                 creates 500 copies of resnet50, more than can fit in memory
                 100 of them are closed loop, 400 are gentle open loop
         spam [modelname]
                 default modelname is resnet50_v2
                 100 instances, each with 100 closed loop
         single-spam
                 resnet50_v2 x 1, with 1000 closed loop
         simple
         simple-slo-factor
                 3 models with closed-loop concurrency of 1
                 Updates each model's slo factor every 10 seconds
         simple-parametric models clients concurrency requests
                 Workload parameters:
                         models: number of model copies
                         clients: number of clients among which the models are partitioned
                         concurrency: number of concurrent requests per client
                         requests: total number of requests per client (for termination)
         poisson-open-loop num_models rate
                 Rate should be provided in requests/second
                 Rate is split across all models
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
         comparison_experiment
                 Description: runs multiple copies of resnet50_v2
                 Workload parameters:
                         num_models: (int, default 15) the number of models you're using
                         total_requests: (int, default 1000) the total requests across all models, per second
         comparison_experiment2
                 Description: closed-loop version of comparison experiment
                 Workload parameters:
                         num_models: (int, default 15) the number of models you're using
                         concurrency: (int, default 16) closed loop workload concurrency
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
```