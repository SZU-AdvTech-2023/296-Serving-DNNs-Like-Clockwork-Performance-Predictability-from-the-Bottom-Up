# Controller

Run `./controller -h` to see available controller options

```
USAGE:
  controller [TYPE] [WORKERS] [OPTIONS]
DESCRIPTION
  Run the controller of the given TYPE. Connects to the specified workers. All
  subsequent options are controller-specific and passed to that controller.
TYPE
  DIRECT    Used for testing
  ECHO      Used for testing
  STRESS    Used for testing
  INFER5    The Clockwork Scheduler.  You should usually be using this.  Options:
       generate_inputs    (bool, default false)  Should inputs and outputs be generated if not present.  Set to true to test network capacity
       max_gpus           (int, default 100)  Set to a lower number to limit the number of GPUs.
       schedule_ahead     (int, default 10000000)  How far ahead, in nanoseconds, should the scheduler schedule.
       default_slo        (int, default 100000000)  The default SLO to use if client's don't specify slo_factor.  Default 100ms
       max_exec        (int, default 25000000)  Don't use batch sizes >1, whose exec time exceeds this number.  Default 25ms
       max_batch        (int, default 16)  Don't use batch sizes that exceed this number.  Default 16.
WORKERS
  Comma-separated list of worker host:port pairs.  e.g.:
    volta03:12345,volta04:12345,volta05:12345
OPTIONS
  -h,  --help
        Print this message
All other options are passed to the specific scheduler on init
```

* `generate_inputs` Used in experiments to change where inference inputs come from.  Used in conjunction with the client-side `CLOCKWORK_DISABLE_INPUTS`
* `max_gpus` The number of worker GPUs to actually use.  Useful when you have many machines, or multiple GPUs per machine, but only want to use 1 or a small number.
* `schedule_ahead` The scheduler will schedule the next 10ms of actions by default.  This parameter adjusts that.
* `default_slo` By default all requests have a 100ms SLO.  This parameter adjusts that.  The default SLO only applies if a request doesn't specify its own, request-specific SLO.  Most workloads use the default SLO.
* `max_exec` Upper limit on model execution time
* `max_batch` Upper limit on batch sizes
