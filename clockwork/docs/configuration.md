# Clockwork Configuration

## Summary

### Required Configuration

1. CLOCKWORK_MODEL_DIR (on workers)
2. CLOCKWORK_LOG_DIR (on controller)

### Optional Configuration

1. AZURE_TRACE_DIR (on client, if using `azure` workload)
2. CLOCKWORK_DISABLE_INPUTS (on client, depending on experiment)
3. CLOCKWORK_CONFIG_FILE (on workers, if overriding defaults from `config/defaults.cfg`)

## Details

## Required: CLOCKWORK_MODEL_DIR

This is required by Clockwork's `./worker` process.

`CLOCKWORK_MODEL_DIR` should point to a local directory containing compiled models.

Pre-compiled models used by Clockwork can be found in the [`clockwork-modelzoo-volta`](https://gitlab.mpi-sws.org/cld/ml/clockwork-modelzoo-volta) repository.

The process of compiling models is not fully automated currently (please contribute!).

## Recommended: CLOCKWORK_LOG_DIR

This is required by Clockwork's `./controller` process.

`CLOCKWORK_LOG_DIR` should point to a directory where the controller can write its output request logs.  Be aware that for long experiments, these files can be GB large.

If not specified or left blank, Clockwork's controller will write to `/local`.  If this does not exist on your machine or is not writable, the controller will not output anything.

Please ensure the directory exists; Clockwork will not create the directory for you.  Upon startup, the controller will print the location it is writing its request and action logs to.

## Optional: AZURE_TRACE_DIR

This is required by Clockwork's `./client` process if you are running the `azure` workload.

`AZURE_TRACE_DIR` should point to a local directory containing Microsoft Azure Functions traces.

The traces can be found in the [`azure-functions`](https://gitlab.mpi-sws.org/cld/trace-datasets/azure-functions) repository.

The trace dataset is a clone of the `AzureFunctionsDataset2019` from Microsoft Azure, which can be found by following the instructions on Microsoft's GitHub repository at [this](https://github.com/Azure/AzurePublicDataset) link.

## Optional: CLOCKWORK_DISABLE_INPUTS

This is used by Clockwork's `./client` process.

For some experiments you will want to generate model inputs at the controller rather than sending them over the network.

Setting `CLOCKWORK_DISABLE_INPUTS=1` will disable clients from sending inputs.

## Optional: CLOCKWORK_CONFIG_FILE

This is used by Clockwork's `./worker` process.

Clockwork has a configuration file located under `config/default.cfg`.  You can specify your own configuration elsewhere and set its path using `CLOCKWORK_CONFIG_FILE`.

The default setting is listed below:

```
WorkerConfig:
{
	memory_settings:
	{
		weights_cache_size = 23085449216L;
		weights_cache_page_size = 16777216L;
		io_pool_size = 536870912L;
		workspace_pool_size = 536870912L;
		host_io_pool_size = 536870912L;
	};

	telemetry_settings:
	{
		enable_task_telemetry = false;
		enable_action_telemetry = false;
	};

	log_dir:
	{
		telemetry_log_dir = "./";
	};

	allow_zero_size_inputs = true;

};
```

### weights_cache_size

This specifies how much GPU memory should be used for model weights.  This memory is pre-allocated by workers on worker startup.  **The default value for `weights_cache_size` assumes you have 32GB GPU memory**.

### memory_settings

The other memory settings relate to how memory is allocated internally in Clockwork, and do not need to be modified.

### allow_zero_size_inputs

If `false`, workers will reject any actions lacking a payload.  If `true` and an infer action is received without a payload, workers will generate an input.  This feature is only required for [sec65_table](https://gitlab.mpi-sws.org/cld/ml/clockwork-results/-/tree/master/sec65_table).

### telemetry_settings and log_dir

Workers can log telemetry for debugging purposes.  This does not need to be enabled for any experiments.