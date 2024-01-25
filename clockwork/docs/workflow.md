# Clockwork Workflow

Clockwork is not feature complete, but we welcome contributions from others!  Please contact our mailing list clockwork-users@googlegroups.com

Currently, Clockwork's workflow is centered on running experiments where the entire system is spun up and torn down each time.  Clockwork does not persist anything between killing and restarting instances.

This page describes the different steps involved in Clockwork's end-to-end workflow.  It assumes Clockwork is built and runnable and that appropriate machines have been provisioned.

## Starting Processes

### 1. User starts worker processes on worker machines

When worker processes are started, they do not load any models.  They listen for a connection from the controller.  All model loading will come via instructions from the controller.

### 2. User starts controller process on controller machine

When the controller process is started, it connects to the specified workers, then listens for incoming client connections.  The controller does not load any models by default.  Instead, models are loaded on request by clients.

### 3. User starts client process(es) on client machine(s)

When a client is started, it connects to the controller.  The client will first load the models it needs to run its workload, then send inference requests.  Most clients run forever, until they are manually terminated.

## Controller Phases

### 1. Wait Phase

When the controller process is started, it waits for incoming client requests.  Eventually a client will connect and send a `LoadModel` command for one or more models.  The first `LoadModel` command transitions the controller into the Load Phase.

### 2. Load Phase

The controller continues to receive `LoadModel` commands from clients, forwarding them to workers.  `LoadModel` commands simply specify a remote model path; the workers load those models from disk into memory.

If the models do not exist on the worker machines, appropriate errors are propagated back to the client.

Models must be placed on worker machines ahead of time.  If you are interested in contributing to the `UploadModel` API and automating this process please contact us!

If clients send other requests, such as `Infer` requests, they will be bounced.

### 3. Load Phase Timeout

Once all `LoadModel` requests have completed, the controller counts down for a configurable period of inactivity (1s by default).  At this point, the controller will no longer accept `LoadModel` commands.  `Infer` requests will start being accepted.

### 4. Infer Phase

This is the main controller phase.  Infer requests are scheduled by the scheduler on the workers.  Currently, the set of loaded models is fixed.  (This isn't fundamental; if you wish to contribute this feature, please contact us!)

## Outputs

Clockwork's controller outputs telemetry about client requests, and about actions sent to workers.  See [Telemetry](telemetry.md) for more information.

