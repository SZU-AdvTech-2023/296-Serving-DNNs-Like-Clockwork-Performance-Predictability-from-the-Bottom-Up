# Running Clockwork without GPUs

All Clockwork experiments can run without needing GPUs, using Clockwork's simulated worker.

Instead of executing `./worker`, execute `./worker_dummy`.  The `-n` flag can be used to specify the number of GPUs to simulate.  For example, `./worker_dummy -n 24` simulates 24 worker GPUs.

From the controller's perspective, the `worker_dummy` is indistinguishable from the `worker`.  It will respond to actions in the same way.  Instead of executing inferences, the `worker_dummy` uses our pre-profiled execution latency measurements, to calculate when an appropriate result should be sent.

Using the simulated worker is useful for calibrating workloads, testing experiment scripts, and testing the controller, without needing GPUs or multiple machines.

In addition, the simulated worker is much faster to initialize when using large numbers of models.  For example, 4000 models can take up to 10 minutes to load on a real GPU; it takes only seconds on the simulated worker.

The simulated worker has no limit on the number of models it can load either, enabling experiments with hypothetically larger GPU memory (e.g. 64GB or 128GB).