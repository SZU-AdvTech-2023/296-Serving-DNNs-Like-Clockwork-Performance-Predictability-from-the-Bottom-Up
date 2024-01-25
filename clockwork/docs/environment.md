# Environment Setup

Clockwork is a high-performance system that depends upon predictability.  There are various tweaks to your environment that will make executions more predictable.  These environment modifications should be made for Clockwork's worker, controller, and client processes. Some are optional but recommended.

## Check your environment

From Clockwork's build folder, run:
```
./profile [check]
```

This will report the status of your environment and tell you any modifications that should be made

## 1. Increase resource limits (memlock, nofile, rtprio)

Limits on the number of open files, and the amount of page-locked memory, reduce the total number of DNNs clockwork can keep in memory at any point in time.  A limit of 1024 is too low.  A limit of 16k or higher is acceptable.

Limits can be checked with the `ulimit` command (`ulimit -aH` lists hard limits, `ulimit -a` lists current)

Increase the `RLIMIT_NOFILE` (number of open files) and `RLIMIT_MEMLOCK` (amount of page-locked memory) to unlimited:
1. Open `/etc/security/limits.conf`
2. Add the following lines:
```
*            hard   memlock           unlimited
*            soft   memlock           unlimited
*            hard   nofile            unlimited
*            soft   nofile            unlimited
*            hard   rtprio            unlimited
*            soft   rtprio            unlimited
```
Note: for MPI cluster machines with the default Debian distribution, you will also need to modify `/etc/security/limits.d/mpiprefs.conf`

3. Modify `/etc/systemd/user.conf` and `/etc/systemd/system.conf` to add:
```
DefaultLimitNOFILE=1048576
```
4. Restart to take effect
5. Upon restarting, use Clockwork's `./profile [check]` to check if the settings took effect

## 2. Increase mmap limits

Clockwork uses a lot of shared objects, and we need to increase the mmap limit.  As root, run
```
/sbin/sysctl -w vm.max_map_count=10000000
```

In general you can check mmap limits with:
```
/sbin/sysctl vm.max_map_count
```

This normally does not require a restart.  You can check using Clockwork's `./profile [check]`.

This normally does not require a restart.  You can check using Clockwork's `./profile [check]`.

## 3. GPU Settings

## 3.1. Disable CUDA JIT

Prevent CUDA from caching compiled kernels (note: the models used by Clockwork do not compile to PTX anyway, but if choose to compile JITable models, this setting is important)
```
export CUDA_CACHE_DISABLE=1
```

### 3.1 Enable persistence mode.

```
nvidia-smi -pm 1
```

NOTE: This must be done on every restart

### 3.2 Enable exclusive process mode

```
nvidia-smi -c 3
```

NOTE: This must be done on every restart


### 3.3 Optional: Disable auto boost

```
nvidia-smi --auto-boost-default=DISABLED
```

NOTE: This must be done on every restart

### 3.4 Optional: Configure GPU clocks

You can specify which clock frequencies to use.  This does not override built-in temperature auto-scaling.

List available GPU clock frequencies
```
nvidia-smi -q -d SUPPORTED_CLOCKS
```

Pick a memory and graphics clock frequency (usually the highest), e.g. on volta machines:
```
    Supported Clocks
        Memory                      : 877 MHz
            Graphics                : 1380 MHz
```

Set the default application clock and system clock to those highest values, e.g. on volta machines:
```
nvidia-smi -ac 877,1380
nvidia-smi -lgc 1380
```

NOTE: This must be done on every restart

## 4. Optional: Disable CPU frequency autoscaling

Set the "performance" governor to prevent CPU clock scaling
```
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```