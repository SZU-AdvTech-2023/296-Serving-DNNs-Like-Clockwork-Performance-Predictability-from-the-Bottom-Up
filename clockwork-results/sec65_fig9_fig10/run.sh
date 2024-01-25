#!/usr/bin/bash

################################################################################
# Regarding experiments in Section 6.5 Is Clockwork Predictable?
# Clockwork with realistic workloadsm Predictable executions (maf-exp-1-and-2)
################################################################################

SSH_PORT=22

# Clockwork docker uses port 2200
# SSH_PORT=2200 

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )" # Absolute path to this script 
EXP_DIR="${SCRIPTPATH}"                         # Experiment dir

exp_name="maf-exp-1-and-2"                             # Codename
timestamp=`date "+%Y-%m-%d-%H-%M-%S"`                  # Start timestamp
logdir="/home/hzq/results/${exp_name}/log/${timestamp}" # Log dir
CLOCKWORK_DISABLE_INPUTS=1
CLOCKWORK_BUILD="/clockwork/build"
AZURE_TRACE_DIR="/azure-functions"
CLOCKWORK_MODEL_DIR="/clockwork-modelzoo-volta"
if [ $# -gt 0 ]; then logdir=${1}; fi # Log dir may be specified as an argument
mkdir -p ${logdir}                    # Create log dir locally

################################################################################

client="172.16.201.4"      # Client (need not have any GPUs)
controller="172.16.201.2"  # Controller (need not have any GPUs)

# Workers (need to have GPUs)
declare -a workers=("172.16.201.5" "172.16.201.3" "172.16.200.3" "172.16.200.4" "172.16.200.5" "172.16.200.2")

# Prepare the argument string for the controller containing all worker hostnames
len1=${#workers[@]}
worker_arg_for_controller="${workers[$i]}:12345"  
for (( i=1; i<${len1}; i++ ))
do
  worker_arg_for_controller="${worker_arg_for_controller},${workers[$i]}:12345" 
done

# Username and password-free ssh command prefix
username="root"
ssh_cmd_prefix="ssh -p ${SSH_PORT} -o StrictHostKeyChecking=no -l ${username}"

mklogdir="mkdir -p ${logdir}; " # Command to create log dir on each machine

# It's a bit tricky to pass environment variables through SSH
# For now, we pass CLOCKWORK_MODEL_DIR, CLOCKWORK_LOG_DIR, and
# CLOCKWORK_DISABLE_INPUTS, and AZURE_TRACE_DIR variables directly
# AFAIK, other variables used during compilation need not be passed
# We will add these env_vars to all ssh commands
env_vars="export CLOCKWORK_MODEL_DIR=${CLOCKWORK_MODEL_DIR}; "
env_vars+="export CLOCKWORK_LOG_DIR=${logdir}; "
env_vars+="export CLOCKWORK_DISABLE_INPUTS=${CLOCKWORK_DISABLE_INPUTS}; "
env_vars+="export AZURE_TRACE_DIR=${AZURE_TRACE_DIR}; "

################################################################################

#-------------------------------------------------------------------------
# Clockwork client process usage:
#   client [address] [workload] [workload parameters (if required)]
# In this experiment, we use the following workload:
#   azure
# The workload parameters are:
#   num_workers use_all_models load_factor memory_load_factor interval trace randomise
# Following is the description of workload parameters:
#   num_workers (int, default 1) the number of workers you're using
#   use_all_models: (bool, default 1) load all models or just resnet50_v2
#   load_factor: (float, default 1.0)
#     the workload will generate approximately this much load to the system,
#     e.g. 0.5 will load by approximately 1/2; 2.0 will overload by a factor of 2
#   memory_load_factor: (1, 2, 3, or 4; default 4):
#     1: loads enough models to fit in 1 GPU's memory
#     2: loads enough models to exceed 1 worker's (2 GPUs) memory by a factor of 2
#     3: loads enough models to fit in all GPUs aggregate memory
#     4: loads the maximum possible models (upper limit of 50 copies)
#   interval: (int, default 60) interval duration in seconds
#   trace: (int, 1 to 13 inclusive, default 1) trace ID to replay
#   randomise: (bool, default false) randomize each client's starting point in the trace
#-------------------------------------------------------------------------
workload="azure"                       # NOTE: Leave only trailing arguments empty
declare -a num_workers_opt=("8")       # (1 GPU x 6 machines) + workload correction
declare -a use_all_models_opt=("1")     # Default
declare -a load_factor_opt=("1")        # Default
declare -a memory_load_factor_opt=("2") # Default
declare -a interval_opt=("")           # Default
declare -a trace_opt=("")              # Default
declare -a randomise_opt=("")          # Default

#-------------------------------------------------------------------------
# Clockwork controller process usage:
#   controller [TYPE] [WORKERS] [OPTIONS]
# Description:
#   Run the controller of the given TYPE. Connects to the specified workers.
#   All subsequent options are controller-specific and passed to that controller.
# In this experiment, we use the following controller TYPE:
#   INFER5 (Up-to-date scheduler with loads and infers)
# Options specific to controller INFER5:
#   generate_inputs (bool, default false)
#     Should inputs and outputs be generated if not present.
#     Set to true to test network capacity.
#   max_gpus (int, default 100)
#     Set to a lower number to limit the number of GPUs.
#   schedule_ahead (int, default 10000000)
#     How far ahead, in nanoseconds, should the scheduler schedule.
#     If generate_inputs is set to true, the default value for this is 15ms, otherwise 5ms.
#   default_slo (int, default 100000000)
#     The default SLO to use if client's don't specify slo_factor.
#     Default 100ms
#   max_exec (int, default 25000000)
#     Don't use batch sizes >1, whose exec time exceeds this number.
#     Default 25ms
#   max_batch (int, default 8)
#     Don't use batch sizes that exceed this number. Default 8.
#-------------------------------------------------------------------------
scheduler="INFER5"                 # NOTE: Leave only trailing arguments empty
declare -a generate_inputs_opt=(1) # Controller generates inputs
declare -a max_gpus_opt=(6)       # Assuming 2 GPUs x 6 workers
# declare -a schedule_ahead_opt=("3000000") # Default
declare -a shedule_ahead_opt=(3000000 4000000 5000000 6000000 7000000)
declare -a default_slo_opt=("")    # Default
declare -a max_exec_opt=("")       # Default
declare -a max_batch_opt=("")      # Default

################################################################################

SECONDS=0
timeout_duration="4h"
printf "\nStarting Exp. ${exp_name} for ${timeout_duration}\n"
printf "The experiment log directory is ${logdir}\n"

config=0

# Iterate over all configurations identifed by the workload and controller
# option arrays defined above
for num_workers in "${num_workers_opt[@]}"; do
for use_all_models in "${use_all_models_opt[@]}"; do
for load_factor in "${load_factor_opt[@]}"; do
for memory_load_factor in "${memory_load_factor_opt[@]}"; do
for interval in "${interval_opt[@]}"; do
for trace in "${trace_opt[@]}"; do
for randomise in "${randomise_opt[@]}"; do
for generate_inputs in "${generate_inputs_opt[@]}"; do
for max_gpus in "${max_gpus_opt[@]}"; do
# for schedule_ahead in "${schedule_ahead_opt[@]}"; do
for default_slo in "${default_slo_opt[@]}"; do
for max_exec in "${max_exec_opt[@]}"; do
for max_batch in "${max_batch_opt[@]}"; do
config=$(( ${config} + 1 ))
echo ""
echo "Starting experiment run ${config}"
# for schedule_ahead in "${shedule_ahead_opt[@]}"; do
# 	client_args="${num_workers} ${use_all_models} ${load_factor}"
# 	client_args+=" ${memory_load_factor} ${interval} ${trace} ${randomise}"
# 	controller_args="${generate_inputs} ${max_gpus} ${schedule_ahead}"
# 	controller_args+=" ${default_slo} ${max_exec} ${max_batch}"

# 	echo ""
# 	remote_cmd="pkill -f ${CLOCKWORK_BUILD}/client"
# 	$(${ssh_cmd_prefix} ${client} "${remote_cmd}")
# 	for worker in "${workers[@]}"
# 	do
# 		echo "Stop any leftover Clockwork worker processes on host ${worker}"
# 		remote_cmd="pkill -f ${CLOCKWORK_BUILD}/worker"
# 		echo "Remote worker cmd: ${remote_cmd}"
# 		$(${ssh_cmd_prefix} ${worker} "${remote_cmd}")
# 	done
# 	echo "Sleeping 30s"
# 	sleep 30s
# 	worker_pids=()
# 	for worker in "${workers[@]}"
# 	do
# 		logfile="${logdir}/file=worker-${worker}_${config}.log"
# 		remote_cmd="${mklogdir} ${env_vars} nohup ${CLOCKWORK_BUILD}/worker"
# 		remote_cmd+=" > ${logfile} 2>&1 < /dev/null & echo \$!"
# 		worker_pid=$(${ssh_cmd_prefix} ${worker} "${remote_cmd}")
# 		worker_pids+=(${worker_pid})
# 	done
# 	sleep 1m
# 	logfile="${logdir}/file=controller_${config}.log"
# 	remote_cmd="${mklogdir} ${env_vars} nohup ${CLOCKWORK_BUILD}/controller"
# 	remote_cmd+=" ${scheduler} ${worker_arg_for_controller} ${controller_args}"
# 	remote_cmd+=" > ${logfile} 2>&1 < /dev/null & echo \$!"
# 	CONTROLLER_PID=$(${ssh_cmd_prefix} ${controller} "${remote_cmd}")
# 	sleep 1m
# 	logfile="${logdir}/file=client_${config}.log"
# 	remote_cmd="${mklogdir} ${env_vars} nohup ${CLOCKWORK_BUILD}/client"
# 	remote_cmd+=" ${controller}:12346 ${workload} ${client_args}"
# 	remote_cmd+=" > ${logfile} 2>&1 < /dev/null & echo \$!"
# 	CLIENT_PID=$(${ssh_cmd_prefix} ${client} "${remote_cmd}")
# 	echo "Client process's PID ${CLIENT_PID}"
# 	echo "Sleeping 20m"
# 	sleep 20m

# 	remote_cmd="kill -2 ${CLIENT_PID}"
# 	$(${ssh_cmd_prefix} ${client} "${remote_cmd}") 
# 	action_telemetryfile="${logdir}/file=controller_action_${schedule_ahead}.tsv"
# 	default_telemetryfile="${logdir}/clockwork_action_log.tsv"
# 	$(scp -P ${SSH_PORT} ${username}@${controller}:${default_telemetryfile} ${action_telemetryfile})

# 	remote_cmd="kill -2 ${CONTROLLER_PID}"
# 	$(${ssh_cmd_prefix} ${controller} "${remote_cmd}")

# 	len1=${#workers[@]}
# 	len2=${#worker_pids[@]}
# 	for (( i=0; i<${len1}; i++ ))
# 	do
# 		worker=${workers[$i]}
# 		worker_pid=${worker_pids[$i]}
# 		remote_cmd="kill -9 ${worker_pid}"
# 	$(${ssh_cmd_prefix} ${worker} "${remote_cmd}")
# 	done

# 	echo "Sleeping 1m"
# 	sleep 1m
# 	echo ""
# done

# schedule_ahead=$(python3 a.py ${logdir})
# echo ${schedule_ahead}
# echo ""
schedule_ahead=3000000
# Argument string for the Clockwork client process
client_args="${num_workers} ${use_all_models} ${load_factor}"
client_args+=" ${memory_load_factor} ${interval} ${trace} ${randomise}"

# Argument string for the Clockwork controller process
controller_args="${generate_inputs} ${max_gpus} ${schedule_ahead}"
controller_args+=" ${default_slo} ${max_exec} ${max_batch}"

echo ""
echo "Config ${config} client arguments: ${client_args}"
echo "Config ${config} controller arguments: ${controller_args}"

echo ""
echo "num_workers = ${num_workers}"
echo "use_all_models = ${use_all_models}"
echo "load_factor = ${load_factor}"
echo "memory_load_factor = ${memory_load_factor}"
echo "interval = ${interval}"
echo "trace = ${trace}"
echo "randomise = ${randomise}"
echo "generate_inputs = ${generate_inputs}"
echo "max_gpus = ${max_gpus}"
echo "schedule_ahead = ${schedule_ahead}"
echo "default_slo = ${default_slo}"
echo "max_exec = ${max_exec}"
echo "max_batch = ${max_batch}"

# Stop any leftover Clockwork worker processes on the workers
echo ""
for worker in "${workers[@]}"
do
	echo "Stop any leftover Clockwork worker processes on host ${worker}"
	remote_cmd="pkill -f ${CLOCKWORK_BUILD}/worker"
	echo "Remote worker cmd: ${remote_cmd}"
	$(${ssh_cmd_prefix} ${worker} "${remote_cmd}")
done

echo "Sleeping 2m"
sleep 2m

# Start the Clockwork worker process remotely on each worker host
# Keep track of all worker process PIDs
# Sleep for a while, since the controller expects all workers to be ready
worker_pids=()
for worker in "${workers[@]}"
do
	echo "Start Clockwork worker remotely on host ${worker}"
	logfile="${logdir}/file=worker-${worker}_${config}.log"
	remote_cmd="${mklogdir} ${env_vars} nohup ${CLOCKWORK_BUILD}/worker"
  remote_cmd+=" > ${logfile} 2>&1 < /dev/null & echo \$!"
	echo "Remote worker cmd: ${remote_cmd}"
	worker_pid=$(${ssh_cmd_prefix} ${worker} "${remote_cmd}")
	worker_pids+=(${worker_pid})
done
echo "All worker processes' PIDs ${worker_pids[@]}"
echo "Sleeping 30s"
sleep 30s

# Start the Clockwork controller process remotely on the specified host
# Keep track of the controller process PID
# Sleep for a while, so that the controller is ready to serve requests
echo ""
echo "Start Clockwork controller remotely on host ${controller}"
logfile="${logdir}/file=controller_${config}.log"
remote_cmd="${mklogdir} ${env_vars} nohup ${CLOCKWORK_BUILD}/controller"
remote_cmd+=" ${scheduler} ${worker_arg_for_controller} ${controller_args}"
remote_cmd+=" > ${logfile} 2>&1 < /dev/null & echo \$!"
echo "Remote controller cmd: ${remote_cmd}"
CONTROLLER_PID=$(${ssh_cmd_prefix} ${controller} "${remote_cmd}")
echo "Controller process's PID ${CONTROLLER_PID}"
echo "Sleeping 30s"
sleep 30s

# Start the Clockwork client process remotely on the specified host
# Keep track of the client process PID
echo ""
echo "Starting Clockwork client remotely on host ${client}"
logfile="${logdir}/file=client_${config}.log"
remote_cmd="${mklogdir} ${env_vars} nohup ${CLOCKWORK_BUILD}/client"
remote_cmd+=" ${controller}:12346 ${workload} ${client_args}"
remote_cmd+=" > ${logfile} 2>&1 < /dev/null & echo \$!"
printf "Remote client cmd: ${remote_cmd}\n"
CLIENT_PID=$(${ssh_cmd_prefix} ${client} "${remote_cmd}")
echo "Client process's PID ${CLIENT_PID}"
echo "Sleeping ${timeout_duration}"
sleep ${timeout_duration}

# Stop the client process after the specified duration
echo ""
echo "Stop Clockwork client on host ${client}"
remote_cmd="kill -2 ${CLIENT_PID}"
$(${ssh_cmd_prefix} ${client} "${remote_cmd}")

# The controller request telemetry file is at /local/clockwork_request_log.tsv
# Copy it to localhost after the experiment
echo ""
echo "Copying controller's request telemetry file to ${logdir}"
request_telemetryfile="${logdir}/file=controller_${config}_request.tsv"
default_telemetryfile="${logdir}/clockwork_request_log.tsv"
$(scp -P ${SSH_PORT} ${username}@${controller}:${default_telemetryfile} ${request_telemetryfile})

# The controller action telemetry file is at /local/clockwork_action_log.tsv
# Copy it to localhost after the experiment
echo ""
echo "Copying controller's action telemetry file to ${logdir}"
action_telemetryfile="${logdir}/file=controller_${config}_action.tsv"
default_telemetryfile="${logdir}/clockwork_action_log.tsv"
$(scp -P ${SSH_PORT} ${username}@${controller}:${default_telemetryfile} ${action_telemetryfile})

# Stop the controller process
echo ""
echo "Stop Clockwork controller on host ${controller}"
remote_cmd="kill -2 ${CONTROLLER_PID}"
$(${ssh_cmd_prefix} ${controller} "${remote_cmd}")

# Stop all worker processes
echo ""
echo "Stop Clockwork workers on hosts ${workers[@]}"
len1=${#workers[@]}
len2=${#worker_pids[@]}
for (( i=0; i<${len1}; i++ ))
do
	worker=${workers[$i]}
	worker_pid=${worker_pids[$i]}
	remote_cmd="kill -9 ${worker_pid}"
  $(${ssh_cmd_prefix} ${worker} "${remote_cmd}")
done

echo "Sleeping 5m"
sleep 5m
echo ""

done
done
done
done
done
done
done
done
done
done
done
done
done

duration=$SECONDS
echo ""
echo "Exiting"
printf "Roughly $(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed.\n"

################################################################################
