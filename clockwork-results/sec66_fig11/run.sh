#!/usr/bin/bash

################################################################################
# Trying out new scalability experiments
################################################################################

SSH_PORT=22

# Clockwork docker uses port 2200
# SSH_PORT=2200 

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )" # Absolute path to this script 
EXP_DIR="${SCRIPTPATH}"                         # Experiment dir

exp_name="azure_scalability_exp_b"                     # Codename
timestamp=`date "+%Y-%m-%d-%H-%M-%S"`                  # Start timestamp
logdir="/home/hzq/results/${exp_name}/log/${timestamp}" # Log dir

CLOCKWORK_MODEL_DIR="/clockwork-modelzoo-volta"
CLOCKWORK_DISABLE_INPUTS=1
AZURE_TRACE_DIR="/azure-functions"
CLOCKWORK_BUILD="/clockwork/build"

if [ $# -gt 0 ]; then logdir=${1}; fi # Log dir may be specified as an argument
mkdir -p ${logdir}                    # Create log dir locally

################################################################################

client="172.16.201.4"      # Client (need not have any GPUs)
controller="172.16.201.2"  # Controller (need not have any GPUs)

# Workers (need to have GPUs)
declare -a workers=( "172.16.200.3" )

# Prepare the argument string for the controller containing all worker hostnames
len1=${#workers[@]}
worker_ports="${workers[$i]}:12345"  
for (( i=1; i<${len1}; i++ ))
do
  worker_ports="${worker_ports},${workers[$i]}:12345" 
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

SECONDS=0
printf "\nStarting Exp. ${exp_name}\n"
printf "The experiment log directory is ${logdir}\n"

config=0
declare -a num_workers_opt=( "10" "20" "30" "40" "50" "60")
for num_workers in "${num_workers_opt[@]}"; do

# Iterate over all configurations
config=$(( ${config} + 1 ))
echo ""
echo "Starting experiment run ${config}"

#-------------------------------------------------------------------------
# Clockwork client process usage:
#   client [address] [workload] [workload parameters (if required)]
# In this experiment, we use the following workload:
#   azure_sclability_exp
# The workload parameters are:
#   num_workers load_factor_min load_factor_max load_factor_inc load_factor_period memory_load_factor
#-------------------------------------------------------------------------
workload="azure_scalability_exp"  # NOTE: Leave only trailing arguments empty
load_factor_min="0.1"             # Increase load factor from 0.1 ...
load_factor_max="4"               # to at max 4 ...
load_factor_inc="1.5"             # in steps of 0.1, 0.15, 0.225, ...
load_factor_period="60"           # every 60 seconds
memory_load_factor="4"            # Default

#-------------------------------------------------------------------------
# Clockwork controller process usage:
#   controller [TYPE] [WORKERS] [OPTIONS]
# Description:
#   Run the controller of the given TYPE. Connects to the specified workers.
#   All subsequent options are controller-specific and passed to that controller.
# In this experiment, we use the following controller TYPE:
#   INFER4 (Up-to-date scheduler with loads and infers)
# Options specific to controller INFER4:
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
scheduler="INFER5"        # INFER5 is the most stable one!
generate_inputs="0"       # False, since workers generate inputs
max_gpus="${num_workers}" # Scalability!
schedule_ahead="5000000"  # Default when generate_inputs is False
default_slo="100000000"   # 100ms
max_exec="25000000"       # Default
max_batch="8"             # Default

################################################################################

# Argument string for the Clockwork client process
client_args="${num_workers} ${load_factor_min} ${load_factor_max}"
client_args+=" ${load_factor_inc} ${load_factor_period} ${memory_load_factor}"

# Argument string for the Clockwork controller process
controller_args="${generate_inputs} ${max_gpus} ${schedule_ahead}"
controller_args+=" ${default_slo} ${max_exec} ${max_batch}"

echo ""
echo "Config ${config} client arguments: ${client_args}"
echo "Config ${config} controller arguments: ${controller_args}"

echo ""
echo "num_models = ${num_models}"
echo "rate_min = ${load_factor_min}"
echo "rate_max = ${load_factor_max}"
echo "rate_factor = ${load_factor_inc}"
echo "rate_op = mul"
echo "period = ${load_factor_period}"
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
	remote_cmd="pkill -f ${CLOCKWORK_BUILD}/worker_dummy"
	echo "Remote worker cmd: ${remote_cmd}"
	$(${ssh_cmd_prefix} ${worker} "${remote_cmd}")
done

echo "Sleeping 30s"
sleep 30s

# Start the Clockwork worker process remotely on each worker host
# Keep track of all worker process PIDs
# Sleep for a while, since the controller expects all workers to be ready
worker_pids=()
for worker in "${workers[@]}"
do
	echo "Start Clockwork worker remotely on host ${worker}"
	logfile="${logdir}/file=worker-${worker}_${config}.log"
	remote_cmd="${mklogdir} ${env_vars} nohup ${CLOCKWORK_BUILD}/worker_dummy -n ${max_gpus}"
  remote_cmd+=" > ${logfile} 2>&1 < /dev/null & echo \$!"
	echo "Remote worker cmd: ${remote_cmd}"
	worker_pid=$(${ssh_cmd_prefix} ${worker} "${remote_cmd}")
	worker_pids+=(${worker_pid})
done
echo "All worker processes' PIDs ${worker_pids[@]}"
echo "Sleeping 10s"
sleep 10s

# Start the Clockwork controller process remotely on the specified host
# Keep track of the controller process PID
# Sleep for a while, so that the controller is ready to serve requests
echo ""
echo "Start Clockwork controller remotely on host ${controller}"
logfile="${logdir}/file=controller_${config}.log"
remote_cmd="${mklogdir} ${env_vars} nohup ${CLOCKWORK_BUILD}/controller"
remote_cmd+=" ${scheduler} ${worker_ports} ${controller_args}"
remote_cmd+=" > ${logfile} 2>&1 < /dev/null & echo \$!"
echo "Remote controller cmd: ${remote_cmd}"
CONTROLLER_PID=$(${ssh_cmd_prefix} ${controller} "${remote_cmd}")
echo "Controller process's PID ${CONTROLLER_PID}"
echo "Sleeping 5s"
sleep 5s

# Start the Clockwork client process remotely on the specified host
# Since our workload is finitely long, wait for the client to terminate
echo ""
echo "Starting Clockwork client remotely on host ${client}"
logfile="${logdir}/file=client_${config}.log"
remote_cmd="${mklogdir} ${env_vars} nohup ${CLOCKWORK_BUILD}/client"
remote_cmd+=" ${controller}:12346 ${workload} ${client_args}"
remote_cmd+=" > ${logfile} 2>&1 < /dev/null & echo \$!"
printf "Remote client cmd: ${remote_cmd}\n"
CLIENT_PID=$(${ssh_cmd_prefix} ${client} "${remote_cmd}")
echo "Client process's PID ${CLIENT_PID}"

sleep 10m
# The controller request telemetry file is at ${logdir}/clockwork_request_log.tsv
# Copy it to localhost after the experiment
echo ""
echo "Copying controller's request telemetry file to ${logdir}"
telemetryfile_dest="${logdir}/file=controller_${config}_request.tsv"
telemetryfile_src="${logdir}/clockwork_request_log.tsv"
$(scp -P ${SSH_PORT} ${username}@${controller}:${telemetryfile_src} ${telemetryfile_dest})

# The controller action telemetry file is at ${logdir}/clockwork_action_log.tsv
# Copy it to localhost after the experiment
echo ""
echo "Copying controller's action telemetry file to ${logdir}"
telemetryfile_dest="${logdir}/file=controller_${config}_action.tsv"
telemetryfile_src="${logdir}/clockwork_action_log.tsv"
$(scp -P ${SSH_PORT} ${username}@${controller}:${telemetryfile_src} ${telemetryfile_dest})

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

echo "Sleeping 1m"
sleep 1m
echo ""

done

duration=$SECONDS
echo ""
echo "Exiting"
printf "Roughly $(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed.\n"

################################################################################
