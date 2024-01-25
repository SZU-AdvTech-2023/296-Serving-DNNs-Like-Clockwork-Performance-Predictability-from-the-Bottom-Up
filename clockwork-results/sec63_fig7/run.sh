#!/usr/bin/bash

################################################################################
# Regarding experiments in Section 6.3 How Low Can Clockwork Go? (slo-exp-1)
################################################################################

SSH_PORT=22

# Clockwork docker uses port 2200
# SSH_PORT=2200 

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )" # Absolute path to this script 
EXP_DIR="${SCRIPTPATH}"                         # Experiment dir

exp_name="slo-exp-1"                                   # Codename
timestamp=`date "+%Y-%m-%d-%H-%M-%S"`                  # Start timestamp
logdir="/home/hzq/results/${exp_name}/log/${timestamp}" # Log dir
CLOCKWORK_MODEL_DIR="/clockwork-modelzoo-volta"
CLOCKWORK_DISABLE_INPUTS=0
CLOCKWORK_BUILD="/clockwork/build"
AZURE_TRACE_DIR="/azure-functions"

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
# CLOCKWORK_DISABLE_INPUTS variables directly
# AFAIK, other variables used during compilation need not be passed
# We will add these env_vars to all ssh commands
env_vars="export CLOCKWORK_MODEL_DIR=${CLOCKWORK_MODEL_DIR}; "
env_vars+="export CLOCKWORK_LOG_DIR=${logdir}; "
env_vars+="export CLOCKWORK_DISABLE_INPUTS=${CLOCKWORK_DISABLE_INPUTS}; "

################################################################################

model="resnet50_v2"                # Model type
declare -a dists=("poisson")       # Request arrival type
declare -a rates=(600 1200 2400)   # Request rate (requests per second)
declare -a num_models_vals=(12 48) # Number of independent model instances
declare -a shedule_ahead_opt=(3000000 4000000 5000000 6000000 7000000)

slo_start=2.9    # Initial SLO = execution latency of batch-1 ResNet50 x 1
slo_end=200    # Maximum SLO = execution latency of batch-1 ResNet50 x 100 
slo_factor=1.5 # Each SLO increment is by 1.5 times, i.e., by 50%
slo_op="mul"   # SLO is increased in a multiplicative manner
period=30      # Each SLO increment happens every 30 seconds

default_slo="86400000000000" # See Clockwork controller process usage below
generate_inputs="0"          # See Clockwork controller process usage below
max_gpus="6"                 # See Clockwork controller process usage below
schedule_ahead="5000000"     # See Clockwork controller process usage below

################################################################################



SECONDS=0
printf "\nStarting Exp. ${exp_name} with ${model} models\n"
printf "The experiment log directory is ${logdir}\n"

# echo ""
# for worker in "${workers[@]}"
# do
# 	echo "Stop any leftover Clockwork worker processes on host ${worker}"
# 	remote_cmd="pkill -f ${CLOCKWORK_BUILD}/worker"
# 	echo "Remote worker cmd: ${remote_cmd}"
# 	$(${ssh_cmd_prefix} ${worker} "${remote_cmd}")
# done

# echo "Sleeping 30s"
# sleep 30s

for dist in "${dists[@]}"
do

  for rate in "${rates[@]}"
  do

    for num_models in "${num_models_vals[@]}"
    do
      # Experiment configuration string, also used by the plotting script
      config="dist=${dist}_rate=${rate}_num-models=${num_models}"
      printf "\nConfig: ${config}\n\n"

      # Start the Clockwork worker process remotely on each worker host
      # Keep track of all worker process PIDs
      # Sleep for a while, since the controller expects all workers to be ready
      worker_pids=()
      for worker in "${workers[@]}"
      do
        printf "Start Clockwork worker remotely on host ${worker}\n"
        logfile="${logdir}/file=worker-${worker}_${config}.log"
        remote_cmd="${mklogdir} ${env_vars} nohup ${CLOCKWORK_BUILD}/worker"
        remote_cmd+=" > ${logfile} 2>&1 < /dev/null  & echo \$!"
        printf "Remote worker cmd: ${remote_cmd}\n"
        worker_pid=$(${ssh_cmd_prefix} ${worker} "${remote_cmd}")
        worker_pids+=(${worker_pid})
      done
      echo "All worker processes' PIDs ${worker_pids[@]}"
      printf "Sleeping 30s\n"
      sleep 30s

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

      # Start the Clockwork controller process remotely on the specified host
      # Keep track of the controller process PID
      # Sleep for a while, so that the controller is ready to serve requests
      printf "\nStart Clockwork controller remotely on host ${controller}\n"
      logfile="${logdir}/file=controller_${config}.log"
      remote_cmd="${mklogdir} ${env_vars} nohup ${CLOCKWORK_BUILD}/controller"
      remote_cmd+=" INFER5 ${worker_arg_for_controller} ${generate_inputs}"
      remote_cmd+=" ${max_gpus} ${schedule_ahead} ${default_slo}"
      remote_cmd+=" > ${logfile} 2>&1 < /dev/null & echo \$!"
      printf "Remote controller cmd: ${remote_cmd}\n"
      CONTROLLER_PID=$(${ssh_cmd_prefix} ${controller} "${remote_cmd}")
      printf "Controller process's PID ${CONTROLLER_PID}\n"
      printf "Sleeping 30s\n"
      sleep 30s

      #-------------------------------------------------------------------------
      # Clockwork client process usage:
      #   client [address] [workload] [workload parameters (if required)]
      # In this experiment, we use the following workload:
      #   slo-exp-1
      # The workload parameters are:
      #   model copies dist rate slo-start slo-end slo-factor slo-op period
      # Following is the description of workload parameters:
      #   model: model name (e.g., "resnet50_v2")
      #   copies: number of model instances
      #   dist: arrival distribution ("poisson"/"fixed-rate")
      #   rate: arrival rate (in requests/second)
      #   slo-start: starting slo multiplier
      #   slo-end: ending slo multiplier
      #   slo-factor: factor by which the slo multiplier should change
      #   slo-op: operator ("add"/"mul") for incrementing slo
      #   period: number of seconds before changing slo
      #-------------------------------------------------------------------------
      
      # Start the Clockwork client process remotely on the specified host
      # Keep track of the client process PID
      # Since our workload is finitely long, wait for the client to terminate
      printf "\nStarting Clockwork client remotely on host ${client}\n"
      logfile="${logdir}/file=client_${config}.log"
      remote_cmd="${mklogdir} ${env_vars} nohup ${CLOCKWORK_BUILD}/client"
      remote_cmd+=" ${controller}:12346 ${exp_name} ${model} ${num_models}"
      remote_cmd+=" ${dist} ${rate} ${slo_start} ${slo_end} ${slo_factor}"
      remote_cmd+=" ${slo_op} ${period}"
      remote_cmd+=" > ${logfile} 2>&1 < /dev/null"
      printf "Remote client cmd: ${remote_cmd}\n"
      $(${ssh_cmd_prefix} ${client} "${remote_cmd}")

      # The controller telemetry_file is at /local/clockwork_request_log.tsv
      # Copy it to localhost after the experiment
      printf "\nCopying controller's request telemetry file to ${logdir}\n"
      telemetryfile="${logdir}/file=controller_${config}_request.csv"
      default_telemetryfile="${logdir}/clockwork_request_log.tsv"
      $(scp -P ${SSH_PORT} ${username}@${controller}:${default_telemetryfile} ${telemetryfile})

      # Stop the controller process
      printf "\nStop Clockwork controller on host ${controller}\n"
      remote_cmd="kill -2 ${CONTROLLER_PID}"
      $(${ssh_cmd_prefix} ${controller} "${remote_cmd}")

      # Stop all worker processes
      printf "\nStop Clockwork workers on hosts ${workers[@]}\n"
      len1=${#workers[@]}
      len2=${#worker_pids[@]}
      for (( i=0; i<${len1}; i++ ))
      do
        worker=${workers[$i]}
        worker_pid=${worker_pids[$i]}
        remote_cmd="kill -9 ${worker_pid}"
        $(${ssh_cmd_prefix} ${worker} "${remote_cmd}")
      done

      # Sleep for a while before starting the next experiment run
      printf "\nSleeping 1m\n"
      sleep 1m
    done

  done

done

duration=$SECONDS
printf "Exiting\n"
printf "Roughly $(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed.\n"

################################################################################
