#!/usr/bin/bash

################################################################################
# Regarding experiments in Section 6.1 How Does Clockwork Compare? (comparison_experiment2)
################################################################################

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )" # Absolute path to this script 
EXP_DIR="${SCRIPTPATH}"                         # Experiment directory

exp_name="comparison_experiment2"                      # Codename
timestamp=`date "+%Y-%m-%d-%H-%M-%S"`                  # Start timestamp
logdir="/local/clockwork/${exp_name}/log/${timestamp}" # Log dir

if [ $# -gt 0 ]; then logdir=${1}; fi # Log dir may be specified as an argument
mkdir -p ${logdir}                    # Create log dir locally

################################################################################

./run.sh ${logdir} > ${logdir}/run.log 2>&1 & # Invoke run script

################################################################################
