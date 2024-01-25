#!/usr/bin/bash

logdir="/home/hzq/results/maf-exp-1-and-2/log/2023-12-17-01-58-46"
config=1
SSH_PORT=22
username="root"
controller="172.16.201.2"

echo ""
echo "Copying controller's request telemetry file to ${logdir}"
request_telemetryfile="${logdir}/file=controller_${config}_request.tsv"
default_telemetryfile="${logdir}/clockwork_request_log.tsv"
$(scp -P ${SSH_PORT} ${username}@${controller}:${default_telemetryfile} ${request_telemetryfile})

echo ""
echo "Copying controller's action telemetry file to ${logdir}"
action_telemetryfile="${logdir}/file=controller_${config}_action.tsv"
default_telemetryfile="${logdir}/clockwork_action_log.tsv"
$(scp -P ${SSH_PORT} ${username}@${controller}:${default_telemetryfile} ${action_telemetryfile})