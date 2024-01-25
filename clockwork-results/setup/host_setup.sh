#!/bin/bash

# host prerequisites
sudo apt update -yqq && sudo apt upgrade -yqq

# === if need to update NVIDIA driver to version 450.51.06
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get -y install nvidia-driver-450
#sudo apt-get -y install cuda

# install docker, nvidia-docker, nvidia-container-toolkit
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common 
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install -y docker-ce

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Recommended modifications

sudo nvidia-smi -pm 1
sudo nvidia-smi --auto-boost-default=DISABLED
sudo sysctl -w vm.max_map_count=10000000
# Dedicate 8GB of memory to loading of models. Note: Needs to be extended for some experiments (up to 768GB)
sudo mount /dev/shm -o remount,rw,exec,size=8G
if [ -e /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor ]; then
	echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
fi
