#!/bin/bash
TAG=clockworkosdi2020ae/clockwork:latest
LTAG=clockwork:local
BASEDIR=~/ # To allow .ssh folder to be included

# if [ "$(sudo docker images -q |grep fc0708f5d855)" = "" -o "$1" == "-f" ] ; then

# 	echo "Provide login details to Docker Hub to access private repo"
# 	sudo docker login || exit 1

# 	echo "Retrieving Clockwork container... Please use the Docker Hub credentials provided"
# 	sudo docker pull ${TAG} || exit 1

# 	echo "Download complete"

# 	echo "Adding SSH keys to the environment ..."
# 	sudo docker build -f $BASEDIR/clockwork-results/setup/Dockerfile -t ${LTAG} $BASEDIR || exit 1

# else 
# 	echo "Clockwork image already exists."
# fi

# if [ "$(sudo docker ps |grep 2200)" != "" ]; then
# 	sudo docker ps
# 	echo "Clockwork is already running. Restart the current process? [y/n]"
# 	read -N 1 ANSWER
# 	if [ "$ANSWER" = "y" -o "$ANSWER" = "Y" ]; then
# 		echo "Terminating process ..."
# 		sudo docker kill $(sudo docker ps |grep 2200 |gawk '{print $1}')  || exit 1
# 	else
# 		echo "Aborting."
# 		exit 0
# 	fi
# fi

echo "Booting Clockwork environment within Docker... (mounting Azure trace and precompiled models if they have been downloaded)"
# mkdir -p $BASEDIR/logs
# mkdir -p $BASEDIR/azure-functions
# mkdir -p $BASEDIR/clockwork-modelzoo-volta
MOUNTS="-v $BASEDIR/azure-functions:/azure-functions -v $BASEDIR/clockwork-modelzoo-volta:/clockwork-modelzoo-volta -v $BASEDIR/logs:/logs -v /dev/shm:/dev/shm"
nvidia-smi 2>&1>/dev/null && HASGPU="--gpus device=2" || HASGPU=""
sudo docker run $HASGPU $MOUNTS --cpuset-cpus 29-40 -m 70GB --name worker03 --ulimit memlock=-1:-1 --ulimit rtprio=-1:-1 -d $LTAG && \
echo "Success. You can now be able to enter the environment through 'ssh -p 2200 clockwork@localhost' if your key was in .ssh/authorized_keys"

# client
# ip 172.16.201.5
# ssh

# controller
# ip 172.16.201.4
# ssh

# worker01
# ip 172.16.201.2
# ssh 

# worker02
# ip 172.16.201.3
# 