#!/bin/bash
TAG=registry:5000/clockwork:artifact
LTAG=clockwork:local
BASEDIR=~/docker-clockwork 

if [ "$(sudo docker images -q |grep ca043986189b)" = "" -o "$1" == "-f" ] ; then

	echo "Retrieving Clockwork container.."
	sudo docker pull ${TAG} || exit 1

	echo "Download complete"

	echo "Adding SSH keys to the environment ..."
	sudo docker build -f $BASEDIR/setup/Dockerfile -t ${LTAG} $BASEDIR || exit 1

else 
	echo "Clockwork image already exists."
fi

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
mkdir -p $BASEDIR/logs
mkdir -p $BASEDIR/azure-functions
mkdir -p $BASEDIR/clockwork-modelzoo-volta
MOUNTS="-v $BASEDIR/azure-functions:/azure-functions -v $BASEDIR/clockwork-modelzoo-volta:/clockwork-modelzoo-volta -v $BASEDIR/logs:/logs -v /dev/shm:/dev/shm"
nvidia-smi 2>&1>/dev/null && HASGPU="--gpus all" || HASGPU=""
sudo docker run $HASGPU $MOUNTS --ulimit memlock=-1:-1 --ulimit rtprio=-1:-1 -d $LTAG && \
echo "Success. You can now be able to enter the environment through 'ssh clockwork@container_ip' if your key was in .ssh/authorized_keys"

