#!/bin/bash
DIR=clockwork-modelzoo-volta

if [ ! -d $DIR -o "$1" == "-f" ] ; then
	echo "Grabbing pre-compiled models (about 16GB), this might take a while ..."
	#mkdir -p $DIR
	#cd $DIR
	#-O - |tar -zx
	wget -c https://gitlab.mpi-sws.org/cld/ml/clockwork-modelzoo-volta/-/archive/master/clockwork-modelzoo-volta-master.tar.gz && \ 
	echo "Models downloaded, now decompressing..." && \
	tar -xf clockwork-modelzoo-volta-master.tar.gz &&
	echo "Removing compressed file..." &&
	rm -f clockwork-modelzoo-volta-master.tar.gz
	
	echo "Models downloaded to $DIR"
else 
	echo "Model zoo directory $DIR already exists, use -f to force download."
fi


