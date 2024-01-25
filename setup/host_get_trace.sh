#!/bin/bash
DIR=azure-functions

if [ ! -d $DIR -o "$1" == "-f" ] ; then
	echo "Grabbing Azure functions trace ..."
	mkdir -p $DIR
	cd $DIR

	git clone --depth=1 --recursive --single-branch --branch master https://gitlab.mpi-sws.org/cld-private/datasets/azure-functions.git

	echo "Azure functions trace downloaded to $DIR"
else 
	echo "Trace directory $DIR already exists, use -f to force download."
fi


