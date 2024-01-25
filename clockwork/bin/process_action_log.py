
import math
import subprocess
import itertools
import os
import json
import time
import argparse
import numpy as np
from scipy import signal
import random
from collections import Counter

parser = argparse.ArgumentParser(description='Process an actions log')
parser.add_argument("inputfile", metavar="INPUTFILE", type=str, help="Name of an actions log file")
# parser.add_argument("outputfile", metavar="OUTPUTFILE", type=str, help="Outputfile for summary")

class ModelInstance:
	def __init__(self, worker_id, gpu_id, model_id, batch_size):
		self.worker_id = worker_id
		self.gpu_id = gpu_id
		self.model_id = model_id
		self.batch_size = batch_size
		self.controller_durations = []
		self.worker_durations = []
		self.samples = []

	def add_sample(self, t, controller_action_duration, worker_exec_duration):
		self.controller_durations.append(controller_action_duration)
		self.worker_durations.append(worker_exec_duration)
		self.samples.append((t, controller_action_duration, worker_exec_duration))

	def summary(self):
		return "%d\t%d\t%d\t%d\t%d\t%d" % (
			self.worker_id, 
			self.gpu_id, 
			self.model_id, 
			self.batch_size, 
			len(self.controller_durations), 
			len(self.worker_durations))

def process(lines):
	models = {}
	for line in lines:
		try:
			splits = [int(v) for v in line.split("\t")]
		except:
			continue

		t = splits[0]
		action_id = splits[1]
		action_type = splits[2]
		status = splits[3]
		worker_id = splits[4]
		gpu_id = splits[5]
		model_id = splits[6]
		batch_size = splits[7]
		controller_action_duration = splits[8]
		worker_exec_duration = splits[9]

		key = (worker_id, gpu_id, model_id, batch_size)

		if key not in models:
			models[key] = ModelInstance(worker_id, gpu_id, model_id, batch_size)
		models[key].add_sample(t, controller_action_duration, worker_exec_duration)

	return models;

def printsummary(models):
	for key, model in models.items():
		print(model.summary())



def simplesummary(args):
	with open(args.inputfile, "r") as f:
		lines = f.readlines()

	headers = lines[0].split("\t")
	models = process(lines[1:])
	printsummary(models)


if __name__ == '__main__':
    args = parser.parse_args()
    exit(simplesummary(args))
