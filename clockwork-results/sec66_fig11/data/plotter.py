import json
import subprocess
import argparse
import time
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter, MaxNLocator
import seaborn as sns
import os.path
import os
from glob import glob

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Process clockwork_request_log.tsv and clockwork_action_log.tsv')
parser.add_argument('-i', "--inputdir", metavar="INPUTDIR", type=str, default=".", help="Path to a directory containing experiment output files.")
parser.add_argument('-o', "--outputdir", metavar="OUTPUTDIR", type=str, default=".", help="Directory to put the processed output.  Directory will be created if it does not exist.")

def plot_goodput_per_config(inputdir, outputdir, config):
  tsvfiles = [y for x in os.walk(inputdir) for y in glob(os.path.join(x[0], 'timeseries_data-' + str(config) + '.tsv'))]

  data = {}
  idx = 0
  for tsvfile in tsvfiles:
    df = pd.read_csv(tsvfile, sep="\t", header=0)
    if data == {}:
      data['bucket'] = df['bucket']
      data['t'] = df['t']
    data['goodput' + str(idx)] = df['goodput']
    data['batchsize' + str(idx)] = df['batchsize']
    data['throughput' + str(idx)] = df['throughput']
    idx += 1
  df = pd.DataFrame(data)
  df['goodput_median'] = (df.filter(regex=r'^goodput', axis=1)).median(axis=1)
  df['batchsize_median'] = (df.filter(regex=r'^batchsize', axis=1)).median(axis=1)
  df['throughput_median'] = (df.filter(regex=r'^throughput', axis=1)).median(axis=1)
  
  plt.clf()
  plt.xticks(range(0, 9), fontsize=13)
  plt.yticks(range(0, 120001, 15000), fontsize=13)

  ax = plt.gca()
  ax.set_xlim(0, 8)
  ax.set_ylim(0, 120000)
  ax.set_ylabel("Request Rate (r/s)", fontsize=15)
  ax = df.plot(kind='line',x='t',y='goodput_median', label="Goodput", ax=ax, figsize=(4, 3.5), color='grey')
  ax = df.plot(kind='line',x='t',y='throughput_median', label="Offered Load", ax=ax, figsize=(4, 3.5), color='black')
  ax.set_xlabel("Time (minutes)", fontsize=15)
  ax.legend(loc="upper left", fontsize=11, handlelength=1)
  plt.grid()

  ax2 = ax.twinx()
  ax2.spines['right'].set_position(('axes', 1.0))
  ax2.set_ylim(0, 8)
  ax2.set_ylabel("Batch Size", fontsize=15)
  df.plot(kind='line',x='t',y='batchsize_median', label="Mean Batch Size", ax=ax2, figsize=(4, 3.5), color='orange', mark_right=True)
  
  #plt.xlabel("Time (minutes)")
  ax2.set_xlabel("Time (minutes)", fontsize=15)
  ax2.legend(loc="center left", fontsize=11, handlelength=1)

  plt.xticks(fontsize=13)
  plt.yticks(fontsize=13)

  plt.savefig("%s/%s.pdf" % (outputdir, "11a_goodput_median_" + str(10 * config) + "_workers"), bbox_inches='tight', transparent=False)
  
  tsvfile = "%s/%s.tsv" % (outputdir, "11a_goodput_median_" + str(10 * config) + "_workers")
  print("Generating " + tsvfile)
  df.to_csv(tsvfile, sep="\t", index=False)

def plot_aggregate_goodput(inputdir, outputdir):
  tsvfiles = [y for x in os.walk(inputdir) for y in glob(os.path.join(x[0], 'goodput_aggregate_data.tsv'))]
  
  data = {}
  idx = 0
  for tsvfile in tsvfiles:
    df = pd.read_csv(tsvfile, sep="\t", header=0)
    if data == {}: data['num_models'] = df['num_models']
    data['goodput' + str(idx)] = df['goodput']
    idx += 1
  df = pd.DataFrame()
  df = df.append({'num_models':0, 'goodput0':0, 'goodput1':0, 'goodput2':0, 'goodput_median':0}, ignore_index=True)
  df = df.append(pd.DataFrame(data))
  df['goodput_median'] = (df.filter(regex=r'^goodput', axis=1)).median(axis=1)

  plt.clf()
  ax = plt.gca()
  df.plot(kind='line',x='num_models',y='goodput_median', ax=ax, label="Goodput", marker='o', figsize=(4, 3.5), color='grey')
  ax.set_ylim(0, 120000)
  ax.set_xlim(0, 150)
  ax.legend(loc="lower right", fontsize=13)
  plt.xlabel("Number of Workers", fontsize=15)
  plt.ylabel("Request Rate (r/s)", fontsize=15)
  plt.xticks(range(10, 160, 20), fontsize=13, rotation=45)
  plt.yticks(range(0, 120001, 15000), fontsize=13)
  plt.grid()
  plt.savefig("%s/%s.pdf" % (outputdir, "11b_goodput_median_aggregate"), bbox_inches='tight', transparent=False)

  tsvfile = "%s/%s.tsv" % (outputdir, "11b_goodput_median_aggregate")
  print("Generating " + tsvfile)
  df.to_csv(tsvfile, sep="\t", index=False)

def process(args):
  inputdir = args.inputdir
  outputdir = args.outputdir

  plot_goodput_per_config(inputdir, outputdir, 4)
  plot_aggregate_goodput(inputdir, outputdir)

if __name__ == '__main__':
    args = parser.parse_args()
    exit(process(args))
