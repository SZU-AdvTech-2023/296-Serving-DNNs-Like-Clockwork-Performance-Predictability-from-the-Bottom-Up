import json
import subprocess
import argparse
import time
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='Process a request log file')
parser.add_argument('-i', "--inputfile", metavar="INPUTFILE", type=str, default="/local/clockwork_request_log.tsv", help="Path to a clockwork_request_log.tsv file.  Uses /local/clockwork_request_log.tsv by default.")
parser.add_argument('-o', "--outputdir", metavar="OUTPUTDIR", type=str, default=".", help="Directory to put the processed output.  Directory will be created if it does not exist.")

def load_request_log(filename):    
    df = pd.read_csv(filename, sep="\t", header=0)
    df = df.dropna()     
    df["minor"] = df.model_id == 3600 # Model ID 3600 is the "Minor" workload
    df["success"] = (df.result == 0) & (df.deadline_met == 1)
    begin = df[df.minor == False].t.min() # t=0 is when the major workload begins
    df.t = df.t - begin
    df["t_sec"] = (df.t / 1000000000).apply(np.floor).astype("int64")
    df["t_10sec"] = (df.t_sec / 10.0).apply(np.floor).astype("int64") / 6
    df = df[df.t_sec <= 3600] # Experiment lasts 60 mins after begin
    return df


def make_timeseries(df):
    timeseries = df.groupby("t_10sec").count()[[]]
    timeseries["throughput"] = df.groupby(["t_10sec"]).result.count() / 10
    timeseries["goodput"] = df[df.result == 0].groupby(["t_10sec"]).result.count() / 10
    timeseries["goodput-major"] = df[(df.success == True) & (df.minor == False)].groupby("t_10sec").result.count() / 10
    timeseries["goodput-minor"] = df[(df.success == True) & (df.minor == True)].groupby("t_10sec").result.count() / 10
    timeseries["latency-major"] = df[(df.success == True) & (df.minor == False)].groupby("t_10sec").latency.quantile(0.5) / 1000000.0
    timeseries["latency-minor"] = df[(df.success == True) & (df.minor == True)].groupby("t_10sec").latency.quantile(0.5) / 1000000.0
    timeseries["latency-max"] = df.groupby("t_10sec").latency.max() / 1000000.0
    timeseries["coldstarts-major"] = df[(df.success == True) & (df.minor == False)].groupby("t_10sec").is_coldstart.mean() * 100
    timeseries["coldstarts-minor"] = df[(df.success == True) & (df.minor == True)].groupby("t_10sec").is_coldstart.mean() * 100
    return timeseries.fillna(0)

def plot_latency(ts, outputfile):
    plt.clf()
    ax = plt.gca()
    ts.plot(kind='line',y='latency-minor',ax=ax, label="Minor")
    ts.plot(kind='line',y='latency-major',ax=ax, label="Major")
    ts.plot(kind='line',y='latency-max',ax=ax, label="Max")
    plt.title("(6b) Request Latency")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Latency (ms)")
    plt.savefig(outputfile)

def plot_throughput(ts, outputfile):
    plt.clf()
    ax = plt.gca()
    ts.plot(kind='line',y='goodput-minor',ax=ax, label="Minor")
    ts.plot(kind='line',y='goodput-major',ax=ax, label="Major")
    plt.title("(6a) Request Goodput")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Goodput (r/s)")
    plt.savefig(outputfile)

def plot_coldstarts(ts, outputfile):
    plt.clf()
    ax = plt.gca()
    ts.plot(kind='line',y='coldstarts-minor',ax=ax, label="Minor")
    ts.plot(kind='line',y='coldstarts-major',ax=ax, label="Major")
    plt.title("(6c) Cold-Start")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Cold-Start (%)")
    plt.savefig(outputfile)

def process(args):

    inputfile = args.inputfile
    outputdir = args.outputdir

    print("Loading %s" % inputfile)
    df = load_request_log(inputfile)
    print("Loaded %d rows" % (len(df)))

    print("Plotting")
    timeseries = make_timeseries(df)
    timeseries.to_csv("%s/%s.tsv" % (outputdir, "request_timeseries_data"), sep="\t", index=False)
    plot_throughput(timeseries, "%s/%s.pdf" % (outputdir, "6a-goodput"))
    plot_latency(timeseries, "%s/%s.pdf" % (outputdir, "6b-latency"))
    plot_coldstarts(timeseries, "%s/%s.pdf" % (outputdir, "6c-coldstarts"))


if __name__ == '__main__':
    args = parser.parse_args()
    exit(process(args))