import json
import subprocess
import argparse
import time
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import pip as sns
import os.path

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Process clockwork_request_log.tsv and clockwork_action_log.tsv')
parser.add_argument('-i', "--inputdir", metavar="INPUTDIR", type=str, default="/local", help="Path to a directory containing experiment output files.  Files should be in format \"file=controller_1_request.tsv\" etc.")
parser.add_argument('-o', "--outputdir", metavar="OUTPUTDIR", type=str, default=".", help="Directory to put the processed output.  Directory will be created if it does not exist.")
parser.add_argument('-l', "--leadin", metavar="LEADIN", type=int, default=0, help="Exclude the first LEADIN seconds of data.  Default 600 (10 minutes).")
parser.add_argument('-d', "--duration", metavar="DURATION", type=int, default=-1, help="Include the first DURATION seconds of data.  Set to -1 to include all data.  Default -1.")
parser.add_argument('-b', "--bucketsize", metavar="BUCKETSIZE", type=int, default=60, help="Interval size for time series data.  Default 60 (1 minute).")

def get_logfiles(logdir):
    logfiles = []
    config = 0
    while True:
        config += 1
        requestfile = logdir + "/" + "file=controller_" + str(config) + "_request.tsv"
        actionsfile = logdir + "/" + "file=controller_" + str(config) + "_action.tsv"
        if (os.path.isfile(actionsfile) and os.path.isfile(requestfile)):
            logfiles.append([config, requestfile, actionsfile])
        else:
            break
    return logfiles

def load_log(filename, leadin, duration):
    df = pd.read_csv(filename, sep="\t", header=0)
    df = df.dropna()


    leadin_ns = leadin * 1000000000
    duration_ns = duration * 1000000000

    t_min = df.t.min() + leadin_ns
    t_max = df.t.max()
    if duration != -1:
        t_max = min(df.t.max(), t_min+duration_ns)

    df = df[(df.t >= t_min) & (df.t <= t_max)]

    print("Results: %f to %f" % (t_min / 1000000000, t_max / 1000000000))

    return df


def make_regular_cdf(actions, requests):
    num_points = 1000
    quantiles = np.linspace(0, 1, num=num_points)

    return make_cdf_data(actions, requests, quantiles)


def make_tail_cdf(actions, requests):
    num_points = 1000
    num_elements = max(len(actions), len(requests))

    quantiles = 1 - 1 / np.logspace(0, np.log10(num_elements), num=num_points)
    ys = np.linspace(0, np.log10(num_elements), num=num_points)

    data = make_cdf_data(actions, requests, quantiles)
    data["y"] = ys

    return data


def make_cdf_data(actions, requests, quantiles):

    actions["prediction_error"] = (actions.worker_exec_duration - actions.expected_exec_duration) / 1000000.0
    actions["prediction_error_ratio"] = actions.prediction_error / actions.worker_exec_duration
    actions["completion_error"] = (actions.worker_exec_complete - actions.expected_exec_complete) / 1000000.0
    actions["completion_error_ratio"] = actions.completion_error / actions.worker_exec_complete

    data = {
    "quantile": quantiles
    }

    metrics = [
        "prediction_error",
        "prediction_error_ratio",
        "completion_error",
        "completion_error_ratio",
    ]
    actiontypes = {
        "load": 1,
        "infer": 2
    }

    for metric in metrics:
        for action, actiontype in actiontypes.items():
            filtered = actions[actions.action_type == actiontype]

            outputname = "%s_%s_over" % (action, metric)
            data[outputname] = filtered[filtered[metric] >= 0][metric].quantile(quantiles)

            outputname = "%s_%s_under" % (action, metric)
            data[outputname] = (filtered[filtered[metric] <=0][metric] * -1).quantile(quantiles)

    data["latency"] = requests.latency.quantile(quantiles) / 1000000.0
    data["cold"] = requests[requests.arrival_count == 0].latency.quantile(quantiles) / 1000000.0
    data["hot"] = requests[requests.arrival_count > 0].latency.quantile(quantiles) / 1000000.0


    return pd.DataFrame(data)


def make_timeseries(requests, actions, intervalsize):
    t_min = max(requests.t.min(), actions.t.min())
    t_max = min(requests.t.max(), actions.t.max())
    bucketsize = intervalsize * 1000000000
    actions["bucket"] = ((actions.t - t_min) / bucketsize).astype("int64")
    actions["batch_size_sum"] = actions.batch_size * actions.batch_size
    requests["bucket"] = ((requests.t - t_min) / bucketsize).astype("int64")
    data = {
    "bucket": requests.bucket.unique()
    }
    data["t"] = data["bucket"] * intervalsize / 60.0  # time in minutes
    data["throughput"] = requests.groupby("bucket").t.count() / intervalsize
    data["request_rate"] = requests.groupby("bucket").t.count() / intervalsize
    data["goodput"] = requests[(requests.result == 0) & (requests.deadline_met == True)].groupby("bucket").t.count() / intervalsize
    data["badput"] = data["throughput"] - data["goodput"]
    data["badput"] = requests[(requests.result == 0)].groupby("bucket").t.count() / intervalsize
    data["batchsize"] = actions[(actions.action_type == 2)].groupby("bucket").batch_size_sum.sum() / actions[(actions.action_type == 2)].groupby("bucket").batch_size.sum()
    data["coldstart"] = requests[(requests.result == 0) & requests.is_coldstart].groupby("bucket").is_coldstart.count() / intervalsize
    data["warmstart"] = requests[(requests.result == 0) & (requests.is_coldstart == False)].groupby("bucket").is_coldstart.count() / intervalsize
    data["coldmodel"] = requests[(requests.result == 0) & requests.is_coldstart].groupby("bucket").model_id.nunique()
    data["models"] = requests.groupby("bucket").model_id.nunique()
    data["latency_median"] = requests.groupby("bucket").latency.quantile(0.5) / 1000000.0
    data["latency_p99"] = requests.groupby("bucket").latency.quantile(0.99) / 1000000.0
    data["latency_max"] = requests.groupby("bucket").latency.max() / 1000000.0
    data["latency_p50_good"] = requests[(requests.result == 0) & (requests.deadline_met == True)].groupby("bucket").latency.quantile(0.5) / 1000000.0
    data["latency_p99_good"] = requests[(requests.result == 0) & (requests.deadline_met == True)].groupby("bucket").latency.quantile(0.99) / 1000000.0
    data["latency_max_good"] = requests[(requests.result == 0) & (requests.deadline_met == True)].groupby("bucket").latency.max() / 1000000.0
    data["warmmodel"] = requests[(requests.result == 0) & (requests.arrival_count > 0)].groupby("bucket").model_id.nunique()

    datadf = pd.DataFrame(data)
    datadf = datadf.fillna(0)
    datadf.drop(datadf.tail(1).index, inplace=True) # drop last row, since throughput falls to zero

    datadf["goodput_p99"] = datadf["goodput"] * (1 / 99.0)
    datadf["goodput_p99"] = datadf[['goodput_p99','badput']].min(axis=1)
    datadf["goodput_p99"] += datadf["goodput"]

    datadf["goodput_p95"] = datadf["goodput"] * (5 / 95.0)
    datadf["goodput_p95"] = datadf[['goodput_p95','badput']].min(axis=1)
    datadf["goodput_p95"] += datadf["goodput"]

    datadf["goodput_p90"] = datadf["goodput"] * (10 / 90.0)
    datadf["goodput_p90"] = datadf[['goodput_p90','badput']].min(axis=1)
    datadf["goodput_p90"] += datadf["goodput"]

    #datadf["goodput_p90"] = datadf["goodput"] + min(datadf["badput"], datadf["goodput"] * 10 / 90.0)
    #datadf["goodput_p95"] = datadf["goodput"] + min(datadf["badput"], datadf["goodput"] * 5 / 95.0)

    return datadf


def plot_throughput(df, outputfile):
    plt.clf()
    ax = plt.gca()
    ax.set_yscale('log')
    df.plot(kind='line',x='t',y='goodput', label="Goodput (100% SLO)", ax=ax)
    df.plot(kind='line',x='t',y='goodput_p99', label="Goodput (99% SLO)", ax=ax)
    df.plot(kind='line',x='t',y='goodput_p95', label="Goodput (95% SLO)", ax=ax)
    df.plot(kind='line',x='t',y='goodput_p90', label="Goodput (90% SLO)", ax=ax)
    df.plot(kind='line',x='t',y='throughput',label="Throughput", ax=ax)
    plt.title("(11a) Throughput")
    plt.xlabel("Time (Minutes)")
    plt.ylabel("Request Rate (r/s)")
    plt.savefig(outputfile)


def plot_latency(df, outputfile):
    plt.clf()
    ax = plt.gca()
    df.plot(kind='line',x='t',y='latency_median',label="Median",ax=ax)
    df.plot(kind='line',x='t',y='latency_p99',label="99th %ile", ax=ax)
    df.plot(kind='line',x='t',y='latency_max',label="Maximum", ax=ax)
    plt.title("(11b) Latency")
    plt.xlabel("Time (Minutes)")
    plt.ylabel("Latency (ms)")
    plt.savefig(outputfile)


def plot_batchsize(df, outputfile):
    plt.clf()
    ax = plt.gca()
    df.plot(kind='line',x='t',y='batchsize',label="Mean",ax=ax)
    plt.title("(11c) Batch Size")
    plt.xlabel("Time (Minutes)")
    plt.ylabel("Batch Size")
    plt.savefig(outputfile)


def plot_throughput_latency(df, outputfile):
    plt.clf()
    ax = plt.gca()
    ax.set_xscale('log')
    df.plot(kind='line',x='goodput',y='latency_p50_good',label="P50 Latency",ax=ax)
    df.plot(kind='line',x='goodput',y='latency_p99_good',label="P99 Latency",ax=ax)
    df.plot(kind='line',x='goodput',y='latency_max_good',label="Max Latency",ax=ax)
    plt.title("(10a) Goodput vs. Latency")
    plt.xlabel("Goodput (r/s)")
    plt.ylabel("Latency (ms)")
    plt.savefig(outputfile)


def process(args):
    inputdir = args.inputdir
    outputdir = args.outputdir

    goodput = []
    goodput_p99 = []
    goodput_p95 = []
    goodput_p90 = []
    throughput = []

    for config, requestfile, actionsfile in get_logfiles(inputdir):
        print(config, requestfile, actionsfile)
        requests = load_log(requestfile, args.leadin, args.duration)
        actions = load_log(actionsfile, args.leadin, args.duration)
        taildata = make_tail_cdf(actions, requests)
        timeseriesdata = make_timeseries(requests, actions, args.bucketsize)
        taildata.to_csv("%s/%s-%s.tsv" % (outputdir, "client_tail_cdf_data", config), sep="\t", index=False)
        timeseriesdata.to_csv("%s/%s-%s.tsv" % (outputdir, "timeseries_data", config), sep="\t", index=False)

        print(config, "Max goodput (100% SLO)", timeseriesdata['goodput'].max())
        print(config, "Max goodput (099% SLO)", timeseriesdata['goodput_p99'].max())
        print(config, "Max goodput (095% SLO)", timeseriesdata['goodput_p95'].max())
        print(config, "Max goodput (090% SLO)", timeseriesdata['goodput_p90'].max())
        print(config, "Max throughput", timeseriesdata['throughput'].max())
        goodput.append(timeseriesdata['goodput'].max())
        goodput_p99.append(timeseriesdata['goodput_p99'].max())
        goodput_p95.append(timeseriesdata['goodput_p95'].max())
        goodput_p90.append(timeseriesdata['goodput_p90'].max())
        throughput.append(timeseriesdata['throughput'].max())

        plot_throughput(timeseriesdata, "%s/%s-%s.pdf" % (args.outputdir, "11a_throughput", config))
        plot_latency(timeseriesdata, "%s/%s-%s.pdf" % (args.outputdir, "11b_timeseries_latency", config))
        plot_batchsize(timeseriesdata, "%s/%s-%s.pdf" % (args.outputdir, "11c_timeseries_batchsize", config))

    data = {'num_models': range(10, 151, 10)}
    data['goodput'] = goodput
    data['goodput_p99'] = goodput_p99
    data['goodput_p95'] = goodput_p95
    data['goodput_p90'] = goodput_p90
    data['throughput'] = throughput
    df = pd.DataFrame(data)
    df.to_csv("%s/%s.tsv" % (outputdir, "goodput_aggregate_data"), sep="\t", index=False)
    print(df)

    plt.clf()
    ax = plt.gca()
    df.plot(kind='line',x='num_models',y='goodput', label="Goodput (100% SLO)", ax=ax)
    df.plot(kind='line',x='num_models',y='goodput_p99', label="Goodput (99% SLO)", ax=ax)
    df.plot(kind='line',x='num_models',y='goodput_p95', label="Goodput (95% SLO)", ax=ax)
    df.plot(kind='line',x='num_models',y='goodput_p90', label="Goodput (90% SLO)", ax=ax)
    df.plot(kind='line',x='num_models',y='throughput', label="Throughput", ax=ax)
    plt.xlabel("Number of GPUs")
    plt.ylabel("Rate (r/s)")
    plt.savefig("%s/%s.pdf" % (args.outputdir, "10b_goodput_aggregate"))


if __name__ == '__main__':
    args = parser.parse_args()
    exit(process(args))
