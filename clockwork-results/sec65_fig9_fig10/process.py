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
import os.path

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Process clockwork_request_log.tsv and clockwork_action_log.tsv')
parser.add_argument('-i', "--inputdir", metavar="INPUTDIR", type=str, default="/local", help="Path to a directory containing experiment output files.  Files should be in format \"file=controller_1_request.tsv\" etc.")
parser.add_argument('-o', "--outputdir", metavar="OUTPUTDIR", type=str, default=".", help="Directory to put the processed output.  Directory will be created if it does not exist.")
parser.add_argument('-l', "--leadin", metavar="LEADIN", type=int, default=600, help="Exclude the first LEADIN seconds of data.  Default 600 (10 minutes).")
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
    df = pd.read_csv(filename, sep="\t", header=0, nrows=100000000)
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

    requests = requests[(requests.t >= t_min) & (requests.t <= t_max)]
    actions = actions[(actions.t >= t_min) & (actions.t <= t_max)]

    bucketsize = intervalsize * 1000000000
    actions["bucket"] = ((actions.t - t_min) / bucketsize).astype("int64")
    actions["batch_size_sum"] = actions.batch_size * actions.batch_size
    requests["bucket"] = ((requests.t - t_min) / bucketsize).astype("int64")
    data = {
    "bucket": requests.bucket.unique()
    }
    data["t"] = data["bucket"] * intervalsize / 60.0  # time in minutes
    data["throughput"] = requests.groupby("bucket").t.count() / intervalsize
    data["goodput"] = requests[(requests.result == 0) & (requests.deadline_met == True)].groupby("bucket").t.count() / intervalsize
    data["batchsize"] = actions[(actions.action_type == 2)].groupby("bucket").batch_size_sum.sum() / actions[(actions.action_type == 2)].groupby("bucket").batch_size.sum()
    data["coldstart"] = requests[(requests.result == 0) & requests.is_coldstart].groupby("bucket").is_coldstart.count() / intervalsize
    data["warmstart"] = requests[(requests.result == 0) & (requests.is_coldstart == False)].groupby("bucket").is_coldstart.count() / intervalsize
    data["coldmodel"] = requests[(requests.result == 0) & requests.is_coldstart].groupby("bucket").model_id.nunique()
    data["models"] = requests.groupby("bucket").model_id.nunique()
    data["latency_median"] = requests.groupby("bucket").latency.quantile(0.5) / 1000000.0
    data["latency_p99"] = requests.groupby("bucket").latency.quantile(0.99) / 1000000.0
    data["latency_max"] = requests.groupby("bucket").latency.max() / 1000000.0
    data["warmmodel"] = requests[(requests.result == 0) & (requests.arrival_count > 0)].groupby("bucket").model_id.nunique()

    datadf = pd.DataFrame(data)
    datadf = datadf.fillna(0)
    return datadf


def plot_infer_prediction(df, outputfile):
    plt.clf()
    ax = plt.gca()

    def format_fn(tick_val, tick_pos):
        percentile = (100 - math.pow(10, 2-tick_val))
        return "%s" % str(percentile)

    ax.yaxis.set_major_formatter(FuncFormatter(format_fn))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    df.plot(kind='line',y='y', x="infer_prediction_error_under", label="Underpredict", ax=ax)
    df.plot(kind='line',y='y', x="infer_prediction_error_over", label="Overpredict", ax=ax)
    plt.title("(10, top-left) Infer Prediction Error")
    plt.ylabel("Percentile")
    plt.xlabel("Error (ms)")
    plt.savefig(outputfile)


def plot_load_prediction(df, outputfile):
    plt.clf()
    ax = plt.gca()

    def format_fn(tick_val, tick_pos):
        percentile = (100 - math.pow(10, 2-tick_val))
        return "%s" % str(percentile)

    ax.yaxis.set_major_formatter(FuncFormatter(format_fn))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    df.plot(kind='line',y='y', x="load_prediction_error_under", label="Underpredict", ax=ax)
    df.plot(kind='line',y='y', x="load_prediction_error_over", label="Overpredict", ax=ax)
    plt.title("(10, top-right) Load Prediction Error")
    plt.ylabel("Percentile")
    plt.xlabel("Error (ms)")
    plt.savefig(outputfile)


def plot_infer_completion(df, outputfile):
    plt.clf()
    ax = plt.gca()

    def format_fn(tick_val, tick_pos):
        percentile = (100 - math.pow(10, 2-tick_val))
        return "%s" % str(percentile)

    ax.yaxis.set_major_formatter(FuncFormatter(format_fn))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    df.plot(kind='line',y='y', x="infer_completion_error_under", label="Underpredict", ax=ax)
    df.plot(kind='line',y='y', x="infer_completion_error_over", label="Overpredict", ax=ax)
    plt.title("(10, bottom-left) Infer Completion Error")
    plt.ylabel("Percentile")
    plt.xlabel("Error (ms)")
    plt.savefig(outputfile)


def plot_load_completion(df, outputfile):
    plt.clf()
    ax = plt.gca()

    def format_fn(tick_val, tick_pos):
        percentile = (100 - math.pow(10, 2-tick_val))
        return "%s" % str(percentile)

    ax.yaxis.set_major_formatter(FuncFormatter(format_fn))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    df.plot(kind='line',y='y', x="load_completion_error_under", label="Underpredict", ax=ax)
    df.plot(kind='line',y='y', x="load_completion_error_over", label="Overpredict", ax=ax)
    plt.title("(10, bottom-right) Load Completion Error")
    plt.ylabel("Percentile")
    plt.xlabel("Error (ms)")
    plt.savefig(outputfile)


def plot_throughput(df, outputfile):
    plt.clf()
    ax = plt.gca()
    df.plot(kind='line',x='t',y='goodput', label="Goodput", ax=ax)
    df.plot(kind='line',x='t',y='throughput',label="Throughput", ax=ax)
    plt.title("(9a) Throughput")
    plt.xlabel("Time (Minutes)")
    plt.ylabel("T'put (r/s)")
    plt.savefig(outputfile)


def plot_latency(df, outputfile):
    plt.clf()
    ax = plt.gca()
    df.plot(kind='line',x='t',y='latency_median',label="Median",ax=ax)
    df.plot(kind='line',x='t',y='latency_p99',label="99th %%ile", ax=ax)
    df.plot(kind='line',x='t',y='latency_max',label="Maximum", ax=ax)
    plt.title("(9b) Latency")
    plt.xlabel("Time (Minutes)")
    plt.ylabel("Latency (ms)")
    plt.savefig(outputfile)


def plot_batchsize(df, outputfile):
    plt.clf()
    ax = plt.gca()
    df.plot(kind='line',x='t',y='batchsize',label="Mean",ax=ax)
    plt.title("(9c) Batch Size")
    plt.xlabel("Time (Minutes)")
    plt.ylabel("Batch Size")
    plt.savefig(outputfile)


def plot_coldmodels(df, outputfile):
    plt.clf()
    ax = plt.gca()
    df.plot(kind='line',x='t',y='coldmodel',label="Total",ax=ax)
    plt.title("(9d) Cold Models")
    plt.xlabel("Time (Minutes)")
    plt.ylabel("Cold Models")
    plt.savefig(outputfile)


def plot_coldstarts(df, outputfile):
    plt.clf()
    ax = plt.gca()
    df.plot(kind='line',x='t',y='coldstart',label="Coldstarts",ax=ax)
    plt.title("(9e) Cold Starts")
    plt.xlabel("Time (Minutes)")
    plt.ylabel("T'put (r's)")
    plt.savefig(outputfile)

def process(args):
    inputdir = args.inputdir
    outputdir = args.outputdir

    for config, requestfile, actionsfile in get_logfiles(inputdir):
        print(config, requestfile, actionsfile)
        requests = load_log(requestfile, args.leadin, args.duration)
        actions = load_log(actionsfile, args.leadin, args.duration)
        taildata = make_tail_cdf(actions, requests)
        timeseriesdata = make_timeseries(requests, actions, args.bucketsize)
        taildata.to_csv("%s/%s.tsv" % (outputdir, "client_tail_cdf_data"), sep="\t", index=False)
        timeseriesdata.to_csv("%s/%s.tsv" % (outputdir, "timeseries_data"), sep="\t", index=False)

        plot_infer_prediction(taildata, "%s/%s.pdf" % (args.outputdir, "10_tl_infer_prediction_error"))
        plot_load_prediction(taildata, "%s/%s.pdf" % (args.outputdir, "10_tr_load_prediction_error"))
        plot_infer_completion(taildata, "%s/%s.pdf" % (args.outputdir, "10_bl_infer_completion_error"))
        plot_load_completion(taildata, "%s/%s.pdf" % (args.outputdir, "10_br_load_completion_error"))
        plot_throughput(timeseriesdata, "%s/%s.pdf" % (args.outputdir, "9a_throughput"))
        plot_latency(timeseriesdata, "%s/%s.pdf" % (args.outputdir, "9b_timeseries_latency"))
        plot_batchsize(timeseriesdata, "%s/%s.pdf" % (args.outputdir, "9c_timeseries_batchsize"))
        plot_coldmodels(timeseriesdata, "%s/%s.pdf" % (args.outputdir, "9d_timeseries_coldmodels"))
        plot_coldstarts(timeseriesdata, "%s/%s.pdf" % (args.outputdir, "9e_timeseries_coldstarts"))


if __name__ == '__main__':
    args = parser.parse_args()
    exit(process(args))
