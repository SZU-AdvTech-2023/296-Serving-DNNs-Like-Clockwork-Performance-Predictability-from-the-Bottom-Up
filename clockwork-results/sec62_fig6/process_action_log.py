import json
import subprocess
import argparse
import time
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='Process an action log file')
parser.add_argument('-i', "--inputfile", metavar="INPUTFILE", type=str, default="/local/clockwork_action_log.tsv", help="Path to a clockwork_action_log.tsv file.  Uses /local/clockwork_action_log.tsv by default.")
parser.add_argument('-o', "--outputdir", metavar="OUTPUTDIR", type=str, default=".", help="Directory to put the processed output.  Directory will be created if it does not exist.")


def percentiles(df, series):
    ts = df.count()[[]]
    ts["max"] = series.max()
    ts["p99"] = series.quantile(0.99)
    ts["p90"] = series.quantile(0.9)
    ts["p50"] = series.quantile(0.5)
    ts["p10"] = series.quantile(0.1)
    ts["p01"] = series.quantile(0.01)
    ts["min"] = series.min()
    return ts

def plot_percentiles(series, ax):
    series.plot(kind='line',y='max',ax=ax)
    series.plot(kind='line',y='p99',ax=ax)
    series.plot(kind='line',y='p90',ax=ax)
    series.plot(kind='line',y='p50',ax=ax)
    series.plot(kind='line',y='p10',ax=ax)
    series.plot(kind='line',y='p01',ax=ax)
    series.plot(kind='line',y='min',ax=ax)

def plot_gpu_utilization_and_goodput_timeseries(args, df):
    # Only plot infer actions
    df = df[df.action_type == 2]

    gpu_count = len(df.groupby(["gpu_id", "worker_id"]))
    df["utilization_norm"] = 100 * df.worker_exec_duration / df.bucketsize
    df["goodput_norm"] = 100 * df.goodput / df.bucketsize

    grouped = df.groupby("bucket")
    ts = grouped.count()[[]]

    ts["utilization"] = grouped.utilization_norm.sum() / gpu_count
    ts["goodput"] = grouped.goodput_norm.sum() / gpu_count

    plt.clf()
    ax = plt.gca()
    ts.plot(kind='line',y='utilization',ax=ax, label="GPU utilization")
    ts.plot(kind='line',y='goodput', color='red', ax=ax, label="GPU utilization goodput")
    plt.title("GPU utilization and goodput")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Utilization (%)")
    plt.savefig("%s/%s.pdf" % (args.outputdir, "gpu_utilization"))

def plot_pci_utilization_and_goodput_timeseries(args, df):
    # Only plot infer actions
    df = df[df.action_type == 1]

    gpu_count = len(df.groupby(["gpu_id", "worker_id"]))
    df["utilization_norm"] = 100 * df.worker_exec_duration / df.bucketsize
    df["goodput_norm"] = 100 * df.goodput / df.bucketsize

    grouped = df.groupby("bucket")
    ts = grouped.count()[[]]

    ts["utilization"] = grouped.utilization_norm.sum() / gpu_count
    ts["goodput"] = grouped.goodput_norm.sum() / gpu_count

    plt.clf()
    ax = plt.gca()
    ts.plot(kind='line',y='utilization',ax=ax, label="PCI utilization")
    ts.plot(kind='line',y='goodput', color='red', ax=ax, label="PCI utilization goodput")
    plt.title("PCI utilization and goodput")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Utilization (%)")
    plt.savefig("%s/%s.pdf" % (args.outputdir, "pci_utilization"))


def plot_action_duration_timeseries(args, df):
    # Only plot successful infer actions
    df = df[df.status == 0]
    df = df[df.action_type == 2]

    df["worker_duration"] = (df.worker_result_sent - df.worker_action_received - df.worker_exec_duration) / 1000000.0
    df["controller_duration"] = (df.controller_action_duration - df.worker_exec_duration) / 1000000.0

    df["worker_pre_output"] = (df.worker_copy_output_complete - df.worker_exec_complete) / 1000000.0
    df["worker_pre_completion"] = (df.worker_result_sent - df.worker_copy_output_complete) / 1000000.0
    df["pre_exec"] = (df.worker_exec_complete - df.worker_exec_duration - df.worker_action_received) / 1000000.0

    df["network_send"] = (df.worker_action_received) / 1000000.0
    df["network_receive"] = (df.controller_result_enqueue - df.worker_result_sent) / 1000000.0
    df["network_duration"] = df.network_send + df.network_receive


    df["result_queue"] = (df.controller_action_duration - df.controller_result_enqueue) / 1000000.0



    grouped = df.groupby("bucket")
    preexec = percentiles(grouped, grouped.pre_exec)
    preoutput = percentiles(grouped, grouped.worker_pre_output)
    preresult = percentiles(grouped, grouped.worker_pre_completion)

    network = percentiles(grouped, grouped.network_duration)
    netsend = percentiles(grouped, grouped.network_send)
    netrcv = percentiles(grouped, grouped.network_receive)

    resultqueue = percentiles(grouped, grouped.result_queue)

    worker = percentiles(grouped, grouped.worker_duration)
    controller = percentiles(grouped, grouped.controller_duration)


    plt.clf()
    ax = plt.gca()
    fig, axs = plt.subplots(9, figsize=(15,15))
    fig.suptitle("Where is the time spent? (ms)")

    plot_percentiles(netsend, axs[0])
    axs[0].set(xlabel="", ylabel="Network Send")

    plot_percentiles(preexec, axs[1])
    axs[1].set(xlabel="", ylabel="Worker Pre-Execute")

    plot_percentiles(preoutput, axs[2])
    axs[2].set(xlabel="", ylabel="Worker Pre-Output")

    plot_percentiles(preresult, axs[3])
    axs[3].set(xlabel="", ylabel="Worker Pre-Completion")

    plot_percentiles(netrcv, axs[4])
    axs[4].set(xlabel="", ylabel="Network Receive")

    plot_percentiles(resultqueue, axs[5])
    axs[5].set(xlabel="", ylabel="Result Queue")

    plot_percentiles(network, axs[6])
    axs[6].set(xlabel="", ylabel="Network")

    plot_percentiles(worker, axs[7])
    axs[7].set(xlabel="", ylabel="Worker")

    plot_percentiles(controller, axs[8])
    axs[8].set(xlabel="", ylabel="Controller")

    plt.savefig("%s/%s.pdf" % (args.outputdir, "time"))




def plot_clocks(args, df):
    # Only plot successful infer actions
    df = df[df.status == 0]
    df = df[df.action_type == 2]

    df["clock_delta"] = df.worker_gpu_clock - df.expected_gpu_clock

    grouped = df.groupby("bucket")
    ts1 = percentiles(grouped, grouped.worker_gpu_clock)
    ts2 = percentiles(grouped, grouped.clock_delta)

    plt.clf()
    ax = plt.gca()
    fig, axs = plt.subplots(2)
    fig.suptitle("GPU clock speed")

    plot_percentiles(ts1, axs[0])

    axs[0].set(xlabel="", ylabel="Clock Speed (MHz)")
    #axs[0].set_ylim(ymin=0)

    plot_percentiles(ts2, axs[1])

    axs[1].set(xlabel="Time (minutes)", ylabel="Clock Delta (MHz)")
    plt.savefig("%s/%s.pdf" % (args.outputdir, "clock"))

def plot_execution_time_predictions(args, df):
    # Only plot successful infer actions
    df = df[df.status == 0]
    df = df[df.action_type == 2]

    # Prediction errors
    df["exec_error"] = (df.expected_exec_duration - df.worker_exec_duration) / 1000000
    df["exec_error_relative"] = 100 * (df.expected_exec_duration - df.worker_exec_duration) / df.worker_exec_duration

    grouped = df.groupby("bucket")
    ts1 = percentiles(grouped, grouped.exec_error)
    ts2 = percentiles(grouped, grouped.exec_error_relative)

    plt.clf()
    ax = plt.gca()
    fig, axs = plt.subplots(4)
    fig.suptitle("Execution Time Error (Predicted - Actual)")
    plot_percentiles(ts1, axs[0])
    axs[0].set(xlabel="", ylabel="Error (ms)")
    plot_percentiles(ts1, axs[1])
    axs[1].set(xlabel="", ylabel="Error (ms)")
    axs[1].set_ylim(ymin=-5, ymax=5)
    plot_percentiles(ts2, axs[2])
    axs[2].set(xlabel="Time (minutes)", ylabel="Error (%)")
    plot_percentiles(ts2, axs[3])
    axs[3].set(xlabel="Time (minutes)", ylabel="Error (%)")
    axs[3].set_ylim(ymin=-5, ymax=5)
    plt.savefig("%s/%s.pdf" % (args.outputdir, "infer_predictions"))

def plot_loadweights_predictions(args, df):
    # Only plot successful loadweights actions
    df = df[df.status == 0]
    df = df[df.action_type == 1]

    # Prediction errors
    df["exec_error"] = (df.expected_exec_duration - df.worker_exec_duration) / 1000000
    df["exec_error_relative"] = 100 * (df.expected_exec_duration - df.worker_exec_duration) / df.worker_exec_duration

    grouped = df.groupby("bucket")
    ts1 = percentiles(grouped, grouped.exec_error)
    ts2 = percentiles(grouped, grouped.exec_error_relative)

    plt.clf()
    ax = plt.gca()
    fig, axs = plt.subplots(4)
    fig.suptitle("LoadWeights Time Error (Predicted - Actual)")
    plot_percentiles(ts1, axs[0])
    axs[0].set(xlabel="", ylabel="Error (ms)")
    plot_percentiles(ts1, axs[1])
    axs[1].set(xlabel="", ylabel="Error (ms)")
    axs[1].set_ylim(ymin=-5, ymax=5)
    plot_percentiles(ts2, axs[2])
    axs[2].set(xlabel="Time (minutes)", ylabel="Error (%)")
    plot_percentiles(ts2, axs[3])
    axs[3].set(xlabel="Time (minutes)", ylabel="Error (%)")
    axs[3].set_ylim(ymin=-5, ymax=5)
    plt.savefig("%s/%s.pdf" % (args.outputdir, "loadweights_predictions"))

def plot_completion_time_predictions(args, df):
    # Only plot successful infer actions
    df = df[df.status == 0]
    df = df[df.action_type == 2]

    df["completion_error"] = (df.expected_exec_complete - df.worker_exec_complete) / 1000000
    df["completion_error_relative"] = 100 * (df.expected_exec_complete - df.worker_exec_complete) / df.worker_exec_complete

    grouped = df.groupby("bucket")
    ts1 = percentiles(grouped, grouped.completion_error)
    ts2 = percentiles(grouped, grouped.completion_error_relative)

    plt.clf()
    ax = plt.gca()



    fig, axs = plt.subplots(4)
    fig.suptitle("Infer Completion Time Error (Predicted - Actual)")
    plot_percentiles(ts1, axs[0])
    axs[0].set(xlabel="", ylabel="Error (ms)")
    plot_percentiles(ts1, axs[1])
    axs[1].set(xlabel="", ylabel="Error (ms)")
    axs[1].set_ylim(ymin=-5, ymax=5)
    plot_percentiles(ts2, axs[2])
    axs[2].set(xlabel="Time (minutes)", ylabel="Error (%)")
    plot_percentiles(ts2, axs[3])
    axs[3].set(xlabel="Time (minutes)", ylabel="Error (%)")
    axs[3].set_ylim(ymin=-5, ymax=5)
    plt.savefig("%s/%s.pdf" % (args.outputdir, "infer_complete_predictions"))

# def plot_completion_time_predictions2(args, df):
#     outputdir = args.outputdir

#     df = df.dropna()

#     # Only plot successful infer actions
#     df = df[df.status == 0]
#     df = df[df.action_type == 2]

#     # Drop the first occurrence of each model and batch size, which won't have an estimate
#     df = df[df.groupby(["model_id", "batch_size"]).cumcount() != 0]

#     # Drop the first 20 minutes
#     leadin = 20 * 60 * 1000000000
#     mint = df.t.min()
#     df = df[df.t > mint + leadin]

#     # Use the next 8 hours
#     duration = 8 * 60 * 60 * 1000000000
#     df = df[df.t <= (mint + leadin + duration)]

#     # Bucket by execution time - default 250ms buckets
#     bucketsize = 1000000
#     df["bucket"] = (df.worker_exec_duration / bucketsize).astype("int64") * bucketsize / 1000000
#     df["prediction"] = df.expected_exec_duration / 1000000
#     df["prediction_error"] = (df.worker_exec_duration - df.expected_exec_duration) / df.worker_exec_duration

#     # Don't include buckets without enough samples
#     df = df.groupby("bucket").filter(lambda x: len(x) > 100)

#     quantiles = list(reversed([0.01, 0.25, 0.5, 0.75, 0.99]))
#     columns = ["p%s" % (q * 100) for q in quantiles]

#     quantiledata = df.groupby("bucket").prediction_error.quantile(quantiles)
#     quantiledata = quantiledata.unstack()
#     quantiledata.columns = columns
#     quantiledata = quantiledata.reset_index()
#     quantiledata["ideal"] = quantiledata["bucket"]
#     quantiledata = quantiledata.set_index("bucket")

#     plt.clf()
#     ax = plt.gca()
#     for column in columns:
#         quantiledata.plot(kind="line", y=column, ax=ax)

#     #quantiledata.plot(kind="line", y="ideal", ax=ax)
#     plt.title("")
#     plt.ylabel("Predicted execution latency")
#     plt.xlabel("Actual execution latency")
#     # plt.yscale("log")
#     # plt.xscale("log")
#     plt.savefig("%s/%s.pdf" % (outputdir, "exec_latency_predictions"))

def plot_loadweights_completion_time_predictions(args, df):
    # Only plot successful loadweights actions
    df = df[df.status == 0]
    df = df[df.action_type == 1]

    df["completion_error"] = (df.expected_exec_complete - df.worker_exec_complete) / 1000000
    df["completion_error_relative"] = 100 * (df.expected_exec_complete - df.worker_exec_complete) / df.worker_exec_complete

    grouped = df.groupby("bucket")
    ts1 = percentiles(grouped, grouped.completion_error)
    ts2 = percentiles(grouped, grouped.completion_error_relative)

    plt.clf()
    ax = plt.gca()
    fig, axs = plt.subplots(4)
    fig.suptitle("LoadWeights Completion Time Error (Predicted - Actual)")
    plot_percentiles(ts1, axs[0])
    axs[0].set(xlabel="", ylabel="Error (ms)")
    plot_percentiles(ts1, axs[1])
    axs[1].set(xlabel="", ylabel="Error (ms)")
    axs[1].set_ylim(ymin=-5, ymax=5)
    plot_percentiles(ts2, axs[2])
    axs[2].set(xlabel="Time (minutes)", ylabel="Error (%)")
    plot_percentiles(ts2, axs[3])
    axs[3].set(xlabel="Time (minutes)", ylabel="Error (%)")
    axs[3].set_ylim(ymin=-5, ymax=5)
    plt.savefig("%s/%s.pdf" % (args.outputdir, "loadweights_complete_predictions"))

def plot_throughput_batchsize_scatter(args, df):
    # Only plot infer actions
    df = df[df.action_type == 2]

    duration = (df["t"].max() - df["t"].min()) / 1000000000

    # Create scatter plots
    grouped = df.groupby("model_id")

    models = grouped.count()[[]]
    models["throughput"] = grouped.batch_size.sum() / duration
    models["batchsize_mean"] = grouped.batch_size.mean()

    plt.clf()
    ax = plt.gca()
    models.plot(kind='scatter',x='batchsize_mean', y='throughput',ax=ax)
    plt.title("Model throughput vs. average batch size")
    plt.ylabel("Model throughput")
    plt.xlabel("Average batch size")
    plt.savefig("%s/%s.pdf" % (args.outputdir, "throughput_batchsize_scatter"))

def plot_throughput_batchsize_scatter_windowed(args, df):
    # Only plot infer actions
    df = df[df.action_type == 2]

    grouped = df.groupby(["t_sec", "model_id"])

    windowed_data = grouped.count()[[]]
    windowed_data["throughput"] = grouped.batch_size.sum()
    windowed_data["batchsize_mean"] = grouped.batch_size.mean()
    windowed_data = windowed_data.reset_index()

    plt.clf()
    ax = plt.gca()
    # plt.hist2d(windowed_data.batchsize_mean, windowed_data.throughput, weights=windowed_data.batchsize_mean, bins=100, cmap='Blues')
    windowed_data.plot(kind='scatter',x='batchsize_mean', y='throughput',ax=ax, s=.1)
    # sns.kdeplot(windowed_data.batchsize_mean, windowed_data.throughput, cmap="Reds", shade=True)
    plt.title("Model throughput vs. average batch size, per model, per 1sec window")
    plt.ylabel("Model throughput")
    plt.xlabel("Average batch size")
    plt.savefig("%s/%s.pdf" % (args.outputdir, "throughput_batchsize_scatter_windowed"))

def plot_infer_loadweights_scatter_windowed(args, df):

    df["is_load"] = df.action_type == 1
    df["is_infer"] = df.action_type == 2
    df["is_evict"] = df.action_type == 3

    grouped = df.groupby(["t_sec", "model_id"])

    windowed_data = grouped.count()[[]]
    windowed_data["is_load"] = grouped.is_load.sum()
    windowed_data["is_infer"] = grouped.is_infer.sum()
    windowed_data = windowed_data.reset_index()

    plt.clf()
    ax = plt.gca()
    # plt.hist2d(windowed_data.batchsize_mean, windowed_data.throughput, weights=windowed_data.batchsize_mean, bins=100, cmap='Blues')
    windowed_data.plot(kind='scatter',x='is_infer', y='is_load',ax=ax, s=.1)
    # sns.kdeplot(windowed_data.batchsize_mean, windowed_data.throughput, cmap="Reds", shade=True)
    plt.title("Loads and Infers for each model over 1-second windows")
    plt.ylabel("Loads")
    plt.xlabel("Infers")
    plt.savefig("%s/%s.pdf" % (args.outputdir, "loads_vs_infers"))


def plot_time_since_load(args, df):

    df["is_load"] = df.action_type == 1
    df["is_infer"] = df.action_type == 2
    df["is_evict"] = df.action_type == 3

    grouping = ["model_id", "worker_id", "gpu_id"]
    df["load_iteration"] = df.groupby(grouping)["is_load"].cumsum()
    df['time_since_last_load'] = df.groupby(grouping + ["load_iteration"])['t'].diff()
    df.loc[:, 'time_since_last_load'] = df.groupby(grouping + ["load_iteration"])['time_since_last_load'].cumsum().fillna(0)
    df.loc[(df.action_type == 1), "time_since_last_load"] = df[df.action_type == 1].groupby(grouping)['t'].diff().fillna(0)


    grouping = ["model_id", "worker_id", "gpu_id"]
    df["evict_iteration"] = df.groupby(grouping)["is_evict"].cumsum()
    df['time_since_last_evict'] = df.groupby(grouping + ["evict_iteration"])['t'].diff()
    df.loc[:, 'time_since_last_evict'] = df.groupby(grouping + ["evict_iteration"])['time_since_last_evict'].cumsum().fillna(0)
    df.loc[(df.action_type == 3), "time_since_last_evict"] = df[df.action_type == 3].groupby(grouping)['t'].diff().fillna(0)

    grouped = df[df.action_type == 2].groupby(grouping + ["load_iteration"])

    ts = grouped.count()[[]]
    ts["work"] = grouped.worker_exec_duration.sum()
    ts["duration"] = grouped.time_since_last_load.max()

    plt.clf()
    ax = plt.gca()
    ts.plot(kind='scatter',x='duration', y='work',ax=ax)
    plt.title("LoadWeights Duration vs. Work Done")
    plt.xlabel("Duration")
    plt.ylabel("Work")
    plt.savefig("%s/%s.pdf" % (outputdir, "work_vs_loadweights"))


def load_action_log(filename):    
    df = pd.read_csv(filename, sep="\t", header=0)
    df = df.dropna()     
    df["minor"] = df.model_id == 3600 # Model ID 3600 is the "Minor" workload
    begin = df[df.minor == False].t.min() # t=0 is when the major workload begins
    df.t = df.t - begin
    df["t_sec"] = (df.t / 1000000000).apply(np.floor).astype("int64")
    df["t_10sec"] = (df.t_sec / 10.0).apply(np.floor).astype("int64") / 6
    df = df[df.t_sec <= 3600] # Experiment lasts 60 mins after begin
    return df

def make_timeseries(df):
    timeseries = df.groupby("t_10sec").count()[[]]
    timeseries["pci-utilization"] = df[df.action_type == 1].groupby("t_10sec").worker_exec_duration.sum() / 100000000.0
    timeseries["pci-goodput"] = df[df.action_type == 1].groupby("t_10sec").goodput.sum() / 100000000.0
    timeseries["gpu-utilization"] = df[df.action_type == 2].groupby("t_10sec").worker_exec_duration.sum() / 100000000.0
    timeseries["gpu-goodput"] = df[df.action_type == 2].groupby("t_10sec").goodput.sum() / 100000000.0
    return timeseries.fillna(0)

def plot_pci(ts, outputfile):
    plt.clf()
    ax = plt.gca()
    ts.plot(kind='line',y='pci-utilization',ax=ax, label="Utilization")
    ts.plot(kind='line',y='pci-goodput',ax=ax, label="Goodput")
    plt.title("(6d) PCI Utilization")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Utilization (%)")
    plt.savefig(outputfile)

def plot_gpu(ts, outputfile):
    plt.clf()
    ax = plt.gca()
    ts.plot(kind='line',y='gpu-utilization',ax=ax, label="Utilization")
    ts.plot(kind='line',y='gpu-goodput',ax=ax, label="Goodput")
    plt.title("(6e) GPU Utilization")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Utilization (%)")
    plt.savefig(outputfile)


def process(args):

    inputfile = args.inputfile
    outputdir = args.outputdir

    print("Loading %s" % inputfile)
    df = load_action_log(inputfile)
    print("Loaded %d rows" % (len(df)))

    print("Plotting")
    timeseries = make_timeseries(df)
    timeseries.to_csv("%s/%s.tsv" % (outputdir, "action_timeseries_data"), sep="\t", index=False)
    plot_pci(timeseries, "%s/%s.pdf" % (outputdir, "6d-pci"))
    plot_gpu(timeseries, "%s/%s.pdf" % (outputdir, "6e-gpu"))




if __name__ == '__main__':
    args = parser.parse_args()
    exit(process(args))