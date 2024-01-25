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

parser = argparse.ArgumentParser(description='Process multiple request log files for different SLOs')
parser.add_argument('inputs', metavar='SLO:FILENAME', type=str, nargs='+',
                    help='An SLO and corresponding filename, separated with a colon, e.g. 25:/local/clockwork_request_log.tsv')
parser.add_argument('-o', "--outputdir", metavar="OUTPUTDIR", type=str, default=".", help="Directory to put the processed output.  Directory will be created if it does not exist.")


def plot_throughput_hist(dfs, slos, outputdir):
    data = {
        "slo": [],
        "throughput": [],
        "goodput": []
    }
    for i in range(len(dfs)):
        df = dfs[i]
        duration = (df.t.max() - df.t.min()) / 1000000000
        throughput = len(df[df.result == 0]) / duration
        goodput = len(df[(df.result == 0) & (df.deadline_met == 1)]) / duration
        data["slo"].append(slos[i])
        data["throughput"].append(throughput)
        data["goodput"].append(goodput)

    histdata = pd.DataFrame(data)

    print(histdata.columns)


    plt.clf()
    ax = plt.gca()
    histdata.plot(kind='bar', x="slo", y="goodput", title="", ax=ax)
    plt.title("Fig.5 (left) Clockwork Goodput")
    plt.xlabel("SLO (ms)")
    plt.ylabel("Goodput (r/s)")
    plt.savefig("%s/%s.pdf" % (outputdir, "fig5_goodput"))

    histdata.to_csv("%s/%s.tsv" % (outputdir, "fig5_goodput_data"), sep="\t", index=False)



def plot_request_latency_cdfs(dfs, slos, outputdir):

    num_points = 1000
    num_elements = 10000000000000
    for i in range(len(dfs)):
        num_elements = min(len(dfs[i]), num_elements)

    quantiles = 1 - 1 / np.logspace(0, np.log10(num_elements), num=num_points)
    ys = np.linspace(0, np.log10(num_elements), num=num_points)

    data = {
        "y": ys, 
        "quantile": quantiles,    
    }

    toplot = []
    for i in range(len(dfs)):
        df = dfs[i]
        series = "%dms_SLO" % slos[i]
        df["latency_ms"] = df.latency / 1000000.0
        data[series] = df.latency_ms.quantile(quantiles)
        toplot.append(series)


    taildata = pd.DataFrame(data)

    print(taildata.columns)


    def format_fn(tick_val, tick_pos):
        percentile = (100 - math.pow(10, 2-tick_val))
        return "%s" % str(percentile)

    for v in toplot:
        plt.clf()
        ax = plt.gca()

        ax.yaxis.set_major_formatter(FuncFormatter(format_fn))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        taildata.plot(kind='line',y='y', x=v, title="", ax=ax)

        plt.title(v)
        plt.ylabel("Percentile")
        plt.xlabel("Latency (ms)")
        plt.savefig("%s/fig5_%s.pdf" % (outputdir, v))

        taildata.to_csv("%s/fig5_%s.tsv" % (outputdir, "%s_data" % v), sep="\t", index=False)




def load_request_log(filename, duration = 10, leadout=1):
    df = pd.read_csv(filename, sep="\t", header=0)
    df = df.dropna()

    t_datastart = df.t.min()

    leadout_ns = 60 * leadout * 1000000000
    duration_ns = 60 * duration * 1000000000

    t_max = df.t.max() - leadout_ns
    df = df[(df.t > t_max - duration_ns) & (df.t <= t_max)]

    print("Results: %f to %f" % ((df.t.min() - t_datastart) / 1000000000, (df.t.max() - t_datastart) / 1000000000))

    return df

# SLOs should be a list of integers
# Filenames should be a list of corresponding clockwork_request_log.tsv filenames
# e.g.
# slos = [25, 50]
# filenames = ["basedir/slo25/clockwork_request_log.tsv", "basedir/slo50/clockwork_request_log.tsv"]
def plot(slos, filenames, outputdir):
    print("Processing:")
    for i in range(len(slos)):
        print("%d ms SLO -> %s" % (slos[i], filenames[i]))

    dfs = []
    for i in range(len(slos)):
        dfs.append(load_request_log(filenames[i]))

    plot_request_latency_cdfs(dfs, slos, outputdir)
    plot_throughput_hist(dfs, slos, outputdir)


def process(args):
    slos = []
    filenames = []

    outputdir = args.outputdir
    inputs = []
    for input_pair in args.inputs:
        slo = int(input_pair.split(":")[0])
        filename = str(input_pair.split(":")[1])
        inputs.append((slo, filename))

    inputs = sorted(inputs)

    slos = [p[0] for p in inputs]
    filenames = [p[1] for p in inputs]

    plot(slos, filenames, outputdir)


if __name__ == '__main__':
    args = parser.parse_args()
    exit(process(args))