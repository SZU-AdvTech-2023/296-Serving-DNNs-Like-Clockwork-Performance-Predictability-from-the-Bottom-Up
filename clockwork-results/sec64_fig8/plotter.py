import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import os
import pandas as pd

expdir = os.path.dirname(os.path.realpath(__file__))
graphdir = expdir + "/graphs/"
if not os.path.exists(graphdir): os.makedirs(graphdir)

parser = argparse.ArgumentParser()
# parser.add_argument('-l', '--logdir', help='Path to the log files', \
#   type=str, required=True, dest='logdir')
parser.add_argument('-l', '--logdir', help='Path to the log files', \
  type=str, default="/home/hzq/results/slo-exp-2/log/2023-12-08-11-04-07", dest='logdir')
args = parser.parse_args()

exp_name = "slo-exp-2"
model_name = "Resnet50-v2"
distributions = ["poisson"] #, "fixed-rate"]
num_fg=6
rate_fg=200
num_bgs = [0, 12, 48] #, 192]

def get_lines_from_file(filename):
  try:
    with open(filename) as f:
      lines = f.readlines()
      return [x.strip() for x in lines]
  except Exception as e:
    print("Reading " + filename + " failed")
    print("Excetion: " + str(e))
    exit()

def get_params(line):
  try:
    tokens = line.split()
    timestamp = int(tokens[0])
    request_id = int(tokens[1])
    result = int(tokens[2])
    user_id = int(tokens[3])
    model_id = int(tokens[4])
    slo_factor = float(tokens[5])
    latency = int(tokens[6]) / 1000000.0 # convert from ns to ms
    deadline = int(tokens[7]) / 1000000.0 # convert from ns to ms
    deadline_met = int(tokens[8])
  except Exception as e:
    raise e
  return [  timestamp, request_id, result, user_id, model_id, slo_factor, \
    latency, deadline, deadline_met ]

def get_concurrency(num_bg):
  if num_bg == 0: return 1
  else: return int(192 / num_bg)
  
def get_slo_violation_rates(distribution, rate_fg, num_bg, concurrency):

  config = "dist=" + distribution + "_rate-fg=" + str(rate_fg) + \
    "_num-bg=" + str(num_bg) + "_concurrency-bg=" + str(concurrency)

  # Parse controller request_telemetry to get the data
  logfile = args.logdir + "/file=controller_" + config + "_request.csv"
  lines = get_lines_from_file(logfile)

  min_ts = -1
  for line in lines[1:-1]:
    try: [timestamp, request_id, result, user_id, model_id, slo_factor, latency, deadline, deadline_met] = get_params(line)
    except Exception as e: continue
    if min_ts == -1: min_ts = timestamp
    else: min_ts = min(min_ts, timestamp)
  
  slo_tracker = []
  slo_tracker_ts = []
  X = []
  Y = []
  for line in lines[1:-1]:
    try: [timestamp, request_id, result, user_id, model_id, slo_factor, latency, deadline, deadline_met] = get_params(line)
    except Exception as e: continue
    if user_id >= num_fg: continue
    slo_tracker.append(deadline_met)
    slo_tracker_ts.append(timestamp - min_ts)
    #if len(slo_tracker) == 1000:
    if len(slo_tracker) > 1:
      duration = max(slo_tracker_ts) - min(slo_tracker_ts)
      if (duration > 5000000000): # 1s
        X.append(timestamp - min_ts)
        Y.append(slo_tracker.count(0) / float(len(slo_tracker)))
        slo_tracker = []
        slo_tracker_ts = []
  
  return [X, Y]

def get_slo_change_ts_and_labels(distribution, rate_fg, num_bg, concurrency):

  config = "dist=" + distribution + "_rate-fg=" + str(rate_fg) + \
    "_num-bg=" + str(num_bg) + "_concurrency-bg=" + str(concurrency)

  # Parse controller request_telemetry to get the data
  logfile = args.logdir + "/file=controller_" + config + "_request.csv"
  lines = get_lines_from_file(logfile)

  min_ts = -1
  for line in lines[1:-1]:
    try: [timestamp, request_id, result, user_id, model_id, slo_factor, latency, deadline, deadline_met] = get_params(line)
    except Exception as e: continue
    if min_ts == -1: min_ts = timestamp
    else: min_ts = min(min_ts, timestamp)
  
  slo_factor_ts = {}
  for line in lines[1:-1]:
    try: [timestamp, request_id, result, user_id, model_id, slo_factor, latency, deadline, deadline_met] = get_params(line)
    except Exception as e: continue
    if user_id != 0: continue
    if slo_factor not in slo_factor_ts: slo_factor_ts[slo_factor] = []
    slo_factor_ts[slo_factor].append(timestamp - min_ts)
  
  for slo_factor in slo_factor_ts:
    slo_factor_ts[slo_factor] = [min(slo_factor_ts[slo_factor]), max(slo_factor_ts[slo_factor])]
  
  Y = list(slo_factor_ts.keys())
  Y.sort()
  X = [ slo_factor_ts[y][0] for y in Y ]
  Y = [ ('%4.1f' % y) for y in Y ]

  return [X, Y]

def get_batch_throughput(distribution, rate_fg, num_bg, concurrency):

  config = "dist=" + distribution + "_rate-fg=" + str(rate_fg) + \
    "_num-bg=" + str(num_bg) + "_concurrency-bg=" + str(concurrency)

  # Parse controller request_telemetry to get the data
  logfile = args.logdir + "/file=controller_" + config + "_request.csv"
  lines = get_lines_from_file(logfile)

  min_ts = -1
  for line in lines[1:-1]:
    try: [timestamp, request_id, result, user_id, model_id, slo_factor, latency, deadline, deadline_met] = get_params(line)
    except Exception as e: continue
    if min_ts == -1: min_ts = timestamp
    else: min_ts = min(min_ts, timestamp)
  
  req_tracker = []
  req_tracker_ts = []
  X = []
  Y = []
  for line in lines[1:-1]:
    try: [timestamp, request_id, result, user_id, model_id, slo_factor, latency, deadline, deadline_met] = get_params(line)
    except Exception as e: continue
    if user_id < num_fg: continue
    req_tracker.append(deadline_met)
    req_tracker_ts.append(timestamp - min_ts)
    if len(req_tracker) > 1:
      duration = max(req_tracker_ts) - min(req_tracker_ts)
      if (duration > 5000000000): # 1s
        X.append(timestamp - min_ts)
        Y.append((req_tracker.count(1) / float(duration)) * 1000000000)
        req_tracker = []
        req_tracker_ts = []
  
  return [X, Y]

for distribution in distributions:
  print("Plotting for latency-sensitive models with " + \
    str(distribution) + " open loop client and total rate " + \
    str(rate_fg) + " rps")
  
  # Plot graphs for this experiment run/config
  fig = plt.figure(figsize=(10, 2.5))
  ax = fig.add_subplot(111)

  #title = "dist=" + distribution + "_rate-fg=" + str(rate_fg)
  #ax.set_title(title)

  label = "SLO Multiplier"
  plt.xlabel(label, fontsize=13)

  ax.set_ylabel('Workload Satisfaction', color="black", fontsize=13)

  print("Plotting vertical lines")
  [A, B] = get_slo_change_ts_and_labels(distribution, rate_fg, 0, get_concurrency(0))
  for a in A: plt.axvline(x=a, color='grey', linewidth=0.5, linestyle='--')

  colors = ['gray', 'orange', 'mediumturquoise',  'orange']
  ls = ['-', '-.', '--', ':']
  lw = [3, 1]
  colors_idx = 0;
  ls_idx = 0;
  lw_idx = 0;

  x_last = 0

  data = {}
  data["SLO_multiplier_TS"] = A
  data["SLO_multiplier_val"] = B
  # df = pd.DataFrame(data)
  # print(df)
  tsvfile = "fig8_dist=" + distribution + "_slo_multiplier_changes.tsv"
  print("Generating " + tsvfile)
  # df.to_csv(graphdir + tsvfile, sep="\t", index=False)

  config_labels = {}
  config_labels[0] = "(a)"
  config_labels[12] = "(b)"
  config_labels[48] = "(c)"

  data = {}
  for num_bg in num_bgs:
    print("Plotting for " + str(num_bg) + " batch models")
    concurrency = get_concurrency(num_bg)
    label = config_labels[num_bg] + " M=" + str(num_bg)
    if num_bg > 0: label += "_C=" + str(concurrency)
    else: label += "_C=0"
    [X, Y] = get_slo_violation_rates(distribution, rate_fg, num_bg, concurrency)
    x_last = max(x_last, max(X))
    Y = [ (1.0 - y) for y in Y ]
    plt.plot(X, Y, label=label, color=colors[colors_idx], linestyle=ls[ls_idx], linewidth=lw[lw_idx]);
    print("Plotting for " + str(num_bg) + " batch models done")
    colors_idx += 1
    ls_idx += 1

    data[label[4:] + "_TS"] = X
    data[label[4:] + "_WS"] = Y
    
  # df = pd.DataFrame(data)
  # print(df)
  tsvfile = "fig8_dist=" + distribution + "_workload_satisfaction.tsv"
  print("Generating " + tsvfile)
  # df.to_csv(graphdir + tsvfile, sep="\t", index=False)

  legend = ax.legend(loc="lower right", title="LS Workload Satisfaction", ncol=1, \
    frameon=True, markerscale=8, fontsize=7)
  plt.setp(legend.get_title(),fontsize=8)

  plt.xticks(A, B, rotation=90, fontsize=10)
  plt.yticks(fontsize=10)
  ax.set_xlim((A[0], x_last))
  #ax.set_ylim((0, 1))

  # ax2=ax.twinx()
  # ax2.set_ylabel("Throughput (reqs/s)", color="black", fontsize=13)
  # #ax2.set_ylim([0.0, 1.0]) # HARD-CODED

  # data = {}
  # colors_idx = 1
  # ls_idx = 1
  # lw_idx += 1
  # for num_bg in num_bgs:
  #   if num_bg == 0: continue
  #   print("Plotting thpt for " + str(num_bg) + " batch models")
  #   concurrency = get_concurrency(num_bg)
  #   label = config_labels[num_bg] + " M=" + str(num_bg) + "_C=" + str(concurrency)
  #   [X, Y] = get_batch_throughput(distribution, rate_fg, num_bg, concurrency)
  #   ax2.plot(X, Y, label=label, color=colors[colors_idx], linestyle=ls[ls_idx])
  #   print("Plotting thpt for " + str(num_bg) + " batch models done")
  #   colors_idx += 1
  #   ls_idx += 1
  #   data[label[4:] + "_TS"] = X
  #   data[label[4:] + "_Tput"] = Y

  # # df = pd.DataFrame(data)
  # # print(df)
  # tsvfile = "fig8_dist=" + distribution + "_tput.tsv"
  # print("Generating " + tsvfile)
  # # df.to_csv(graphdir + tsvfile, sep="\t", index=False)

  # legend = ax2.legend(loc="lower left", title="BC Throughput", ncol=1, \
  #   frameon=True, markerscale=2, fontsize=7)
  # plt.setp(legend.get_title(),fontsize=8)
  # #ax2.set_yscale('log')
  # ax2.set_ylim([0, 3000])
  # #ax2.set_ylim(bottom=0)

  plotfile = "fig8_dist3=" + distribution + ".pdf"
  plt.savefig(graphdir + plotfile, bbox_inches='tight', transparent=False)

plt.show()
exit()
