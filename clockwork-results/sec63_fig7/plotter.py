import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd

# Create graph directory, relative to pwd
expdir = os.path.dirname(os.path.realpath(__file__))
graphdir = expdir + "/data/"
if not os.path.exists(graphdir): os.makedirs(graphdir)

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--logdir', help='Path to the log files', \
  type=str, required=True, dest='logdir')
args = parser.parse_args()

# Experiment settings
exp_name = "slo-exp-1"
model_name = "Resnet50-v2"
distributions = ["poisson"]
rates = [600, 1200, 2400]
num_models_vals = [12, 48]

def get_lines_from_file(filename):
  try:
    with open(filename) as f:
      lines = f.readlines()
      return [x.strip() for x in lines]
  except:
    print("Reading " + filename + " failed")
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
  else: return 64 / num_bg
  
def get_slo_violation_rates(distribution, num_models, rate):
  # Parse controller request_telemetry to get the data
  config = "dist=" + distribution + "_rate=" + str(rate) + "_num-models=" + str(num_models)
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
    slo_tracker.append(deadline_met)
    slo_tracker_ts.append(timestamp - min_ts)
    #if len(slo_tracker) == 1000:
    if len(slo_tracker) > 1:
      duration = max(slo_tracker_ts) - min(slo_tracker_ts)
      if (duration > 4000000000): # 1s
        X.append(timestamp - min_ts)
        Y.append(slo_tracker.count(0) / float(len(slo_tracker)))
        slo_tracker = []
        slo_tracker_ts = []
  
  return [X, Y]

def get_slo_change_ts_and_labels(distribution, num_models, rate):
  # Parse controller request_telemetry to get the data
  config = "dist=" + distribution + "_rate=" + str(rate) + "_num-models=" + str(num_models)
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

for distribution in distributions:
    print("Plotting for " + str(distribution))

    # Plot graphs for this experiment run/config
    fig = plt.figure(figsize=(10, 2.5))
    ax = fig.add_subplot(111)
    
    #title = distribution + " open-loop client"
    #ax.set_title(title)
    
    #xlabel = "SLO factor (deadline = SLO factor x single request latency for batch size 1)"
    plt.xlabel("SLO Multiplier", fontsize=13)
    
    ax.set_ylabel('Workload Satisfaction', color="black", fontsize=13)
    
    print("Plotting vertical lines")
    [A, B] = get_slo_change_ts_and_labels(distribution, num_models_vals[0], rates[0])
    for a in A: plt.axvline(x=a, color='grey', linewidth=0.5, linestyle='--')
    
    colors = ['gray', 'mediumturquoise', 'orange'] #, 'orange']
    ls = ['-', ':', '--'] #, '-.']
    lw = [1, 3]
    colors_idx = 0;
    ls_idx = 0;
    lw_idx = 0;

    x_last = 0

    data = {}
    data["SLO_multiplier_TS"] = A
    data["SLO_multiplier_val"] = B
    df = pd.DataFrame(data)
    print(df)
    tsvfile = "fig7_dist=" + distribution + "_slo_multiplier_changes.tsv"
    print("Generating " + tsvfile)
    df.to_csv(graphdir + tsvfile, sep="\t", index=False)

    data = {}
    for num_models in num_models_vals:
      for rate in rates:
        print("Plotting for rate " + str(rate))
        [X, Y] = get_slo_violation_rates(distribution, num_models, rate)
        Y = [ (1.0 - y) for y in Y ]
        label = "N=" + str(num_models) + ", R=" + str(rate)
        x_last = max(x_last, max(X))
        plt.plot(X, Y, label=label, color=colors[colors_idx], linestyle=ls[ls_idx], linewidth=lw[lw_idx]);
        print("Plotting for rate " + str(rate) + " done")
        colors_idx = (colors_idx + 1) % len(colors)
        ls_idx = (ls_idx + 1) % len(ls)

        data['N=' + str(num_models) + "_R=" + str(rate) + "_TS"] = X
        data['N=' + str(num_models) + "_R=" + str(rate) + "_WS"] = Y
      lw_idx = (lw_idx + 1) % len(lw)

    df = pd.DataFrame(data)
    print(df)
    
    ax.legend(loc="lower right", ncol=1, frameon=True, markerscale=8, fontsize=8)
    ax.set_xlim((A[0], x_last))
    plt.xticks(A, B, rotation=90, fontsize=10)
    plt.yticks(fontsize=10)

    tsvfile = "fig7_dist=" + distribution + "_workload_satisfaction.tsv"
    print("Generating " + tsvfile)
    df.to_csv(graphdir + tsvfile, sep="\t", index=False)
    plotfile = "fig7_dist1=" + distribution + ".pdf"
    print("Generating " + plotfile)
    plt.savefig(graphdir + plotfile, bbox_inches='tight', transparent=False)

plt.show()
exit()
