import pandas as pd
import sys
import os

def calculate_average_error(file_path):
    try:
        df = pd.read_csv(file_path, sep='\t')  # Assuming the file is tab-separated
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

    if 'expected_exec_duration' not in df.columns or 'worker_exec_duration' not in df.columns:
        print(f"Required columns not found in {file_path}")
        return None

    df['error'] = abs(df['expected_exec_duration'] - df['worker_exec_duration'])
    return df['error'].mean()

def main(logdir):
    schedule_aheads = [3000000, 4000000, 5000000, 6000000, 7000000]
    min_error = float('inf')
    min_error_schedule_ahead = None

    for schedule_ahead in schedule_aheads:
        file_name = f"{logdir}/file=controller_action_{schedule_ahead}.tsv"
        average_error = calculate_average_error(file_name)
        if average_error is not None and average_error < min_error:
            min_error = average_error
            min_error_schedule_ahead = schedule_ahead

    if min_error_schedule_ahead is not None:
        print(min_error_schedule_ahead)
        return min_error_schedule_ahead
    else:
        print("No valid data found.")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <logdir>")
        sys.exit(1)
    else:
        logdir = sys.argv[1]
        main(logdir)
