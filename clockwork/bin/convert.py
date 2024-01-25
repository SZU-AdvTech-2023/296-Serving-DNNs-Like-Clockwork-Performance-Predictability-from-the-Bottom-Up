import argparse
import os
import subprocess


"""
Clockwork must be build in order to run this script.

This script uses the `convert` binary and expects it to exist in the `build` folder
"""

convert_exec = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../build/convert")


parser = argparse.ArgumentParser(description='Convert a TVM model into a Clockwork model')
parser.add_argument("input_dir", metavar="INDIR", type=str, help="Base directory where TVM models exist.  The utility expects multiple models, one per batch size, each in a subdirectory.")
parser.add_argument("output_dir", metavar="OUTDIR", type=str, help="Output directory.  Directory will be created if it does not exist.")
parser.add_argument('-p', "--page_size", type=int, default=16777216, help="Page size to use for compiled models.  16MB by default.")
parser.add_argument("--subdir_prefix", type=str, default="b", help="Within input_dir, a prefix for how subdirectories are named.  Default \"b\" followed by the batch size.")


def find_tvm_models(path):
	dir_contents = os.listdir(path)
	dir_contents_paths = [os.path.join(path, c) for c in dir_contents]
	dir_files = [c for c in dir_contents_paths if os.path.isfile(c)]

	so_files = [f[:-3] for f in dir_files if f.endswith(".so")]
	model_choices = [f for f in so_files if is_model(f)]
	return model_choices

def is_model(path_prefix):
	suffixes = ["so", "params", "json"]
	for suffix in suffixes:
		if not os.path.exists("%s.%s" % (path_prefix, suffix)):
			return False
	return True

def find_models(path, subdir_prefix):
	found_models = []
	for entry in os.listdir(path):
		entry_path = os.path.join(path, entry)
		if not os.path.isdir(entry_path):
			print("Ignoring non-directory %s" % entry_path)
			continue
		if not entry.startswith(subdir_prefix):
			print("Skipping non-matching (prefix=\"%s\") directory %s " % (subdir_prefix, entry_path))
			continue

		candidates = find_tvm_models(entry_path)
		if len(candidates) == 0:
			print("Skipping directory with no valid models (expect .so .json and .params with matching names) %s" % entry_path)
			continue
		if len(candidates) > 1:
			print("Skipping directory with multiple valid models %s" % entry_path)
			continue

		batch_size = int(entry[len(subdir_prefix):])

		found_models.append((batch_size, candidates[0]))

	return sorted(found_models)


def convert(args):

	models = find_models(args.input_dir, args.subdir_prefix)

	print("Found %d models in input directory %s:" % (len(models), args.input_dir))
	for batch_size, model in models:
		print("  %d %s" % (batch_size, model))

	# Create output directory
	if not os.path.exists(args.output_dir):
		print("Output directory %s will be created" % args.output_dir)
	else:
		print("Will output to existing directory %s" % args.output_dir)

	print("The following command will run:")

	pargs = [str(v) for v in [
		convert_exec,
		"-o", args.output_dir,
		"-p", args.page_size
	] + [x for m in models for x in m]]
	print(" ".join(pargs))
	print("Press <return> to continue or CTRL-C to abort")
	input()

	if not os.path.exists(args.output_dir):
		print("Created output directory %s" % args.output_dir)
		os.makedirs(args.output_dir)

	popen = subprocess.Popen(pargs)
	popen.wait()

if __name__ == '__main__':
    args = parser.parse_args()
    exit(convert(args))