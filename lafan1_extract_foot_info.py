import sys
import os
import glob
import numpy as np

import tqdm

import dataset.lafan1_dataset as lafan1_dataset
import dataset.util.bvh as bvh
import dataset.dataset_builder as dataset_builder
import data.LAFAN1 as LAFAN1

import dataset.util.bvh as bvh_util
import dataset.util.plot as plot_util
import dataset.util.motion_struct as motion_struct
import util.arg_parser as arg_parser

def load_args(argv):
    args = arg_parser.ArgParser()
    args.load_args(argv[1:])

    arg_file = args.parse_string("arg_file", "")
    if (arg_file != ""):
        succ = args.load_file(arg_file)
        assert succ, print("Failed to load args from: " + arg_file)

    return args

def build_dataset(config, load_full_dataset):
    dataset = dataset_builder.build_dataset(config, load_full_dataset)
    return dataset

def load_dataset(args):
    model_config_file = args.parse_string("model_config", "")
    dataset = build_dataset(model_config_file, load_full_dataset=True)

def main(argv):
    args = load_args(argv)
    load_dataset(args)
    return

if __name__ == "__main__":
    main(sys.argv)
