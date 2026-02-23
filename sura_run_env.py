import warnings
warnings.filterwarnings("ignore")

import os
import sys
import shutil
import torch
import numpy as np

import dataset.dataset_builder as dataset_builder
import model.model_builder as model_builder
import model.trainer_builder as trainer_builder
import policy.envs.env_builder as env_builder
import policy.learning.agent_builder as agent_builder

from policy.common.misc_utils import EpisodeRunner
import util.arg_parser as arg_parser
import util.rand_util as rand_util
import util.mp_util as mp_util

# 환경 변수 설정
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29501"

sys.argv = [
    "sura_run_env.py",
    "--env_config", "config/model/sura_lafan1.yaml"]

def set_np_formatting():
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)
    return

def build_dataset(config, load_full_dataset):
    dataset = dataset_builder.build_dataset(config, load_full_dataset)
    return dataset

def load_dataset(config):
    dataset = build_dataset(config, load_full_dataset=True)
    return dataset

def build_env(config, model, dataset, device):
    env = env_builder.build_bvh_envs(config, model, dataset, device)
    return env

def load_args(argv):
    args = arg_parser.ArgParser()
    args.load_args(argv[1:])

    arg_file = args.parse_string("arg_file", "")
    if (arg_file != ""):
        succ = args.load_file(arg_file)
        assert succ, print("Failed to load args from: " + arg_file)

    return args

def test_no_agent(env):
    env.reset()
    env.reset_initial_frames()
    with EpisodeRunner(env) as runner:
        while not runner.done:

            #frame, lfoot_vec, rfoot_vec, lreon, rrecon= env.get_next_frame()
            frame, lfoot_vec, rfoot_vec = env.get_next_frame()

            for i in range(env.frame_skip):
                _, reward, done, info = env.calc_env_state(frame, lfoot_vec, rfoot_vec)#, lreon, rrecon)
                '''if done.any():
                    reset_indices = env.parallel_ind_buf.masked_select(done.squeeze())
                    env.reset_index(reset_indices)'''
                #try:
                #    if info.get("reset").all():
                #        env.reset()
                #except:
                #    if info.get("reset"):
                #        env.reset()
    return

def run(args):
    device = args.parse_string("device", 'cuda:0')
    config = args.parse_string("env_config", "")
    dataset = load_dataset(config)
    model = None
    env = build_env(config, model, dataset, device)
    test_no_agent(env)

def main(argv):
    args = load_args(argv)
    run(args)
    return


if __name__ == "__main__":
    main(sys.argv)
