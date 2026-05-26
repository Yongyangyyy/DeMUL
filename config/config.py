"""
Training and evaluation options for DeMUL.
"""

import os
import sys
import time
import json
import pprint
import argparse
import torch
from utils.basic_utils import mkdirp, load_json, save_json, make_zipfile


def parse_with_config(parser):
    args = parser.parse_args()
    if args.config is not None:
        config_args = json.load(open(args.config))
        override_keys = {arg[2:].split('=')[0]
                         for arg in sys.argv[1:] if arg.startswith('--')}
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
    del args.config
    return args


class BaseOptions:
    saved_option_filename = "opt.json"
    ckpt_filename         = "model.ckpt"
    tensorboard_log_dir   = "tensorboard_log"
    train_log_filename    = "train.log.txt"
    eval_log_filename     = "eval.log.txt"

    def __init__(self):
        self.parser      = argparse.ArgumentParser()
        self.initialized = False
        self.opt         = None

    def initialize(self):
        self.initialized = True
        p = self.parser

        # experiment
        p.add_argument("--dset_name", type=str, default="tvr",
                        choices=["tvr", "didemo"])
        p.add_argument("--eval_split_name", type=str, default="val")
        p.add_argument("--data_ratio", type=float, default=1.0)
        p.add_argument("--debug", action="store_true")
        p.add_argument("--disable_eval", action="store_true")
        p.add_argument("--results_root", type=str, default="results")
        p.add_argument("--exp_id", type=str, default=None)
        p.add_argument("--seed", type=int, default=2018)
        p.add_argument("--device", type=int, default=0,
                        help="0 = cuda:0, -1 = cpu")
        p.add_argument("--device_ids", type=int, nargs="+", default=[0])
        p.add_argument("--num_workers", type=int, default=8)

        # training hyper-parameters
        p.add_argument("--lr", type=float, default=1e-4)
        p.add_argument("--lr_mul", type=float, default=1.0,
                        help="LR multiplier for pretrained encoder params")
        p.add_argument("--wd", type=float, default=0.01)
        p.add_argument("--n_epoch", type=int, default=50)
        p.add_argument("--max_es_cnt", type=int, default=3,
                        help="-1 to disable early stopping")
        p.add_argument("--bsz", type=int, default=32)
        p.add_argument("--eval_query_bsz", type=int, default=8)
        p.add_argument("--grad_clip", type=float, default=-1)
        p.add_argument("--eval_epoch_num", type=int, default=1)
        p.add_argument("--no_eval_untrained", action="store_true")

        # data
        p.add_argument("--max_ctx_len", type=int, default=100)
        p.add_argument("--max_desc_len", type=int, default=30)
        p.add_argument("--clip_length", type=float, default=1.5)
        p.add_argument("--ctx_mode", type=str, default="visual_sub")
        p.add_argument("--dataset_config", type=str)
        p.add_argument("--neg_video_num", type=int, default=3)
        p.add_argument("--use_extend_pool", type=int, default=1000)

        # task / loss
        p.add_argument("--stop_task", type=str, default="VCMR",
                        choices=["VCMR", "SVMR", "VR"])
        p.add_argument("--eval_tasks_at_training", type=str, nargs="+",
                        default=["VCMR", "SVMR", "VR"])
        p.add_argument("--lw_st_ed", type=float, default=0.01)
        p.add_argument("--lw_video_ce", type=float, default=0.05)

        # model architecture
        p.add_argument("--visual_dim", type=int, default=4352)
        p.add_argument("--text_dim", type=int, default=768)
        p.add_argument("--query_dim", type=int, default=768)
        p.add_argument("--hidden_dim", type=int, default=384)
        p.add_argument("--model_config", type=str)
        p.add_argument("--encoder_pretrain_ckpt_filepath", type=str,
                        default="None")
        p.add_argument("--similarity_measure", type=str, default="general",
                        choices=["general", "exclusive", "disjoint"])

        # inference / post-processing
        p.add_argument("--min_pred_l", type=int, default=0)
        p.add_argument("--max_pred_l", type=int, default=24)
        p.add_argument("--max_before_nms", type=int, default=200)
        p.add_argument("--max_vcmr_video", type=int, default=10)
        p.add_argument("--nms_thd", type=float, default=-1)
        # Use precomputed external VR scores ("inference_vr_scores" from the
        # dataset) directly, instead of recomputing query→video similarities
        # from the model output. True only for general similarity measure.
        p.add_argument("--use_interal_vr_scores", action="store_true",
                       help="whether to use precomputed VR scores; "
                            "true only for general similarity measure")

        p.add_argument("--config", help="JSON config file path")

    def display_save(self, opt):
        print("------------ Options -------------\n{}\n-------------------"
              .format(pprint.pformat(
                  {str(k): str(v) for k, v in sorted(vars(opt).items())},
                  indent=4)))
        if not isinstance(self, TestOptions):
            save_json(vars(opt),
                      os.path.join(opt.results_dir,
                                   self.saved_option_filename),
                      save_pretty=True)

    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = parse_with_config(self.parser)

        if opt.debug:
            opt.results_root = os.path.sep.join(
                opt.results_root.split(os.path.sep)[:-1]
                + ["debug_results"])

        if isinstance(self, TestOptions):
            opt.model_dir = os.path.join("ablation-results", opt.model_dir)
            saved = load_json(
                os.path.join(opt.model_dir, self.saved_option_filename))
            skip = {"results_root", "nms_thd", "debug", "dataset_config",
                    "model_config", "device", "eval_split_name",
                    "eval_query_bsz", "device_ids", "max_vcmr_video",
                    "max_pred_l", "min_pred_l"}
            for arg, val in saved.items():
                if arg not in skip:
                    setattr(opt, arg, val)
        else:
            if opt.exp_id is None:
                raise ValueError("--exp_id is required for training!")
            opt.results_dir = os.path.join(
                opt.results_root,
                "-".join([opt.dset_name, opt.exp_id,
                          time.strftime("%Y_%m_%d_%H_%M_%S")]))
            mkdirp(opt.results_dir)
            code_dir = os.path.dirname(os.path.dirname(
                os.path.realpath(__file__)))
            make_zipfile(
                code_dir,
                os.path.join(opt.results_dir, "code.zip"),
                enclosing_dir="code",
                exclude_dirs_substring="results",
                exclude_dirs=["results", "__pycache__",
                              "didemo_feature_release",
                              "tvr_feature_release"],
                exclude_extensions=[".pyc", ".ipynb", ".swap"])

        self.display_save(opt)

        assert opt.stop_task in opt.eval_tasks_at_training
        opt.ckpt_filepath      = os.path.join(opt.results_dir,
                                               self.ckpt_filename)
        opt.train_log_filepath = os.path.join(opt.results_dir,
                                               self.train_log_filename)
        opt.eval_log_filepath  = os.path.join(opt.results_dir,
                                               self.eval_log_filename)
        opt.tensorboard_log_dir = os.path.join(opt.results_dir,
                                                self.tensorboard_log_dir)
        opt.device = torch.device(
            "cuda:%d" % opt.device_ids[0] if opt.device >= 0 else "cpu")
        self.opt = opt
        return opt


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument("--eval_id", type=str)
        self.parser.add_argument("--model_dir", type=str)
        self.parser.add_argument(
            "--tasks", type=str, nargs="+",
            choices=["VCMR", "SVMR", "VR"],
            default=["VCMR", "SVMR", "VR"])
