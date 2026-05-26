import os
import time
import json
import pprint
import random
import numpy as np
from tqdm import tqdm, trange
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from config.config import BaseOptions
from model.demul import DeMUL
from data_loader.dataset import StartEndDataset
from inference import eval_epoch
from optim.adamw import AdamW
from utils.basic_utils import AverageMeter, load_config
from utils.model_utils import count_parameters, move_cuda, start_end_collate

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def set_seed(seed, use_cuda=True):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=False)

def train_epoch(model, train_loader, optimizer, opt, epoch_i, scheduler=None, training=True):
    logger.info("use train_epoch func for training: {}".format(training))
    model.train(mode=training)
    # init meters
    dataloading_time = AverageMeter()
    prepare_inputs_time = AverageMeter()
    model_forward_time = AverageMeter()
    model_backward_time = AverageMeter()
    loss_meters = OrderedDict(
                              visual_ce_loss=AverageMeter(),
                              text_ce_loss=AverageMeter(),
                              fusion_ce_loss=AverageMeter(),
                              loss_overall=AverageMeter())

    num_training_examples = len(train_loader)
    timer_dataloading = time.time()
    for batch_idx, batch in tqdm(enumerate(train_loader),
                                 desc="Training Iteration",
                                 total=num_training_examples):
        global_step = epoch_i * num_training_examples + batch_idx
        dataloading_time.update(time.time() - timer_dataloading)

        timer_start = time.time()
        if opt.device.type == "cuda":
            model_inputs = move_cuda(batch["model_inputs"], opt.device)
        else:
            model_inputs = batch["model_inputs"]

        prepare_inputs_time.update(time.time() - timer_start)

        timer_start = time.time()

        loss, loss_dict = model(model_inputs)
        model_forward_time.update(time.time() - timer_start)
        timer_start = time.time()

        if training:
            optimizer.zero_grad()
            loss.backward()
            if opt.grad_clip != -1:
                total_norm = nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
                if total_norm > opt.grad_clip:
                    print("clipping gradient: {} with coef {}".format(total_norm, opt.grad_clip / total_norm))

            optimizer.step()
            model_backward_time.update(time.time() - timer_start)
            opt.writer.add_scalar("Train/LR_top", float(optimizer.param_groups[0]["lr"]), global_step)
            opt.writer.add_scalar("Train/LR_pretrain", float(optimizer.param_groups[-1]["lr"]), global_step)
            for k, v in loss_dict.items():
                opt.writer.add_scalar("Train/{}".format(k), v, global_step)

        for k, v in loss_dict.items():
            loss_meters[k].update(float(v))

        timer_dataloading = time.time()

    if training:
        to_write = opt.train_log_txt_formatter.format(
            time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
            epoch=epoch_i,
            loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in loss_meters.items()]))
        with open(opt.train_log_filepath, "a") as f:
            f.write(to_write)
        print("Epoch time stats:")
        print("dataloading_time: max {dataloading_time.max} "
              "min {dataloading_time.min} avg {dataloading_time.avg}\n"
              "prepare_inputs_time: max {prepare_inputs_time.max} "
              "min {prepare_inputs_time.min} avg {prepare_inputs_time.avg}\n"
              "model_forward_time: max {model_forward_time.max} "
              "min {model_forward_time.min} avg {model_forward_time.avg}\n"
              "model_backward_time: max {model_backward_time.max} "
              "min {model_backward_time.min} avg {model_backward_time.avg}\n"
              "".format(dataloading_time=dataloading_time, prepare_inputs_time=prepare_inputs_time,
                        model_forward_time=model_forward_time, model_backward_time=model_backward_time))
    else:
        for k, v in loss_meters.items():
            opt.writer.add_scalar("Eval_Loss/{}".format(k), v.avg, epoch_i)

def rm_key_from_odict(odict_obj, rm_suffix):
    """remove key entry from the OrderedDict"""
    return OrderedDict([(k, v) for k, v in odict_obj.items() if rm_suffix not in k])

def build_optimizer(model, opts):
    param_optimizer = [(n, p) for n, p in model.named_parameters()
                       if (n.startswith('encoder') or n.startswith('query_weight')) and p.requires_grad]

    param_top = [(n, p) for n, p in model.named_parameters()
                 if (not n.startswith('encoder') and not n.startswith('query_weight')) and p.requires_grad]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_top
                    if not any(nd in n for nd in no_decay)],
            'weight_decay': opts.wd},
        {'params': [p for n, p in param_top
                    if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
            'lr': opts.lr_mul * opts.lr,
            'weight_decay': opts.wd},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
            'lr': opts.lr_mul * opts.lr,
            'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=opts.lr)
    return optimizer

def train(model, train_dataset, train_eval_dataset, val_dataset, opt):
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        assert len(opt.device_ids) == 1

    g = torch.Generator()
    g.manual_seed(opt.seed)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(train_dataset,
                              collate_fn=start_end_collate,
                              batch_size=opt.bsz,
                              num_workers=opt.num_workers,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              worker_init_fn=seed_worker,
                              generator=g)

    train_eval_loader = DataLoader(train_eval_dataset,
                                   collate_fn=start_end_collate,
                                   batch_size=opt.bsz,
                                   num_workers=opt.num_workers,
                                   shuffle=False,
                                   pin_memory=True,
                                   drop_last=True,
                                   worker_init_fn=seed_worker,
                                   generator=g)

    optimizer = build_optimizer(model, opt)
    prev_best_score = 0.
    es_cnt = 0
    start_epoch = 0 if opt.no_eval_untrained else -1
    eval_tasks_at_training = opt.eval_tasks_at_training
    save_submission_filename = \
        "latest_{}_{}_predictions_{}.json".format(opt.dset_name, opt.eval_split_name, "_".join(eval_tasks_at_training))
    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"):
        if epoch_i >= 0:
            train_epoch(model, train_loader, optimizer, opt, epoch_i, scheduler=None, training=True)

        global_step = (epoch_i + 1) * len(train_loader)
        if not opt.disable_eval:
            if epoch_i % opt.eval_epoch_num == 0 or epoch_i == opt.n_epoch - 1 or epoch_i == start_epoch:
                with torch.no_grad():
                    train_epoch(model, train_eval_loader, optimizer, opt, epoch_i, training=False)
                    metrics_no_nms, metrics_nms, latest_file_paths = \
                        eval_epoch(model, val_dataset, opt, save_submission_filename,
                                   tasks=eval_tasks_at_training, max_after_nms=100)
                to_write = opt.eval_log_txt_formatter.format(
                    time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                    epoch=epoch_i,
                    eval_metrics_str=json.dumps(metrics_no_nms))
                with open(opt.eval_log_filepath, "a") as f:
                    f.write(to_write)
                logger.info("metrics_no_nms {}".format(
                    pprint.pformat(rm_key_from_odict(metrics_no_nms, rm_suffix="by_type"), indent=4)))
                logger.info("metrics_nms {}".format(pprint.pformat(metrics_nms, indent=4)))

                metrics = metrics_no_nms
                for task_type in ["SVMR", "VCMR"]:
                    if task_type in metrics:
                        task_metrics = metrics[task_type]
                        for iou_thd in [0.5, 0.7]:
                            opt.writer.add_scalars("Eval/{}-{}".format(task_type, iou_thd),
                                                   {k: v for k, v in task_metrics.items() if str(iou_thd) in k},
                                                   global_step)

                task_type = "VR"
                if task_type in metrics:
                    task_metrics = metrics[task_type]
                    opt.writer.add_scalars("Eval/{}".format(task_type),
                                           {k: v for k, v in task_metrics.items()},
                                           global_step)

                stop_metric_names = ["r1"] if opt.stop_task == "VR" else ["0.5-r1", "0.7-r1"]
                stop_score = sum([metrics[opt.stop_task][e] for e in stop_metric_names])

                if stop_score > prev_best_score:
                    es_cnt = 0
                    prev_best_score = stop_score

                    checkpoint = {
                        "model": model.state_dict(),
                        "model_cfg": model.config,
                        "epoch": epoch_i}
                    torch.save(checkpoint, opt.ckpt_filepath)

                    best_file_paths = [e.replace("latest", "best") for e in latest_file_paths]
                    for src, tgt in zip(latest_file_paths, best_file_paths):
                        os.renames(src, tgt)
                    logger.info("The checkpoint file has been updated.")
                else:
                    es_cnt += 1
                    if opt.max_es_cnt != -1 and es_cnt > opt.max_es_cnt:
                        with open(opt.train_log_filepath, "a") as f:
                            f.write("Early Stop at epoch {}".format(epoch_i))
                        logger.info("Early stop at {} with {} {}"
                                    .format(epoch_i, " ".join([opt.stop_task] + stop_metric_names), prev_best_score))
                        break
        else:
            checkpoint = {
                "model": model.state_dict(),
                "model_cfg": model.config,
                "epoch": epoch_i}
            torch.save(checkpoint, opt.ckpt_filepath)

        if opt.debug:
            break

    opt.writer.close()

def start_training():
    logger.info("Setup config, data and model...")
    opt = BaseOptions().parse()
    set_seed(opt.seed)

    opt.writer = SummaryWriter(opt.tensorboard_log_dir)
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Metrics] {eval_metrics_str}\n"

    data_config = load_config(opt.dataset_config)

    train_dataset = StartEndDataset(
        config=data_config,
        mode="train",
        data_ratio=opt.data_ratio,
        neg_video_num=opt.neg_video_num,
        max_ctx_len=opt.max_ctx_len,
        use_extend_pool=opt.use_extend_pool,
    )

    if not opt.disable_eval:
        train_eval_dataset = StartEndDataset(
            config=data_config,
            max_ctx_len=opt.max_ctx_len,
            max_desc_len=opt.max_desc_len,
            clip_length=opt.clip_length,
            ctx_mode=opt.ctx_mode,
            mode="val",
            data_ratio=opt.data_ratio,
            neg_video_num=opt.neg_video_num,
            use_extend_pool=opt.use_extend_pool,
        )

        eval_dataset = StartEndDataset(
            config=data_config,
            max_ctx_len=opt.max_ctx_len,
            max_desc_len=opt.max_desc_len,
            clip_length=opt.clip_length,
            ctx_mode=opt.ctx_mode,
            mode=opt.eval_split_name,
            data_ratio=opt.data_ratio,
            is_eval=True,
            inference_top_k=opt.max_vcmr_video,
        )
    else:
        train_eval_dataset = None
        eval_dataset = None

    model_config = load_config(opt.model_config)
    logger.info("model_config {}".format(pprint.pformat(model_config, indent=4)))

    model = DeMUL(
        model_config,
        visual_dim=opt.visual_dim,
        text_dim=opt.text_dim,
        query_dim=opt.query_dim,
        hidden_dim=opt.hidden_dim,
        video_len=opt.max_ctx_len,
        ctx_mode=opt.ctx_mode,
        lw_video_ce=opt.lw_video_ce,
        lw_st_ed=opt.lw_st_ed,
        similarity_measure=opt.similarity_measure,
        use_debug=opt.debug,
    )
    print(model)

    if opt.encoder_pretrain_ckpt_filepath != "None":
        checkpoint = torch.load(
            opt.encoder_pretrain_ckpt_filepath, weights_only=False
        )
        loaded_state_dict = checkpoint["model"]
        model.load_state_dict(loaded_state_dict, strict=False)

    count_parameters(model)

    logger.info("Start Training...")
    train(model, train_dataset, train_eval_dataset, eval_dataset, opt)
    return opt.results_dir, opt.eval_split_name, opt.debug

if __name__ == '__main__':
    model_dir, eval_split_name, debug = start_training()
