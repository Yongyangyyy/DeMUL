"""
Dataset for DeMUL training and evaluation.

Each sample provides:
  - query features (BERT)
  - visual features (ResNet + SlowFast, concatenated → 4352-d)
  - subtitle features (RoBERTa → 768-d)
  - start/end label indices (training only)

The dataset uses:
  - A first-stage VR ranklist (LMDB) to select candidate videos.
  - LMDB stores for query, visual and subtitle features.
"""

import io
import os
import math
import json
import random
import logging

import lmdb
import msgpack
import msgpack_numpy
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.basic_utils import load_jsonl, load_json, l2_normalize_np_array

logger = logging.getLogger(__name__)

class StartEndDataset(Dataset):
    """Video-query dataset for moment retrieval.

    Args:
        config           : EasyDict dataset config
        mode             : "train" | "val" | "test"
        max_ctx_len      : max number of video clips
        max_desc_len     : max number of query tokens
        clip_length      : duration of each clip in seconds
        ctx_mode         : modality string, e.g. "visual_sub"
        is_eval          : if True, use inference-mode sampling
        neg_video_num    : number of negative videos per sample (train)
        data_ratio       : fraction of dataset to use (debug)
        use_extend_pool  : how far past the GT video rank to sample negatives
        inference_top_k  : top-k videos to retrieve during eval
    """

    def __init__(self, config, mode="train",
                 max_ctx_len=100, max_desc_len=30, clip_length=1.5,
                 ctx_mode="visual_sub", is_eval=False,
                 neg_video_num=3, data_ratio=1.0,
                 use_extend_pool=1000, inference_top_k=10):

        self.dset_name = config.dset_name
        self.root_path = config.root_path

        self.desc_bert_path = os.path.join(self.root_path, config.desc_bert_path)
        self.vid_feat_path  = os.path.join(self.root_path, config.vid_feat_path)

        self.ctx_mode = ctx_mode
        self.use_sub  = "sub" in ctx_mode
        if self.use_sub:
            self.sub_bert_path = os.path.join(self.root_path,
                                              config.sub_bert_path)

        self.max_ctx_len  = max_ctx_len
        self.max_desc_len = max_desc_len
        self.clip_length  = clip_length
        self.neg_video_num = neg_video_num
        self.is_eval = is_eval
        self.mode    = mode

        # data paths by split
        split_data = {
            "train": config.train_data_path,
            "val":   config.eval_data_path,
            "test":  config.test_data_path,
        }
        split_vr = {
            "train": config.train_first_VR_ranklist_path,
            "val":   config.eval_first_VR_ranklist_path_hero,
            "test":  config.test_public_first_VR_ranklist_path_hero,
        }
        self.data_path = os.path.join(self.root_path, split_data[mode])
        self.first_VR_ranklist_path = os.path.join(
            self.root_path, split_vr[mode])

        # load query annotations
        self.query_data = load_jsonl(self.data_path)
        if data_ratio != 1.0:
            n = int(len(self.query_data) * data_ratio)
            self.query_data = self.query_data[:n]
            logger.info("Using %.1f%% data: %d examples",
                        data_ratio * 100, n)

        # LMDB handles
        def _open_lmdb(path):
            return lmdb.open(path, readonly=True, create=False,
                             max_readers=4096 * 8, readahead=False)

        self.vr_env   = _open_lmdb(self.first_VR_ranklist_path)
        self.vr_txn   = self.vr_env.begin(buffers=True)
        self.desc_env = _open_lmdb(self.desc_bert_path)
        self.desc_txn = self.desc_env.begin(buffers=True)
        self.vid_env  = _open_lmdb(self.vid_feat_path)
        self.vid_txn  = self.vid_env.begin(buffers=True)
        if self.use_sub:
            self.sub_env = _open_lmdb(self.sub_bert_path)
            self.sub_txn = self.sub_env.begin(buffers=True)

        # video index maps
        video_data = load_json(
            os.path.join(self.root_path,
                         config.video_duration_idx_path))[mode]
        self.video_data  = [{"vid_name": k, "duration": v[0]}
                            for k, v in video_data.items()]
        self.video2idx   = {k: v[1] for k, v in video_data.items()}
        self.idx2video   = {v[1]: k for k, v in video_data.items()}

        self.use_extend_pool  = use_extend_pool
        self.inference_top_k  = inference_top_k
        self.normalize_vfeat  = True
        self.normalize_tfeat  = True
        self.visual_token_id  = 0
        self.text_token_id    = 1

    def __len__(self):
        return len(self.query_data)

    # ── feature helpers ──────────────────────────────────────────────────────

    def _pad_feature(self, feature, max_len):
        """Pad (N_clip, D) to (max_len, D) and return (feat_pad, feat_mask)."""
        N, D = feature.shape
        feat_pad  = torch.zeros(max_len, D)
        feat_mask = torch.zeros(max_len, dtype=torch.long)
        feat_pad[:N]  = torch.from_numpy(feature)
        feat_mask[:N] = 1
        return feat_pad, feat_mask

    def _get_query_feat(self, desc_id):
        dump = self.desc_txn.get(str(desc_id).encode())
        with io.BytesIO(dump) as reader:
            feat = np.load(reader, allow_pickle=True)['features'][:self.max_desc_len]
        if self.normalize_tfeat:
            feat = l2_normalize_np_array(feat)
        feat_pad, feat_mask = self._pad_feature(feat, self.max_desc_len)
        return {
            "feat":         feat_pad,
            "feat_mask":    feat_mask,
            "feat_pos_id":  torch.arange(self.max_desc_len, dtype=torch.long),
            "feat_token_id": torch.full((self.max_desc_len,),
                                        self.text_token_id, dtype=torch.long),
        }

    def _get_visual_feat(self, vid_name):
        dump = self.vid_txn.get(vid_name.encode())
        feat = {k: np.copy(v)
                for k, v in msgpack_numpy.loads(dump, raw=False).items()
                }['features'][:self.max_ctx_len]
        if self.normalize_vfeat:
            feat = l2_normalize_np_array(feat)
        return feat

    def _get_sub_feat(self, vid_name):
        dump = self.sub_txn.get(vid_name.encode())
        with io.BytesIO(dump) as reader:
            feat = np.load(reader, allow_pickle=True
                           )['features'][:self.max_ctx_len]
        if self.normalize_tfeat:
            feat = l2_normalize_np_array(feat)
        return feat

    # ── label helpers ────────────────────────────────────────────────────────

    def _get_st_ed_label(self, ts, max_idx, total_length=None):
        """Convert [start_sec, end_sec] → clip indices + offset sequence."""
        st_ = min(ts[0] / self.clip_length, max_idx)
        ed_ = min(ts[1] / self.clip_length, max_idx)
        st_idx = max(0, min(math.floor(ts[0] / self.clip_length), max_idx))
        ed_idx = max(st_idx, min(math.floor(ts[1] / self.clip_length), max_idx))
        assert 0 <= st_idx <= ed_idx <= max_idx

        if total_length is None:
            total_length = max_idx + 1
        indices       = torch.arange(total_length, dtype=torch.float32)
        offset_sequence = torch.stack([st_ - indices, ed_ - indices], dim=1)

        return (torch.LongTensor([st_idx, ed_idx]),
                torch.FloatTensor([st_, ed_]),
                offset_sequence)

    # ── __getitem__ ──────────────────────────────────────────────────────────

    def __getitem__(self, index):
        raw = self.query_data[index]
        if self.dset_name == "tvr":
            meta = dict(
                desc_id=raw["desc_id"],
                desc=raw["desc"],
                vid_name=raw["vid_name"] if "vid_name" in raw else None,
                ts=raw["ts"] if "ts" in raw else None,
            )
        elif self.dset_name == "didemo":
            meta = dict(
                desc_id=raw["desc_id"],
                desc=raw["desc"],
                vid_name=raw["vid_name"] if "vid_name" in raw else None,
                ts=raw["target"] if "target" in raw else None,
            )
        else:
            meta = dict(
                desc_id=raw["desc_id"],
                desc=raw["desc"],
                vid_name=raw.get("vid_name"),
                ts=raw.get("ts"),
            )

        model_inputs = {}
        model_inputs["query"] = self._get_query_feat(meta["desc_id"])

        # determine candidate video list
        vr_res = msgpack.loads(
            self.vr_txn.get(str(meta["desc_id"]).encode()))

        if not self.is_eval:
            # find GT video rank
            location = 100
            for idx, item in enumerate(vr_res):
                if meta["vid_name"] == self.idx2video[item[0]]:
                    location = idx
                    break
            if self.mode == "train":
                assert 0 <= location < 100

            neg_pool = [self.idx2video[item[0]] for item in vr_res
                        if meta["vid_name"] != self.idx2video[item[0]]]
            sampled_negatives = random.sample(
                neg_pool[:location + self.use_extend_pool],
                k=self.neg_video_num)
            total_vid_list = [meta["vid_name"]] + sampled_negatives
            shared_video_num = 1 + self.neg_video_num
        else:
            inf_list   = [self.idx2video[item[0]]
                          for item in vr_res[:self.inference_top_k]]
            inf_scores = [item[1] for item in vr_res[:self.inference_top_k]]
            model_inputs["inference_vr_scores"] = torch.FloatTensor(inf_scores)
            total_vid_list = [meta["vid_name"]] + inf_list
            shared_video_num = 1 + self.inference_top_k

        meta["sample_vid_name_list"] = total_vid_list[1:]

        # ── visual features ──────────────────────────────────────────────────
        ref_visual = self._get_visual_feat(meta["vid_name"])
        ctx_l, vis_dim = ref_visual.shape

        vis_feat_pad  = torch.zeros(shared_video_num, self.max_ctx_len, vis_dim)
        vis_feat_mask = torch.zeros(shared_video_num, self.max_ctx_len,
                                    dtype=torch.long)
        for i, vname in enumerate(total_vid_list):
            fp, fm = self._pad_feature(
                self._get_visual_feat(vname), self.max_ctx_len)
            vis_feat_pad[i]  = fp
            vis_feat_mask[i] = fm

        model_inputs["visual"] = {
            "feat":         vis_feat_pad,
            "feat_mask":    vis_feat_mask,
            "feat_pos_id":  torch.arange(
                self.max_ctx_len, dtype=torch.long
            ).unsqueeze(0).expand(shared_video_num, -1),
            "feat_token_id": torch.full(
                (shared_video_num, self.max_ctx_len),
                self.visual_token_id, dtype=torch.long),
        }

        # ── subtitle features ────────────────────────────────────────────────
        ref_sub = self._get_sub_feat(meta["vid_name"])
        _, sub_dim = ref_sub.shape

        sub_feat_pad  = torch.zeros(shared_video_num, self.max_ctx_len, sub_dim)
        sub_feat_mask = torch.zeros(shared_video_num, self.max_ctx_len,
                                    dtype=torch.long)
        for i, vname in enumerate(total_vid_list):
            fp, fm = self._pad_feature(
                self._get_sub_feat(vname), self.max_ctx_len)
            sub_feat_pad[i]  = fp
            sub_feat_mask[i] = fm

        model_inputs["sub"] = {
            "feat":         sub_feat_pad,
            "feat_mask":    sub_feat_mask,
            "feat_pos_id":  torch.arange(
                self.max_ctx_len, dtype=torch.long
            ).unsqueeze(0).expand(shared_video_num, -1),
            "feat_token_id": torch.full(
                (shared_video_num, self.max_ctx_len),
                self.text_token_id, dtype=torch.long),
        }

        # ── ground-truth labels (training only) ──────────────────────────────
        if not self.is_eval:
            (model_inputs["st_ed_indices"],
             model_inputs["st_ed"],
             model_inputs["offset_sequence"]) = self._get_st_ed_label(
                meta["ts"], max_idx=ctx_l - 1,
                total_length=self.max_ctx_len)

        return {"meta": meta, "model_inputs": model_inputs}
