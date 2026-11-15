import argparse
import datetime
import json
import logging
import math
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/SLURM/home/slurm_s202281480/.cache/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/SLURM/home/slurm_s202281480/.cache/huggingface/hub"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"
import pathlib
import pickle
import random
import shutil
import time
import warnings
from collections import OrderedDict

import accelerate
import matplotlib.pyplot as plt
import numpy as np
import torch

# torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.nn.functional as func
import torchvision.transforms as transforms
from accelerate.logging import get_logger
from diffusers.configuration_utils import FrozenDict
from diffusers.optimization import get_scheduler
from einops import rearrange
from fastdtw import fastdtw
from omegaconf import OmegaConf
from sentence_transformers import SentenceTransformer, util
from torch.utils.data import DataLoader
from tqdm import tqdm

from decoded_fsq_metrics import (
    ground_truth_frames,
    load_fsq_decoder,
    normalized_decoded_dtw,
    predicted_frames,
)
from models.multihead_t2vqpgpt import Text2VQPoseGPT
from signdatasets import VQSignTextDataset
from utils import get_num_params, seed_everything

warnings.filterwarnings("ignore")
logger = get_logger(__name__, log_level="INFO")


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train/stage_1.yaml")
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--workspace_dir",
        type=str,
        default=None,
        help="Use an explicit workspace directory instead of a timestamped one.",
    )
    args = parser.parse_args()

    # Load the configuration file
    cfg = OmegaConf.load(args.config)
    cfg.exp_name = pathlib.Path(args.config).stem

    # Update cfg with values from args
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    args_dict = vars(args)
    merged_dict = {**cfg_dict, **args_dict}

    # Convert merged_dict back to OmegaConf object
    merged_cfg = OmegaConf.create(merged_dict)

    return merged_cfg


def load_modules(cfg, device, weight_dtype):
    modules_cfg = cfg.modules

    text_model = SentenceTransformer(
        modules_cfg.text_model,
        cache_folder="/deepo_data/signvipworkspace/models",
        local_files_only=True,
    )
    text_model = text_model.to(device, weight_dtype)
    text_model.eval()
    tokenizer = text_model.tokenizer
    logger.info(f"Loaded text model from {modules_cfg.text_model}.")

    t2pgpt = Text2VQPoseGPT(
        num_vq=modules_cfg.codebook_size + 2,
        embed_dim=modules_cfg.embed_dim,
        clip_dim=modules_cfg.clip_dim,
        block_size=modules_cfg.block_size,
        num_layers=modules_cfg.num_layers,
        n_head=modules_cfg.n_head,
        drop_out_rate=modules_cfg.drop_out_rate,
        fc_rate=modules_cfg.fc_rate,
        attn_type=modules_cfg.get("attn_type", "global"),
        pose_size=cfg.pose_size,
        head_layers=modules_cfg.head_layers,
        head_type=modules_cfg.get("head_type", "default"),
    )

    if modules_cfg.ckpt is not None:
        state_dict = torch.load(modules_cfg.ckpt, map_location="cpu")
        if all(k.startswith("module.") for k in state_dict):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        elif any(k.startswith("module.") for k in state_dict):
            raise RuntimeError("Checkpoint mixes prefixed and unprefixed keys.")
        t2pgpt.load_state_dict(state_dict, strict=True)
        logger.info(
            f"Strictly loaded all {len(state_dict)} tensors from {modules_cfg.ckpt}."
        )

    t2pgpt = t2pgpt.to(device)
    return (
        tokenizer,
        text_model,
        t2pgpt,
    )


def count_params(model: Text2VQPoseGPT):
    logger.info("***** Parameters Counting *****")

    # Text2VQPoseGPT
    params = list(model.parameters())
    num_params, num_trainable_params = get_num_params(params)
    logger.info(
        f"  Text2VQPoseGPT: {num_params:.3f} M (trainable: {num_trainable_params:.3f}M)"
    )


def save_model(model, cfg, model_path, is_main_process, weight_dtype):
    if is_main_process:
        pathlib.Path(os.path.join(model_path, "t2pgpt")).mkdir(
            parents=True, exist_ok=True
        )
        torch.save(
            model.state_dict(),
            os.path.join(model_path, "t2pgpt/model.bin"),
        )
        logger.info(f"Saved t2pgpt to {os.path.join(model_path, 't2pgpt')}.")


@torch.no_grad()
def log_valid(
    cfg,
    valid_dataloader,
    accelerator,
    model,
    text_encoder,
    tokenizer,
    device,
    weight_dtype,
    global_step,
    out_path,
    fsq_decoder=None,
):
    model.eval()

    valid_acc = []
    dtw_scores = []
    raw_dtw_scores = []
    teacher_loss_sum = 0.0
    teacher_correct = 0
    teacher_tokens = 0
    predicted_token_sets = []
    target_token_sets = []
    length_ratios = []
    invalid_predictions = 0
    one_frame_predictions = 0
    static_predictions = 0
    changed_token_count = 0
    temporal_token_count = 0
    premature_special_heads = 0
    evaluated_heads = 0
    metric_cfg = cfg.get("validation_metric", {})
    use_decoded_dtw = fsq_decoder is not None
    max_samples = int(metric_cfg.get("max_samples", 0))
    if max_samples:
        max_local_samples = math.ceil(max_samples / accelerator.num_processes)
    else:
        max_local_samples = None
    local_samples = 0

    for step, batch in tqdm(
        enumerate(valid_dataloader), total=len(valid_dataloader), desc="Validating"
    ):
        if max_local_samples is not None and local_samples >= max_local_samples:
            break
        batch_size = len(batch["pose_latents"])
        pose_latents = batch["pose_latents"].to(device)
        pose_len = batch["pose_len"].to(device)
        text_input_ids = batch["text_input_ids"].to(device)
        text_attn_mask = batch["text_attn_mask"].to(device)
        paths = batch["path"]
        actual_mask = torch.zeros_like(pose_latents, device=device)
        for i in range(pose_latents.shape[0]):
            actual_mask[i, : pose_len[i]] = 1

        text_embeds = text_encoder(
            {"input_ids": text_input_ids, "attention_mask": text_attn_mask}
        )
        text_embeds = text_embeds["sentence_embedding"].to(weight_dtype)

        teacher_input = pose_latents[:, : -cfg.pose_size]
        teacher_logits = model(teacher_input, text_embeds)
        teacher_loss_sum += func.cross_entropy(
            rearrange(teacher_logits, "b l c -> b c l"),
            pose_latents,
            ignore_index=cfg.codebook_size + 1,
            reduction="sum",
        ).item()
        teacher_predictions = teacher_logits.argmax(dim=-1)
        teacher_mask = pose_latents != cfg.codebook_size + 1
        teacher_correct += int(
            ((teacher_predictions == pose_latents) & teacher_mask).sum().item()
        )
        teacher_tokens += int(teacher_mask.sum().item())
        del teacher_input, teacher_logits, teacher_predictions, teacher_mask

        pred_idx = accelerator.unwrap_model(model).sample(
            text_embeds, cfg.max_pose_len + 1
        )  # [bs, max_pose_len, pose_dim]
        if not use_decoded_dtw:
            pred_idx, pose_latents, pose_len = accelerator.gather_for_metrics(
                (pred_idx, pose_latents, pose_len)
            )
            pred_idx = pred_idx.cpu()
            pose_latents = pose_latents.cpu()

        remaining = batch_size
        if max_local_samples is not None:
            remaining = min(remaining, max_local_samples - local_samples)
        for i in range(remaining):
            pred = pred_idx[i, : pose_len[i]]
            gt = pose_latents[i, : pose_len[i]]
            valid_acc.append((pred == gt).sum().item() / pose_len[i].item())
            raw_distance, _ = fastdtw(
                np.asarray(pred.detach().cpu()), np.asarray(gt.detach().cpu())
            )
            raw_dtw_scores.append(raw_distance / max(len(pred), len(gt)))
            if use_decoded_dtw:
                gt_frames = ground_truth_frames(
                    pose_latents[i], pose_len[i], cfg.pose_size, cfg.codebook_size
                )
                pred_frames = predicted_frames(
                    pred_idx[i], cfg.pose_size, cfg.codebook_size
                )
                target_token_sets.append(gt_frames.detach().cpu())
                predicted_token_sets.append(pred_frames.detach().cpu())
                length_ratios.append(len(pred_frames) / max(len(gt_frames), 1))
                if len(pred_frames) <= 1:
                    one_frame_predictions += 1
                if len(pred_frames) > 1:
                    changed = (pred_frames[1:] != pred_frames[:-1]).sum().item()
                    comparisons = pred_frames[1:].numel()
                    changed_token_count += changed
                    temporal_token_count += comparisons
                    static_predictions += changed / comparisons < 0.01
                else:
                    static_predictions += 1

                raw_frames = pred_idx[i].reshape(-1, cfg.pose_size)
                for head_index in range(cfg.pose_size):
                    special = torch.nonzero(
                        raw_frames[:, head_index] >= cfg.codebook_size,
                        as_tuple=False,
                    )
                    first_special = int(special[0]) if len(special) else len(raw_frames)
                    premature_special_heads += first_special < max(len(gt_frames) // 2, 1)
                    evaluated_heads += 1
                if len(pred_frames) == 0:
                    invalid_predictions += 1
                    dtw_scores.append(float(metric_cfg.get("failure_penalty", 10.0)))
                else:
                    dtw_scores.append(
                        normalized_decoded_dtw(
                            fsq_decoder,
                            pred_frames,
                            gt_frames,
                            pool_size=metric_cfg.get("pool_size", [4, 4]),
                            chunk_size=int(metric_cfg.get("decode_chunk_size", 32)),
                        )
                    )
            else:
                distance, _ = fastdtw(np.array(pred.cpu()), np.array(gt.cpu()))
                dtw_scores.append(distance / max(len(pred), len(gt)))
        local_samples += remaining

        if step == 0 and accelerator.is_main_process:
            paths = [
                os.path.join(out_path, f"step-{global_step}", p).replace(".mp4", ".pkl")
                for p in paths
            ][:20]
            os.makedirs(os.path.dirname(paths[0]), exist_ok=True)
            # save 20 samples
            for i in range(min(20, len(paths), pred_idx.shape[0])):
                save_path = paths[i]
                with open(save_path, "wb") as f:
                    pickle.dump(
                        {
                            "gt": pose_latents[i].cpu().numpy(),
                            "pred": pred_idx[i].cpu().numpy(),
                            "len": pose_len[i].cpu().numpy(),
                        },
                        f,
                    )

    if use_decoded_dtw:
        local_totals = torch.tensor(
            [
                sum(valid_acc),
                sum(dtw_scores),
                sum(length_ratios),
                invalid_predictions,
                one_frame_predictions,
                static_predictions,
                changed_token_count,
                temporal_token_count,
                premature_special_heads,
                evaluated_heads,
                len(dtw_scores),
                sum(raw_dtw_scores),
                teacher_loss_sum,
                teacher_correct,
                teacher_tokens,
            ],
            device=device,
            dtype=torch.float64,
        )
        totals = accelerator.reduce(local_totals, reduction="sum")
        sample_count = totals[10].clamp_min(1)
        predicted_tokens = (
            torch.cat(predicted_token_sets).flatten()
            if predicted_token_sets
            else torch.empty(0)
        )
        target_tokens = (
            torch.cat(target_token_sets).flatten()
            if target_token_sets
            else torch.empty(0)
        )
        predicted_counts = torch.bincount(
            predicted_tokens.long(), minlength=cfg.codebook_size
        ).to(device=device, dtype=torch.float64)
        target_counts = torch.bincount(
            target_tokens.long(), minlength=cfg.codebook_size
        ).to(device=device, dtype=torch.float64)
        predicted_counts = accelerator.reduce(predicted_counts, reduction="sum")
        target_counts = accelerator.reduce(target_counts, reduction="sum")

        def distribution_metrics(counts):
            probabilities = counts[counts > 0] / counts.sum().clamp_min(1)
            perplexity = torch.exp(-(probabilities * probabilities.log()).sum())
            return int((counts > 0).sum()), float(perplexity)

        pred_active, pred_perplexity = distribution_metrics(predicted_counts)
        gt_active, gt_perplexity = distribution_metrics(target_counts)
        mean_dtw = float(totals[1] / sample_count)
        log_dict = {
            "teacher_forced_loss": float(totals[12] / totals[14].clamp_min(1)),
            "teacher_forced_accuracy": float(totals[13] / totals[14].clamp_min(1)),
            "rollout_token_accuracy": float(totals[0] / sample_count),
            "valid_acc": float(totals[0] / sample_count),
            "raw_token_dtw_score": float(totals[11] / sample_count),
            "decoded_dtw_score": mean_dtw,
            "predicted_gt_length_ratio": float(totals[2] / sample_count),
            "invalid_prediction_rate": float(totals[3] / sample_count),
            "one_frame_prediction_rate": float(totals[4] / sample_count),
            "static_prediction_rate": float(totals[5] / sample_count),
            "temporal_token_change_rate": float(totals[6] / totals[7].clamp_min(1)),
            "premature_special_head_rate": float(totals[8] / totals[9].clamp_min(1)),
            "predicted_active_codes": pred_active,
            "predicted_code_perplexity": pred_perplexity,
            "target_active_codes": gt_active,
            "target_code_perplexity": gt_perplexity,
            "validation_samples": int(totals[10]),
        }
        dtw_scores = [mean_dtw]
    else:
        log_dict = {
            "valid_acc": np.mean(valid_acc),
            "dtw_score": np.mean(dtw_scores),
        }
    if accelerator.is_main_process:
        history_entry = {"step": global_step, **log_dict}
        history_path = os.path.join(os.path.dirname(out_path), "validation_history.jsonl")
        with open(history_path, "a", encoding="utf-8") as history_file:
            history_file.write(json.dumps(history_entry, sort_keys=True) + "\n")
    if cfg.report_to == "wandb":
        if accelerator.is_main_process:
            import wandb

            wandb.log(log_dict, step=global_step)
    else:
        accelerator.log(log_dict, step=global_step)
    return valid_acc, dtw_scores, log_dict


def main():
    cfg = parse_config()
    experiment_index = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    workspace_dir = cfg.workspace_dir or f"./workspace/{cfg.exp_name}/{experiment_index}"

    if cfg.resume_from_checkpoint:
        workspace_dir = cfg.resume_from_checkpoint

    pathlib.Path(workspace_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(workspace_dir + "/log").mkdir(parents=True, exist_ok=True)

    config = accelerate.utils.ProjectConfiguration(
        project_dir=".", logging_dir=workspace_dir + "/log"
    )
    # save config under workspace_dir
    with open(os.path.join(workspace_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)

    accelerator = accelerate.Accelerator(
        log_with=None if cfg.report_to == "wandb" else cfg.report_to,
        project_config=config,
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
    )
    if cfg.report_to == "wandb":
        if accelerator.is_main_process:
            import wandb

            wandb.init(project="signvip", name=cfg.exp_name, dir=workspace_dir)
    else:
        accelerator.init_trackers("signvip")
    device = accelerator.local_process_index

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if cfg.seed is not None:
        seed_everything(cfg.seed)

    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    elif cfg.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training!"
        )

    (
        tokenizer,
        text_encoder,
        t2pgpt,
    ) = load_modules(cfg, device, weight_dtype)

    fsq_decoder = None
    metric_cfg = cfg.get("validation_metric", {})
    if metric_cfg.get("type") == "decoded_fsq_dtw":
        fsq_decoder = load_fsq_decoder(
            checkpoint=metric_cfg.checkpoint,
            model_kwargs=OmegaConf.to_container(metric_cfg.model_kwargs, resolve=True),
            device=device,
            dtype=torch.float32,
        )
        logger.info(
            "Loaded FSQ decoder for Appendix G validation from "
            f"{metric_cfg.checkpoint}."
        )

    text_encoder.requires_grad_(False)
    model = t2pgpt
    model = model.to(device)

    if cfg.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    if cfg.solver.optimizer.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam by running `pip install bitsandbytes`."
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optimizer_cls(
        trainable_params,
        lr=cfg.solver.optimizer.learning_rate,
        betas=(cfg.solver.optimizer.adam_beta1, cfg.solver.optimizer.adam_beta2),
        weight_decay=cfg.solver.optimizer.adam_weight_decay,
        eps=cfg.solver.optimizer.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler.name,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_scheduler.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps
        * accelerator.num_processes,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps
        * accelerator.num_processes,
    )

    dataset = VQSignTextDataset(
        frame_size=cfg.dataset.frame_size,
        frame_scale=cfg.dataset.frame_scale,
        sample_rate=cfg.dataset.sample_rate,
        tokenizer=tokenizer,
        max_pose_len=cfg.max_pose_len,
        pose_size=cfg.pose_size,
        codebook_size=cfg.codebook_size,
        roots=cfg.dataset.roots,
        sk_roots=cfg.dataset.sk_roots,
        hamer_roots=cfg.dataset.hamer_roots,
        pose_roots=cfg.dataset.pose_roots,
        meta_paths=cfg.dataset.meta_paths,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=True,
        num_workers=cfg.dataloader.num_workers,
    )
    valid_dataset = VQSignTextDataset(
        frame_size=cfg.dataset.frame_size,
        frame_scale=cfg.dataset.frame_scale,
        sample_rate=cfg.dataset.sample_rate,
        tokenizer=tokenizer,
        max_pose_len=cfg.max_pose_len,
        pose_size=cfg.pose_size,
        codebook_size=cfg.codebook_size,
        roots=cfg.dataset.roots,
        pose_roots=getattr(cfg.validation_data, "pose_roots", cfg.dataset.pose_roots),
        meta_paths=cfg.validation_data.meta_paths,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=int(cfg.get("validation_batch_size", 64)),
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
    )

    count_params(model)

    (model, optimizer, dataloader, valid_dataloader, lr_scheduler) = (
        accelerator.prepare(
            model, optimizer, dataloader, valid_dataloader, lr_scheduler
        )
    )

    num_steps_per_epoch = math.ceil(
        len(dataloader) / cfg.solver.gradient_accumulation_steps
    )
    num_epochs = math.ceil(cfg.solver.max_train_steps / num_steps_per_epoch)

    total_batch_size = (
        cfg.dataloader.batch_size
        * accelerator.num_processes
        * cfg.solver.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.dataloader.batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {cfg.solver.max_train_steps}")

    global_step = 0
    start_epoch = 0
    # log_valid(
    #     cfg,
    #     valid_dataloader,
    #     accelerator,
    #     model,
    #     text_encoder,
    #     tokenizer,
    #     device,
    #     weight_dtype,
    #     global_step,
    #     workspace_dir + "/log",
    # )

    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            resume_dir = cfg.resume_from_checkpoint
        else:
            resume_dir = workspace_dir

        dirs = os.listdir(resume_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1]
        accelerator.load_state(os.path.join(resume_dir, path))
        accelerator.print(f"Resuming from checkpoint {path}")
        global_step = int(path.split("-")[1])

        start_epoch = global_step // num_steps_per_epoch

    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description(f"Exp: {cfg.exp_name}")
    best_train_dtw_score = float("inf")
    for epoch in range(start_epoch, num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        t_data_start = time.time()
        for step, batch in enumerate(dataloader):
            t_data = time.time() - t_data_start
            model.train()
            with accelerator.accumulate(model):
                pose_latents = batch["pose_latents"].to(device)
                pose_len = batch["pose_len"].to(device)
                text_input_ids = batch["text_input_ids"].to(device)
                text_attn_mask = batch["text_attn_mask"].to(device)

                # get the mask for the training using the pose_len
                train_mask = torch.zeros_like(pose_latents, device=device)
                for i in range(pose_latents.shape[0]):
                    train_mask[i, : pose_len[i]] = 1

                target = pose_latents
                input_poses = target[
                    :, : -cfg.pose_size
                ]  # [bs, max_pose_len-pose_size]
                mask = torch.bernoulli(
                    cfg.p_keep
                    * torch.ones(input_poses.shape, device=input_poses.device)
                )
                mask = mask.round().to(dtype=torch.int64)
                r_indices = torch.randint_like(input_poses, cfg.codebook_size)
                input_poses = mask * input_poses + (1 - mask) * r_indices

                with torch.no_grad():
                    text_embeds = text_encoder(
                        {"input_ids": text_input_ids, "attention_mask": text_attn_mask}
                    )
                    text_embeds = text_embeds["sentence_embedding"]
                pred = model(input_poses, text_embeds)
                pred = rearrange(pred, "b l c -> b c l")

                # loss = func.mse_loss(pred.float(), target.float(), reduction="none")
                loss = func.cross_entropy(
                    pred, target, ignore_index=cfg.codebook_size + 1
                )

                pred = rearrange(pred, "b c l -> b l c")
                sampleds = pred.argmax(dim=-1)
                train_acc_item = torch.tensor(0.0, device=device)
                for i in range(pose_latents.shape[0]):
                    sampled = sampleds[i, : pose_len[i]]
                    gt = pose_latents[i, : pose_len[i]]
                    train_acc_item += (sampled == gt).sum() / pose_len[i]

                train_acc_item = train_acc_item / pose_latents.shape[0]

                total_loss = loss

                avg_loss = accelerator.gather(
                    total_loss.repeat(cfg.dataloader.batch_size)
                ).mean()
                avg_acc = accelerator.gather(
                    train_acc_item.repeat(cfg.dataloader.batch_size)
                ).mean()

                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps
                train_acc += avg_acc.item() / cfg.solver.gradient_accumulation_steps

                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_params, cfg.solver.max_grad_norm
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                log_dict = {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                if cfg.report_to == "wandb":
                    if accelerator.is_main_process:
                        import wandb

                        wandb.log(log_dict, step=global_step)
                else:
                    accelerator.log(log_dict, step=global_step)
                train_loss = 0.0
                train_acc = 0.0
                if global_step % cfg.valid_steps == 0 and global_step > 0:
                    valid_acc, dtw_scores, validation_metrics = log_valid(
                        cfg,
                        valid_dataloader,
                        accelerator,
                        model,
                        text_encoder,
                        tokenizer,
                        device,
                        weight_dtype,
                        global_step,
                        workspace_dir + "/log",
                        fsq_decoder=fsq_decoder,
                    )
                    if (
                        accelerator.is_main_process
                        and np.mean(dtw_scores) < best_train_dtw_score
                    ):
                        best_train_dtw_score = np.mean(dtw_scores)
                        best_prefix = (
                            "best-decoded-dtw-"
                            if fsq_decoder is not None
                            else "best-dtw-"
                        )

                        # Remove old best model if exists and only if new score is better
                        for old_dir in os.listdir(workspace_dir):
                            if old_dir.startswith(best_prefix):
                                try:
                                    old_score = float(old_dir.split("-")[-1])
                                    if best_train_dtw_score < old_score:
                                        old_path = os.path.join(workspace_dir, old_dir)
                                        if os.path.isdir(old_path):
                                            shutil.rmtree(old_path)
                                            logger.info(
                                                f"Removed old best model ({old_score:.4f}) with new best score ({best_train_dtw_score:.4f})"
                                            )
                                except Exception as e:
                                    logger.warning(
                                        f"Could not parse DTW score from directory name: {old_dir}, error: {e}"
                                    )

                        # Save new best model with DTW score in path
                        save_model(
                            model,
                            cfg,
                            os.path.join(
                                workspace_dir,
                                (
                                    f"best-decoded-dtw-{best_train_dtw_score:.6f}"
                                    if fsq_decoder is not None
                                    else f"best-dtw-{best_train_dtw_score:.4f}"
                                ),
                            ),
                            accelerator.is_main_process,
                            weight_dtype,
                        )
            logs = {
                "step_loss": total_loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "td": f"{t_data:.2f}s",
            }
            t_data_start = time.time()
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.solver.max_train_steps:
                break

            if global_step % cfg.checkpointing_steps == 0:
                if accelerator.is_main_process:
                    # Save DiT checkpoint:
                    if cfg.max_ckpt is not None:
                        checkpoints = os.listdir(workspace_dir)
                        checkpoints = [
                            d for d in checkpoints if d.startswith("checkpoint")
                        ]
                        checkpoints = sorted(
                            checkpoints, key=lambda x: int(x.split("-")[1])
                        )

                        # before we save the new checkpoint, we need to have at _most_ `max_ckpt - 1` checkpoints
                        if len(checkpoints) >= cfg.max_ckpt:
                            num_to_remove = len(checkpoints) - cfg.max_ckpt + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(
                                f"removing checkpoints: {', '.join(removing_checkpoints)}"
                            )

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(
                                    workspace_dir, removing_checkpoint
                                )
                                shutil.rmtree(removing_checkpoint)

                    if cfg.max_model is not None:
                        models = os.listdir(workspace_dir)
                        models = [d for d in models if d.startswith("model")]
                        models = sorted(models, key=lambda x: int(x.split("-")[1]))

                        # before we save the new model, we need to have at most `max_model - 1` models
                        if len(models) >= cfg.max_model:
                            num_to_remove = len(models) - cfg.max_model + 1
                            removing_models = models[0:num_to_remove]

                            logger.info(
                                f"{len(models)} models already exist, removing {len(removing_models)} models"
                            )
                            logger.info(
                                f"removing models: {', '.join(removing_models)}"
                            )

                            for removing_model in removing_models:
                                removing_model_path = os.path.join(
                                    workspace_dir, removing_model
                                )
                                shutil.rmtree(removing_model_path)

                ckpt_path = os.path.join(workspace_dir, f"checkpoint-{global_step}")
                model_path = os.path.join(workspace_dir, f"model-{global_step}")

                accelerator.save_state(ckpt_path)
                save_model(
                    model,
                    cfg,
                    model_path,
                    accelerator.is_main_process,
                    weight_dtype,
                )

    ckpt_path = os.path.join(workspace_dir, f"checkpoint-{global_step}")
    model_path = os.path.join(workspace_dir, f"model-{global_step}")

    accelerator.save_state(ckpt_path)
    save_model(model, cfg, model_path, accelerator.is_main_process, weight_dtype)

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
