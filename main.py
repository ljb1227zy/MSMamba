import argparse
import datetime
import json
import random
import time
from contextlib import suppress
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.utils import ModelEma, NativeScaler, get_state_dict

import models.msfd  

import utils as utils
from engine import evaluate, train_one_epoch
from model.utils import create_optimizer
from ref_dataset import build_dataset, collate_fn


def get_args_parser():
    """Build command-line arguments for training and evaluating MSMamba."""
    parser = argparse.ArgumentParser(
        "MSMamba training and evaluation script",
        add_help=False,
    )

    # -------------------------------------------------------------------------
    # Basic training settings
    # -------------------------------------------------------------------------
    parser.add_argument("--batch-size", "--batch_size", default=8, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", "--num_workers", default=4, type=int)
    parser.add_argument("--pin-mem", "--pin_mem", action="store_true", dest="pin_mem")
    parser.add_argument("--no-pin-mem", "--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # -------------------------------------------------------------------------
    # Model settings
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--model",
        default="MSMamba",
        choices=["MSMamba"],
        type=str,
        help="Model name registered in timm.",
    )
    parser.add_argument("--model-size", "--model_size", default="base", type=str)
    parser.add_argument("--input-size", "--input_size", default=480, type=int)
    parser.add_argument("--drop", default=0.0, type=float)
    parser.add_argument("--drop-path", "--drop_path", default=0.1, type=float)

    # -------------------------------------------------------------------------
    # Dataset settings
    # -------------------------------------------------------------------------
    parser.add_argument("--data-path", "--data_path", default="./ref_dataset/data", type=str)
    parser.add_argument(
        "--data-set", "--data_set",
        default="rrsisd",
        choices=[
            "refcoco",
            "refcoco+",
            "refcocog",
            "rrsisd",
            "risbench",
            "rrsishr",
        ],
        type=str,
    )
    parser.add_argument("--eval-split", "--eval_split", default="val", type=str)
    parser.add_argument("--eval-batch-size", "--eval_batch_size", default=1, type=int)

    # -------------------------------------------------------------------------
    # Optimizer settings
    # -------------------------------------------------------------------------
    parser.add_argument("--opt", default="adamw", type=str)
    parser.add_argument("--opt-eps", "--opt_eps", default=1e-8, type=float)
    parser.add_argument("--opt-betas", "--opt_betas", default=None, type=float, nargs="+")
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", "--weight_decay", default=1e-4, type=float)
    parser.add_argument("--clip-grad", "--clip_grad", default=None, type=float)

    # Different learning rates are kept for parameter groups used by create_optimizer.
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--lr-decoder", "--lr_decoder", default=5e-5, type=float)
    parser.add_argument("--lr-backbone", "--lr_backbone", default=2.5e-5, type=float)

    # -------------------------------------------------------------------------
    # Learning-rate scheduler settings
    # -------------------------------------------------------------------------
    parser.add_argument("--sched", default="cosine", type=str)
    parser.add_argument("--warmup-lr", "--warmup_lr", default=1e-6, type=float)
    parser.add_argument("--min-lr", "--min_lr", default=1e-6, type=float)
    parser.add_argument("--decay-epochs", "--decay_epochs", default=30, type=float)
    parser.add_argument("--warmup-epochs", "--warmup_epochs", default=0, type=int)
    parser.add_argument("--cooldown-epochs", "--cooldown_epochs", default=10, type=int)
    parser.add_argument("--patience-epochs", "--patience_epochs", default=10, type=int)
    parser.add_argument("--decay-rate", "--decay_rate", default=0.1, type=float)

    # -------------------------------------------------------------------------
    # Checkpoint and evaluation settings
    # -------------------------------------------------------------------------
    parser.add_argument("--output-dir", "--output_dir", default="", type=str)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--finetune", default="", type=str)
    parser.add_argument("--start-epoch", "--start_epoch", default=0, type=int)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval-freq", "--eval_freq", default=1, type=int)
    parser.add_argument(
        "--best-metric", "--best_metric",
        default="oiou",
        choices=["iou", "oiou"],
        help="Metric used to select best_checkpoint.pth.",
    )
    parser.add_argument(
        "--save-latest", "--save_latest",
        action="store_true",
        help="Overwrite checkpoint_latest.pth after each epoch.",
    )

    # -------------------------------------------------------------------------
    # Exponential Moving Average
    # -------------------------------------------------------------------------
    parser.add_argument("--model-ema", "--model_ema", action="store_true")
    parser.add_argument("--no-model-ema", "--no_model_ema", action="store_false", dest="model_ema")
    parser.set_defaults(model_ema=False)
    parser.add_argument("--model-ema-decay", "--model_ema_decay", default=0.9999, type=float)
    parser.add_argument("--model-ema-force-cpu", "--model_ema_force_cpu", action="store_true", default=False)

    # -------------------------------------------------------------------------
    # Mixed precision
    # -------------------------------------------------------------------------
    parser.add_argument("--amp", action="store_true", dest="if_amp")
    parser.add_argument("--no-amp", action="store_false", dest="if_amp")
    parser.set_defaults(if_amp=True)

    # -------------------------------------------------------------------------
    # Distributed training
    # -------------------------------------------------------------------------
    parser.add_argument("--distributed", action="store_true", default=False)
    parser.add_argument("--dist-eval", "--dist_eval", action="store_true", default=False)
    parser.add_argument("--world-size", "--world_size", default=1, type=int)
    parser.add_argument("--dist-url", "--dist_url", default="env://")
    parser.add_argument("--local-rank", "--local_rank", default=0, type=int)

    # Some training pipelines need to freeze BatchNorm/Dropout behavior manually.
    parser.add_argument("--train-mode", "--train_mode", action="store_true", dest="train_mode")
    parser.add_argument("--no-train-mode", "--no_train_mode", action="store_false", dest="train_mode")
    parser.set_defaults(train_mode=True)

    # Use a small subset of the data for quick code checks.
    parser.add_argument("--debug", action="store_true")

    return parser


def set_random_seed(seed):
    """Set random seeds for reproducible experiments."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(args):
    """Create training and validation dataloaders."""
    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_dataset(is_train=False, args=args, split=args.eval_split)

    if args.debug:
        dataset_train = torch.utils.data.Subset(
            dataset_train,
            range(min(1000, len(dataset_train))),
        )
        dataset_val = torch.utils.data.Subset(
            dataset_val,
            range(min(300, len(dataset_val))),
        )

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=True,
        )

        if args.dist_eval:
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val,
                num_replicas=num_tasks,
                rank=global_rank,
                shuffle=False,
            )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        collate_fn=collate_fn,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=collate_fn,
    )

    return dataset_train, dataset_val, data_loader_train, data_loader_val


def load_checkpoint(path):
    """Load a local or URL checkpoint."""
    if path.startswith("https"):
        return torch.hub.load_state_dict_from_url(
            path,
            map_location="cpu",
            check_hash=True,
        )
    return torch.load(path, map_location="cpu")


def metric_to_percent(value):
    """Convert a metric to percentage format for consistent printing."""
    if value is None:
        return 0.0

    value = float(value)
    return value * 100.0 if value <= 1.0 else value


def print_eval_stats(stats, prefix="Eval"):
    """Print IoU, oIoU, and optional Pr@threshold metrics."""
    iou = metric_to_percent(stats.get("iou", 0.0))
    oiou = metric_to_percent(stats.get("oiou", 0.0))
    print(f"{prefix}: IoU {iou:.2f}% | oIoU {oiou:.2f}%")

    pr_keys = sorted(
        [key for key in stats.keys() if key.startswith("pr@")],
        key=lambda key: float(key.split("@")[-1]),
    )

    for key in pr_keys:
        print(f"  {key}: {metric_to_percent(stats[key]):.2f}%")


def save_checkpoint(
    args,
    output_dir,
    filename,
    epoch,
    model_without_ddp,
    optimizer,
    lr_scheduler,
    loss_scaler,
    model_ema=None,
    save_ema_weights=False,
):
    """Save model, optimizer, scheduler, and scaler states."""
    if not args.output_dir or not utils.is_main_process():
        return

    if save_ema_weights and model_ema is not None:
        model_state = get_state_dict(model_ema)
    else:
        model_state = model_without_ddp.state_dict()

    checkpoint = {
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
        "args": args,
    }

    if model_ema is not None:
        checkpoint["model_ema"] = get_state_dict(model_ema)

    if loss_scaler != "none":
        checkpoint["scaler"] = loss_scaler.state_dict()

    utils.save_on_master(checkpoint, output_dir / filename)


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    set_random_seed(seed)
    print(f"Random seed: {seed}")

    cudnn.benchmark = True

    dataset_train, dataset_val, data_loader_train, data_loader_val = build_dataloaders(args)

    print(f"Creating model: {args.model}")
    model, new_param = create_model(
        args.model,
        img_size=args.input_size,
        model_size=args.model_size,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
    )

    if args.finetune:
        checkpoint = load_checkpoint(args.finetune)
        checkpoint_model = checkpoint.get("model", checkpoint)
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(f"Loaded finetune checkpoint: {msg}")

    model.to(device)

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else "",
            resume="",
        )

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=True,
        )
        model_without_ddp = model.module

    n_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Trainable parameters: {n_parameters / 1e6:.2f}M")

    optimizer = create_optimizer(args, model_without_ddp, new_param)
    lr_scheduler, _ = create_scheduler(args, optimizer)

    amp_autocast = torch.cuda.amp.autocast if args.if_amp else suppress
    loss_scaler = NativeScaler() if args.if_amp else "none"

    output_dir = Path(args.output_dir)

    if args.resume:
        checkpoint = load_checkpoint(args.resume)

        msg = model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
        print(f"Loaded resume checkpoint: {msg}")

        if (
            "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
            and not args.eval
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1

            if loss_scaler != "none" and "scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["scaler"])

        if model_ema is not None:
            if "model_ema" in checkpoint:
                utils._load_checkpoint_for_ema(model_ema, checkpoint["model_ema"])
            else:
                model_ema.ema.load_state_dict(model_without_ddp.state_dict(), strict=False)

        lr_scheduler.step(args.start_epoch)

    if args.eval:
        eval_model = model_ema.ema if model_ema is not None else model
        test_stats = evaluate(data_loader_val, eval_model, device, amp_autocast)
        print_eval_stats(test_stats, prefix=args.eval_split)
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    best_score = -1.0
    best_epoch = -1
    best_stats = {}

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model=model,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            amp_autocast=amp_autocast,
            max_norm=args.clip_grad,
            model_ema=model_ema,
            set_training_mode=args.train_mode,
        )

        lr_scheduler.step(epoch)

        do_eval = ((epoch + 1) % args.eval_freq == 0) or (epoch + 1 == args.epochs)
        test_stats = {}
        test_stats_ema = {}

        if do_eval:
            test_stats = evaluate(
                data_loader_val,
                model,
                device,
                amp_autocast,
                log_every=50,
            )
            print_eval_stats(test_stats, prefix=f"Epoch {epoch}")

            if model_ema is not None:
                test_stats_ema = evaluate(
                    data_loader_val,
                    model_ema.ema,
                    device,
                    amp_autocast,
                    log_every=50,
                )
                print_eval_stats(test_stats_ema, prefix=f"Epoch {epoch} EMA")

            score_source = test_stats_ema if model_ema is not None else test_stats
            current_score = metric_to_percent(score_source.get(args.best_metric, 0.0))

            if current_score > best_score:
                best_score = current_score
                best_epoch = epoch
                best_stats = score_source

                save_checkpoint(
                    args=args,
                    output_dir=output_dir,
                    filename="best_checkpoint.pth",
                    epoch=epoch,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    loss_scaler=loss_scaler,
                    model_ema=model_ema,
                    save_ema_weights=model_ema is not None,
                )
                print(
                    f"New best {args.best_metric}: {best_score:.2f}% "
                    f"at epoch {best_epoch}"
                )

        if args.save_latest:
            save_checkpoint(
                args=args,
                output_dir=output_dir,
                filename="checkpoint_latest.pth",
                epoch=epoch,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                loss_scaler=loss_scaler,
                model_ema=model_ema,
                save_ema_weights=False,
            )

        log_stats = {
            **{f"train_{key}": value for key, value in train_stats.items()},
            **{f"test_{key}": value for key, value in test_stats.items()},
            **{f"test_ema_{key}": value for key, value in test_stats_ema.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as file:
                file.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    print(f"Best epoch: {best_epoch}")
    print_eval_stats(best_stats, prefix="Best")
    print(f"Training time: {total_time_str}")

    if args.output_dir and utils.is_main_process():
        with (output_dir / "log.txt").open("a") as file:
            file.write(f"Best epoch: {best_epoch}\n")
            file.write(f"Best {args.best_metric}: {best_score:.2f}%\n")
            file.write(f"Training time: {total_time_str}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
