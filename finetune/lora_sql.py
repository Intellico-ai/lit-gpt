import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy
import torch.optim.lr_scheduler as lr_scheduler

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from lit_gpt.lora import GPT, Block, Config, lora_filter, mark_only_lora_as_trainable
from lit_gpt.speed_monitor import SpeedMonitorFabric as SpeedMonitor
from lit_gpt.speed_monitor import estimate_flops, measure_flops
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.utils import (
    check_valid_checkpoint_dir,
    chunked_cross_entropy,
    get_default_supported_precision,
    lazy_load,
    num_parameters,
    quantization,
    step_csv_logger,
)
from scripts.prepare_sql import generate_prompt

from lightning.pytorch.loggers import TensorBoardLogger
from datetime import datetime
from tqdm import tqdm, trange

# default configuration of llama, maybe they are not right for the small amount, i'll do 2 different tries
LLAMA_CONFIG = dict(
    beta1=0.9,
    bera2=0.95,
    adam_eps=1e-5,
    warmup_steps=2000,
    lr_decay_perc=0.1,
    weight_decay=0.1,
    gradient_clipping=1,
    lr=2e-5,
)
USE_LLAMA_CONFIG = True

batch_size = 128
micro_batch_size = 4  # i dont know what happens if it's not divisible
eval_interval = 5  # evaluate every eval_interval batch
save_interval = 200  # save every eval_interval batch
eval_iters = 100  # how many sample for validation are extracted (100*micro_batch_size)
log_interval = 1  # every tot iterations the training metrics are evaluated
devices = 1
# change this value to force a maximum sequence length
override_max_seq_length = None
clip_grad_value = 1

# Hyperparameters
learning_rate = 3e-4

TEMPERATURE = 0.1
gradient_accumulation_iters = batch_size // micro_batch_size
assert gradient_accumulation_iters > 0
max_iters = 20000  # number of microbatch processed
weight_decay = 0.01
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
lora_query = True
lora_key = True
lora_value = True
lora_projection = True
lora_mlp = True
lora_head = True
warmup_steps = 100

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}


def setup(
    data_dir: Path = Path("data/sql-create-context"),
    checkpoint_dir: Path = Path("checkpoints/meta-llama/Llama-2-7b-chat-hf/"),
    out_dir: Path = Path("out/lora/sql_llama_prompt"),
    precision: Optional[str] = None,
    tpu: bool = False,
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq"]] = None,
):
    precision = precision or get_default_supported_precision(training=True, tpu=tpu)

    fabric_devices = devices
    if fabric_devices > 1:
        if quantize:
            raise NotImplementedError(
                "Quantization is currently not supported for multi-GPU training. "
                "Please set devices=1 when using the --quantization flag."
            )
        if tpu:
            # For multi-host TPU training, the device count for Fabric is limited to the count on a single host.
            fabric_devices = "auto"
            strategy = XLAStrategy(sync_module_states=False)
        else:
            strategy = FSDPStrategy(
                auto_wrap_policy={Block},
                activation_checkpointing_policy={Block},
                state_dict_type="full",
                limit_all_gathers=True,
            )
    else:
        strategy = "auto"

    logger = step_csv_logger(out_dir.parent, out_dir.name, flush_logs_every_n_steps=log_interval)
    fabric = L.Fabric(devices=fabric_devices, strategy=strategy, precision=precision, loggers=logger)
    fabric.print(hparams)
    fabric.launch(main, data_dir, checkpoint_dir, out_dir, quantize)


def main(fabric: L.Fabric, data_dir: Path, checkpoint_dir: Path, out_dir: Path, quantize: Optional[str] = None):
    # checks if the checkpoint is correct
    check_valid_checkpoint_dir(checkpoint_dir)
    # something that checks the speed of the training
    speed_monitor = SpeedMonitor(fabric, window_size=50, time_unit="seconds")

    fabric.seed_everything(1337)  # same seed for every process to init model (FSDP)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    # load train and val data from data_dir
    train_data = torch.load(data_dir / "train.pt")
    val_data = torch.load(data_dir / "test.pt")

    # tells where lora is put
    if not any((lora_query, lora_key, lora_value, lora_projection, lora_mlp, lora_head)):
        fabric.print("Warning: all LoRA layers are disabled!")
    # creates a configuration given all the values at the beginning of this file
    config = Config.from_name(
        name=checkpoint_dir.name,
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
        to_query=lora_query,
        to_key=lora_key,
        to_value=lora_value,
        to_projection=lora_projection,
        to_mlp=lora_mlp,
        to_head=lora_head,
    )
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}")
    # i added the quantization, but it doesnt work
    # --------------
    # ERROR:  File "/home/ubuntu/lit-parrot/lit_gpt/rmsnorm.py", line 19, in forward
    # norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
    # RuntimeError: CUDA error: device-side assert triggered
    # with fabric.init_module(empty_init=False), quantization("bnb.nf4"):
    # --------------
    with fabric.init_module(empty_init=False), quantization(quantize):
        model = GPT(config)
        model.apply(model._init_weights)  # for the LoRA weights
    # load original weights
    with lazy_load(checkpoint_path) as checkpoint:
        # strict=False because missing keys due to LoRA weights not contained in state dict
        model.load_state_dict(checkpoint, strict=False)
    # freezes big model
    mark_only_lora_as_trainable(model)

    fabric.print(f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")
    fabric.print(f"Number of non trainable parameters: {num_parameters(model, requires_grad=False):,}")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    max_steps = max_iters//micro_batch_size
    if quantize and quantize.startswith("bnb."):
        import bitsandbytes as bnb

        if USE_LLAMA_CONFIG:
            llama_lr = LLAMA_CONFIG["lr"]
            llama_lr_perc = LLAMA_CONFIG["lr_decay_perc"]
            optimizer = bnb.optim.PagedAdamW(
                trainable_params,
                lr=LLAMA_CONFIG["lr"],
                weight_decay=LLAMA_CONFIG["weight_decay"],
                betas=(LLAMA_CONFIG["beta1"], LLAMA_CONFIG["beta2"]),
                eps=LLAMA_CONFIG["adam_eps"],
            )
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps-warmup_steps, eta_min=llama_lr * llama_lr_perc)

        else:
            optimizer = bnb.optim.PagedAdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
    else:
        if USE_LLAMA_CONFIG:
            llama_lr = LLAMA_CONFIG["lr"]
            llama_lr_perc = LLAMA_CONFIG["lr_decay_perc"]
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=LLAMA_CONFIG["lr"],
                weight_decay=LLAMA_CONFIG["weight_decay"],
                betas=(LLAMA_CONFIG["beta1"], LLAMA_CONFIG["beta2"]),
                eps=LLAMA_CONFIG["adam_eps"],
            )

            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps-warmup_steps, eta_min=llama_lr * llama_lr_perc)
        else:
            optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)

    assert isinstance(
        scheduler, (torch.optim.lr_scheduler.ReduceLROnPlateau, torch.optim.lr_scheduler.CosineAnnealingLR)
    ), f"Scheduler is of type {type(scheduler)}, not ReduceLROnPlateau or CosineAnnealingLR"

    model, optimizer = fabric.setup(model, optimizer)

    fabric.seed_everything(1337 + fabric.global_rank)

    train_time = time.perf_counter()
    train(fabric, model, optimizer, train_data, val_data, checkpoint_dir, out_dir, speed_monitor, scheduler)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    # Save the final LoRA checkpoint at the end of training
    save_path = out_dir / "lit_model_lora_finetuned.pth"
    save_lora_checkpoint(fabric, model, save_path)


def train(
    fabric: L.Fabric,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    train_data: List[Dict],
    val_data: List[Dict],
    checkpoint_dir: Path,
    out_dir: Path,
    speed_monitor: SpeedMonitor,
    scheduler: Optional[lr_scheduler._LRScheduler] = None
) -> None:
    # tensorboard
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%d_%m_%Y-%H-%M")

    tb: TensorBoardLogger = TensorBoardLogger("tb", name=f"LoRA_{formatted_datetime}", flush_secs=1)
    tokenizer = Tokenizer(checkpoint_dir)
    max_seq_length, longest_seq_length, longest_seq_ix = get_max_seq_length(train_data)

    validate(fabric, model, val_data, tokenizer, longest_seq_length)  # sanity check

    with torch.device("meta"):
        meta_model = GPT(model.config)
        mark_only_lora_as_trainable(meta_model)
        # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
        # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
        # consider passing `SpeedMonitor(flops_per_batch=estimated_flops)` instead
        estimated_flops = estimate_flops(meta_model) * micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        # this assumes that all samples have a fixed length equal to the longest sequence length
        # which is most likely false during finetuning
        x = torch.randint(0, 1, (micro_batch_size, longest_seq_length))
        measured_flops = measure_flops(meta_model, x)
        fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    step_count = 0
    total_lengths = 0
    total_t0 = time.perf_counter()

    if fabric.device.type == "xla":
        import torch_xla.core.xla_model as xm

        xm.mark_step()

    iters_progress_bar = trange(max_iters)
    for iter_num in iters_progress_bar:
        if step_count <= warmup_steps:
            # linear warmup
            lr = learning_rate * step_count / warmup_steps
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        iter_t0 = time.perf_counter()

        input_ids, targets = get_batch(
            fabric, train_data, longest_seq_length, longest_seq_ix if iter_num == 0 else None
        )

        is_accumulating = (iter_num + 1) % gradient_accumulation_iters != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids, max_seq_length=max_seq_length, lm_head_chunk_size=128)
            # shift the targets such that output n predicts token n+1
            logits[-1] = logits[-1][..., :-1, :]
            loss = chunked_cross_entropy(logits, targets[..., 1:])
            fabric.backward(loss / gradient_accumulation_iters)

        if not is_accumulating:
            if USE_LLAMA_CONFIG:
                fabric.clip_gradients(model, clip_norm=LLAMA_CONFIG["gradient_clipping"])
            else:
                fabric.clip_gradients(model, clip_norm=clip_grad_value)

            optimizer.step()
            optimizer.zero_grad()
            # if cosine scheduler and not warmup, apply schedule
            step_count += 1
            # cosine scheduler does this only if not in warmup
            if step_count >= warmup_steps and isinstance(scheduler, lr_scheduler.CosineAnnealingLR):
                scheduler.step()
            iters_progress_bar.set_description(f"Opt Step {step_count}, Iter {iter_num}")
        elif fabric.device.type == "xla":
            xm.mark_step()

        t1 = time.perf_counter()
        total_lengths += input_ids.size(1)
        speed_monitor.on_train_batch_end(
            (iter_num + 1) * micro_batch_size,
            t1 - total_t0,
            # this assumes that device FLOPs are the same and that all devices have the same batch size
            fabric.world_size,
            flops_per_batch=measured_flops,
            lengths=total_lengths,
        )
        if iter_num % log_interval == 0:
            # fabric.print(
            #     f"iter {iter_num} step {step_count}: loss {loss.item():.4f}, iter time:"
            #     f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
            # )
            for param_group in optimizer.param_groups:
                log_lr = param_group["lr"]
            tb.log_metrics({"loss/train": loss.item(), "optim/lr": log_lr}, iter_num)

        if not is_accumulating and step_count % eval_interval == 0:
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_data, tokenizer, longest_seq_length)
            t1 = time.perf_counter() - t0
            speed_monitor.eval_end(t1)
            fabric.print(f"step {iter_num}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms")
            fabric.barrier()
            tb.log_metrics(
                {
                    "loss/val": val_loss,
                },
                iter_num,
            )
            if step_count >= warmup_steps and isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
                
        if not is_accumulating and step_count % save_interval == 0:
            checkpoint_path = out_dir / f"iter-{iter_num:06d}-ckpt.pth"
            save_lora_checkpoint(fabric, model, checkpoint_path)


@torch.no_grad()
def validate(
    fabric: L.Fabric, model: GPT, val_data: List[Dict], tokenizer: Tokenizer, longest_seq_length: int
) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        input_ids, targets = get_batch(fabric, val_data, longest_seq_length)
        logits = model(input_ids)
        loss = chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:], chunk_size=0)
        losses[k] = loss.item()
    val_loss = losses.mean()

    # produce an example:
    instruction = "How many heads of the departments are older than 56 ?"
    input_string = "CREATE TABLE head (age INTEGER)"
    fabric.print(instruction)
    sample = {"instruction": instruction, "input": input_string}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    max_returned_tokens = len(encoded) + 100
    output = generate(
        model,
        idx=encoded,
        max_returned_tokens=max_returned_tokens,
        max_seq_length=max_returned_tokens,
        temperature=TEMPERATURE,
    )
    output = tokenizer.decode(output)
    fabric.print(output)

    model.reset_cache()

    model.train()
    return val_loss.item()


def get_batch(
    fabric: L.Fabric, data: List[Dict], longest_seq_length: int, longest_seq_ix: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data), (micro_batch_size,))
    if longest_seq_ix is not None:
        # force the longest sample at the beginning so potential OOMs happen right away
        ix[0] = longest_seq_ix

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]

    # it's better to pad to a fixed seq length with XLA to avoid recompilation
    max_len = max(len(s) for s in input_ids) if fabric.device.type != "xla" else longest_seq_length

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])

    if fabric.device.type == "cuda" and x.device.type == "cpu":
        x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    else:
        x, y = fabric.to_device((x, y))
    return x, y


def get_max_seq_length(data: List[Dict]) -> Tuple[int, int, int]:
    # find out the minimum max_seq_length required during fine-tuning (saves memory!)
    lengths = [len(d["input_ids"]) for d in data]
    max_seq_length = max(lengths)
    longest_seq_ix = lengths.index(max_seq_length)
    # support easy override at the top of the file
    return (
        override_max_seq_length if isinstance(override_max_seq_length, int) else max_seq_length,
        max_seq_length,
        longest_seq_ix,
    )


def save_lora_checkpoint(fabric: L.Fabric, model, file_path: Path):
    fabric.print(f"Saving LoRA weights to {str(file_path)!r}")
    fabric.save(file_path, {"model": model}, filter={"model": lora_filter})


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)