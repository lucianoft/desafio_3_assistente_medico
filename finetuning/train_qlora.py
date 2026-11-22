"""Fine-tuning QLoRA (PEFT + TRL) do Qwen2.5-1.5B-Instruct com o dataset do hospital.

Pensado para o Google Colab GRATUITO (GPU T4 16 GB, ~12 GB de RAM, sessao que pode cair):
  * modelo base em 4 bits (bitsandbytes) + treino so dos adaptadores LoRA (~1% dos parametros);
  * fp16 (a T4 nao tem bf16), gradient checkpointing, batch 4 x acumulacao 4;
  * 1 epoca e sequencias de 768 tokens por padrao (~30-40 min na T4);
  * checkpoints a cada 50 steps + --resume: se o Colab desconectar, o treino continua de onde parou
    (salve --output-dir no Google Drive).

Uso:
  python -m finetuning.train_qlora --epochs 2
  python -m finetuning.train_qlora --base-model meta-llama/Llama-3.2-1B-Instruct --output-dir models/llama-lora
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from core.config import settings
from finetuning.compat import (dtype_kwarg, gpu_supports_bf16, patch_trl_for_old_gpus, supported_kwargs,
                               trainable_params_to_fp32)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=settings.base_model)
    ap.add_argument("--data-dir", default=str(settings.processed_dir))
    ap.add_argument("--output-dir", default=str(settings.adapter_path))
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-train-samples", type=int, default=None, help="limita o treino (teste rapido)")
    ap.add_argument("--max-eval-samples", type=int, default=150, help="validacao menor = treino mais rapido")
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--resume", action="store_true", help="retoma do ultimo checkpoint em --output-dir")
    ap.add_argument("--no-4bit", action="store_true", help="desliga quantizacao (CPU / GPU sem bitsandbytes)")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args(argv)


def main(argv=None) -> None:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    ds = load_dataset("json", data_files={"train": str(data_dir / "train.jsonl"),
                                          "validation": str(data_dir / "val.jsonl")})
    ds = ds.select_columns(["messages"])
    if args.max_train_samples:
        ds["train"] = ds["train"].select(range(min(args.max_train_samples, len(ds["train"]))))
    if args.max_eval_samples:
        ds["validation"] = ds["validation"].shuffle(seed=args.seed).select(
            range(min(args.max_eval_samples, len(ds["validation"]))))

    use_cuda = torch.cuda.is_available()
    use_4bit = use_cuda and not args.no_4bit
    bf16 = gpu_supports_bf16()  # T4 (sm75) -> fp16
    patch_trl_for_old_gpus()
    dtype = torch.bfloat16 if bf16 else torch.float16 if use_cuda else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=dtype) if use_4bit else None
    model = AutoModelForCausalLM.from_pretrained(args.base_model, quantization_config=quant, **dtype_kwarg(dtype),
                                                 device_map="auto" if use_cuda else None)
    if use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    cfg_kwargs = dict(
        output_dir=args.output_dir, num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=10, eval_strategy="steps", eval_steps=args.save_steps, save_strategy="steps",
        save_steps=args.save_steps, save_total_limit=2,
        bf16=bf16, fp16=use_cuda and not bf16, gradient_checkpointing=use_cuda, report_to="none",
        seed=args.seed, max_seq_length=args.max_len,
    )
    config = SFTConfig(**supported_kwargs(SFTConfig, cfg_kwargs))
    trainer_kwargs = supported_kwargs(SFTTrainer, dict(
        model=model, args=config, train_dataset=ds["train"], eval_dataset=ds["validation"], peft_config=lora,
        processing_class=tokenizer))
    trainer = SFTTrainer(**trainer_kwargs)
    trainer.model.print_trainable_parameters()
    if use_cuda and not bf16:
        trainable_params_to_fp32(trainer.model)  # GradScaler (fp16) exige gradientes fp32

    start = time.time()
    from transformers.trainer_utils import get_last_checkpoint

    last = get_last_checkpoint(args.output_dir) if args.resume and Path(args.output_dir).is_dir() else None
    if last:
        print(f"Retomando do checkpoint {last}")
    result = trainer.train(resume_from_checkpoint=last)
    elapsed = time.time() - start
    eval_metrics = trainer.evaluate()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    history = [h for h in trainer.state.log_history]
    summary = {
        "base_model": args.base_model, "adapter_dir": args.output_dir, "train_samples": len(ds["train"]),
        "val_samples": len(ds["validation"]), "epochs": args.epochs, "learning_rate": args.lr,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
        "quantizacao_4bit": use_4bit, "train_loss": result.training_loss, "eval_loss": eval_metrics.get("eval_loss"),
        "tempo_treino_min": round(elapsed / 60, 1),
        "gpu": torch.cuda.get_device_name(0) if use_cuda else "cpu",
    }
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    (settings.reports_dir / "finetuning_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (settings.reports_dir / "finetuning_log_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    _plot_loss(history, settings.reports_dir / "finetuning_loss.png")
    print(json.dumps(summary, indent=2))


def _plot_loss(history: list[dict], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    tr = [(h["step"], h["loss"]) for h in history if "loss" in h]
    ev = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]
    plt.figure(figsize=(7, 4))
    if tr:
        plt.plot(*zip(*tr), label="treino")
    if ev:
        plt.plot(*zip(*ev), marker="o", label="validacao")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Fine-tuning QLoRA - curva de perda")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
