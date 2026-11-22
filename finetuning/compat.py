"""Compatibilidade entre versoes de transformers/TRL (o Colab pode instalar versoes diferentes)."""
from __future__ import annotations

import inspect


def dtype_kwarg(dtype) -> dict:
    """transformers >= 5 usa `dtype`; versoes anteriores usam `torch_dtype`."""
    import transformers

    major = int(transformers.__version__.split(".")[0])
    return {"dtype": dtype} if major >= 5 else {"torch_dtype": dtype}


def supported_kwargs(cls, kwargs: dict) -> dict:
    """Mantem so os argumentos aceitos por cls.__init__ e adapta nomes que mudaram entre versoes."""
    params = inspect.signature(cls.__init__).parameters
    kw = dict(kwargs)
    if "max_seq_length" in kw and "max_seq_length" not in params and "max_length" in params:
        kw["max_length"] = kw.pop("max_seq_length")
    if "warmup_ratio" in kw and "warmup_ratio" not in params:
        kw["warmup_steps"] = kw.pop("warmup_ratio")  # transformers 5: float < 1 = proporcao
    if "eval_strategy" in kw and "eval_strategy" not in params and "evaluation_strategy" in params:
        kw["evaluation_strategy"] = kw.pop("eval_strategy")
    if "processing_class" in kw and "processing_class" not in params and "tokenizer" in params:
        kw["tokenizer"] = kw.pop("processing_class")
    dropped = sorted(set(kw) - set(params))
    if dropped:
        print(f"[compat] argumentos ignorados por {cls.__name__}: {dropped}")
    return {k: v for k, v in kw.items() if k in params}


def gpu_supports_bf16() -> bool:
    """bf16 nativo so a partir de compute capability 8.0 (Ampere). A T4 do Colab free e 7.5.

    `torch.cuda.is_bf16_supported()` pode retornar True na T4 (emulacao), o que quebra o treino.
    """
    import torch

    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (8, 0)


def patch_trl_for_old_gpus() -> None:
    """TRL recente usa `torch.addmm(..., out_dtype=float32)` com entradas fp16/bf16 na perda (chunked_logprob),
    o que so funciona em GPUs com compute capability >= 8.0. Na T4 desliga esse atalho (usa o caminho em fp32)."""
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) >= (8, 0):
        return
    try:
        import trl.kernels.chunked_logprob as ck
    except ImportError:  # versoes antigas da TRL nao tem o kernel
        return
    if getattr(ck, "_MM_OUT_DTYPE", False):
        ck._MM_OUT_DTYPE = False
        print("[compat] GPU < sm80: desligado addmm(out_dtype=float32) da TRL")


def trainable_params_to_fp32(model) -> None:
    """Com fp16 (T4) o GradScaler so aceita gradientes fp32. Dependendo da versao de peft/TRL, os adaptadores
    LoRA saem no dtype original do modelo (bf16 no Qwen2.5) -> NotImplementedError em
    _amp_foreach_non_finite_check_and_unscale_cuda. Converte so os parametros treinaveis (~1%) para fp32."""
    import torch
    from collections import Counter

    before = Counter()
    for p in model.parameters():
        if p.requires_grad:
            before[str(p.dtype)] += 1
            if p.dtype != torch.float32:
                p.data = p.data.float()
    print(f"[compat] parametros treinaveis convertidos para fp32 (antes: {dict(before)})")


def patch_peft_torchao() -> None:
    """O Colab traz torchao antigo (0.10) e o peft recente levanta ImportError ao carregar um adaptador num
    modelo nao quantizado ("Found an incompatible version of torchao"). O projeto nao usa torchao: trata
    versao incompativel como "nao instalado"."""
    import importlib
    import sys

    def _safe(orig):
        def wrapper(*a, **k):
            try:
                return orig(*a, **k)
            except ImportError:
                return False
        wrapper._fase3_safe = True
        return wrapper

    for name in ("peft.import_utils", "peft.tuners.lora.torchao", "peft.utils.quantization_utils"):
        try:
            mod = sys.modules.get(name) or importlib.import_module(name)
        except Exception:
            continue
        fn = getattr(mod, "is_torchao_available", None)
        if fn is not None and not getattr(fn, "_fase3_safe", False):
            mod.is_torchao_available = _safe(fn)
