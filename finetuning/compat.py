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
