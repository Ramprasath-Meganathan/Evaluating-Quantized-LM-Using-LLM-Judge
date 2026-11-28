import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from config import MODEL_NAME, MODEL_DTYPE


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    return tokenizer


def load_model(precision):
    if precision == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=MODEL_DTYPE,
            device_map="auto",
        )

    elif precision == "int8":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            dtype=MODEL_DTYPE,
            device_map="auto",
        )

    elif precision == "nf4":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=MODEL_DTYPE,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            dtype=MODEL_DTYPE,
            device_map="auto",
        )

    else:
        raise ValueError(
            "precision must be fp16, int8, or nf4"
        )

    model.eval()

    return model


def get_model_device(model):
    return next(model.parameters()).device