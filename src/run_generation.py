import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from config import (
    INPUT_FILE,
    MAX_NEW_TOKENS,
    MODEL_NAME,
    SEED,
)

from model_utils import (
    get_model_device,
    load_model,
    load_tokenizer,
)

from generation_utils import (
    append_jsonl,
    get_completed_ids,
    generate_response,
    load_jsonl,
    warm_up,
)


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def save_metrics(path, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists()

    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=metrics.keys(),
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(metrics)


def run(args):
    set_seed()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory_gb = (
        torch.cuda.get_device_properties(0).total_memory
        / 1024**3
    )

    print("GPU:", gpu_name)
    print("GPU memory:", round(gpu_memory_gb, 2), "GB")

    input_path = Path(args.input)
    output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(input_path)
    completed_ids = get_completed_ids(output_path)

    print("Total problems:", len(records))
    print("Already completed:", len(completed_ids))

    tokenizer = load_tokenizer()
    model = load_model(args.precision)
    device = get_model_device(model)

    print("Model device:", device)

    warm_up(
        model=model,
        tokenizer=tokenizer,
        question=records[0]["question"],
        device=device,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    total_start = time.perf_counter()
    total_generation_time = 0.0
    total_output_tokens = 0
    completed_count = 0

    for index, record in enumerate(records):
        problem_id = record["problem_id"]

        if problem_id in completed_ids:
            continue

        try:
            result = generate_response(
                model=model,
                tokenizer=tokenizer,
                question=record["question"],
                device=device,
                max_new_tokens=MAX_NEW_TOKENS,
            )

            output_record = {
                "problem_id": problem_id,
                "model": MODEL_NAME,
                "precision": args.precision,
                "question": record["question"],
                "response": result["response"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "generation_time_seconds": (
                    result["generation_time_seconds"]
                ),
                "gpu": gpu_name,
                "max_new_tokens": MAX_NEW_TOKENS,
                "batch_size": 1,
                "seed": SEED,
            }

            append_jsonl(output_path, output_record)

            completed_count += 1
            total_output_tokens += result["output_tokens"]
            total_generation_time += (
                result["generation_time_seconds"]
            )

            print(
                f"[{index + 1}/{len(records)}] "
                f"problem_id={problem_id} "
                f"time={result['generation_time_seconds']:.2f}s"
            )

        except Exception as error:
            print(
                f"Error on problem {problem_id}: {error}"
            )
            print("Continuing with the next problem.")

    total_elapsed = time.perf_counter() - total_start

    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / 1024**2
    )

    tokens_per_second = None

    if total_generation_time > 0:
        tokens_per_second = (
            total_output_tokens / total_generation_time
        )

    metrics = {
        "model": MODEL_NAME,
        "precision": args.precision,
        "gpu": gpu_name,
        "num_dataset_records": len(records),
        "completed_this_run": completed_count,
        "peak_gpu_memory_mb": peak_memory_mb,
        "total_elapsed_seconds": total_elapsed,
        "total_generation_seconds": total_generation_time,
        "total_output_tokens": total_output_tokens,
        "tokens_per_second": tokens_per_second,
        "max_new_tokens": MAX_NEW_TOKENS,
        "batch_size": 1,
        "seed": SEED,
    }

    save_metrics(args.metrics, metrics)

    print("\nExperiment complete.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--precision",
        required=True,
        choices=["fp16", "int8", "nf4"],
    )

    parser.add_argument(
        "--input",
        default=INPUT_FILE,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--metrics",
        default="results/gpu_metrics.csv",
    )

    args = parser.parse_args()

    run(args)