import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import traceback

from config import (
    INPUT_FILE,
    MAX_NEW_TOKENS,
    MODEL_NAME,
    SEED,
    BATCH_SIZE,
)

from model_utils import (
    get_model_device,
    load_model,
    load_tokenizer,
)

from generation_utils import (
    append_jsonl,
    get_completed_ids,
    generate_batch,
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
    print(
        "GPU memory:",
        round(gpu_memory_gb, 2),
        "GB"
    )

    input_path = Path(args.input)
    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    records = load_jsonl(input_path)

    completed_ids = get_completed_ids(
        output_path
    )

    print("Total problems:", len(records))
    print(
        "Already completed:",
        len(completed_ids)
    )

    tokenizer = load_tokenizer()
    print("pad:", tokenizer.pad_token_id, tokenizer.pad_token)
    print("eos:", tokenizer.eos_token_id, tokenizer.eos_token)
    model = load_model(args.precision)
    print("dtype:", model.get_input_embeddings().weight.dtype)
    device = get_model_device(model)

    print("Model device:", device)
    print("Batch size:", BATCH_SIZE)

    # Warm up the GPU before measuring performance
    warm_up(
        model=model,
        tokenizer=tokenizer,
        question=records[0]["question"],
        device=device,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    total_start_time = time.perf_counter()
    total_generation_time = 0.0
    total_output_tokens = 0
    completed_count = 0

    # Process multiple questions in each batch
    for start in range(
        0,
        len(records),
        BATCH_SIZE
    ):
        batch_records = records[
            start:start + BATCH_SIZE
        ]

        # Remove problems already saved from an earlier run
        batch_records = [
            record
            for record in batch_records
            if record["problem_id"]
            not in completed_ids
        ]

        if not batch_records:
            continue

        questions = [
            record["question"]
            for record in batch_records
        ]

        try:
            batch_results, batch_time = generate_batch(
                model=model,
                tokenizer=tokenizer,
                questions=questions,
                device=device,
                max_new_tokens=MAX_NEW_TOKENS,
            )

            batch_output_tokens = 0

            for record, result in zip(
                batch_records,
                batch_results,
            ):
                output_record = {
                    "problem_id": record["problem_id"],
                    "model": MODEL_NAME,
                    "precision": args.precision,
                    "question": record["question"],
                    "response": result["response"],
                    "output_tokens": result["output_tokens"],
                    "batch_size": BATCH_SIZE,
                    "batch_time_seconds": batch_time,
                    "gpu": gpu_name,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "seed": SEED,
                    "truncated": result["output_tokens"] >= MAX_NEW_TOKENS
                }

                # Save each response immediately
                append_jsonl(
                    output_path,
                    output_record,
                )

                completed_ids.add(
                    record["problem_id"]
                )

                completed_count += 1

                batch_output_tokens += (
                    result["output_tokens"]
                )

            total_output_tokens += (
                batch_output_tokens
            )

            # Add the batch time only once
            total_generation_time += batch_time

            print(
                "Completed problems "
                f"{start + 1}-"
                f"{start + len(batch_records)} "
                f"in {batch_time:.2f} seconds"
            )

        except Exception as error:
            problem_ids = [
                record["problem_id"]
                for record in batch_records
            ]

            print(f"Batch failed: {problem_ids}")
            traceback.print_exc()

            append_jsonl(
                "results/failed_batches.jsonl",
                {
                    "problem_ids": problem_ids,
                    "precision": args.precision,
                    "error": repr(error),
                },
            )

            if isinstance(error, torch.cuda.OutOfMemoryError):
                raise

    total_elapsed_time = (
        time.perf_counter()
        - total_start_time
    )

    peak_memory_mb = (
        torch.cuda.max_memory_allocated()
        / 1024**2
    )

    tokens_per_second = None

    if total_generation_time > 0:
        tokens_per_second = (
            total_output_tokens
            / total_generation_time
        )

    average_time_per_problem = None

    if completed_count > 0:
        average_time_per_problem = (
            total_generation_time
            / completed_count
        )

    metrics = {
        "model": MODEL_NAME,
        "precision": args.precision,
        "gpu": gpu_name,
        "num_dataset_records": len(records),
        "completed_this_run": completed_count,
        "batch_size": BATCH_SIZE,
        "peak_gpu_memory_mb": peak_memory_mb,
        "total_elapsed_seconds": total_elapsed_time,
        "total_generation_seconds": total_generation_time,
        "average_time_per_problem_seconds": (
            average_time_per_problem
        ),
        "total_output_tokens": total_output_tokens,
        "tokens_per_second": tokens_per_second,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": SEED,
    }

    save_metrics(
        args.metrics,
        metrics
    )

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