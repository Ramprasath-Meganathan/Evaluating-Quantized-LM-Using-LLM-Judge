import json
import time
from pathlib import Path
import time
import torch


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def build_inputs(question, tokenizer, device):
    prompt = f"""Solve the following math problem.
Think step by step, then end your answer with:
Final answer: <number>

Problem:
{question}
"""

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    return inputs.to(device)


def synchronize_gpu():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def generate_response(
    model,
    tokenizer,
    question,
    device,
    max_new_tokens,
):
    inputs = build_inputs(
        question=question,
        tokenizer=tokenizer,
        device=device,
    )

    input_tokens = inputs["input_ids"].shape[-1]

    synchronize_gpu()
    start_time = time.perf_counter()

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    synchronize_gpu()
    elapsed_time = time.perf_counter() - start_time

    generated_ids = output_ids[0][input_tokens:]

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return {
        "response": response,
        "input_tokens": int(input_tokens),
        "output_tokens": int(len(generated_ids)),
        "generation_time_seconds": elapsed_time,
    }


def get_completed_ids(output_path):
    completed_ids = set()

    if not Path(output_path).exists():
        return completed_ids

    with open(output_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                record = json.loads(line)
                completed_ids.add(record["problem_id"])
            except Exception:
                continue

    return completed_ids


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")
        file.flush()


def warm_up(model, tokenizer, question, device):
    print("Running warm-up generation...")

    generate_response(
        model=model,
        tokenizer=tokenizer,
        question=question,
        device=device,
        max_new_tokens=32,
    )

    torch.cuda.empty_cache()

def generate_batch(
    model,
    tokenizer,
    questions,
    device,
    max_new_tokens,
):
    prompts = []

    for question in questions:
        prompt = f"""Solve the following math problem.
Think step by step, then end your answer with:
Final answer: <number>

Problem:
{question}
"""

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        prompts.append(formatted_prompt)

    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)

    padded_input_length = inputs["input_ids"].shape[1]

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    batch_time = time.perf_counter() - start_time

    results = []

    for index in range(len(questions)):
        generated_ids = output_ids[index][
            padded_input_length:
        ]

        response = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        results.append({
            "response": response,
            "output_tokens": len(generated_ids),
        })

    return results, batch_time