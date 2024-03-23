import argparse
import random
import uuid
from statistics import mean, stdev
from typing import List
import subprocess
import torch
import torchmetrics
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="norallm/normistral-7b-warm"
    )  # Path to the pre-trained model
    parser.add_argument(
        "--n_shots", type=int, default=1
    )  # Number of random examples to sample
    parser.add_argument("--n_repetitions", type=int, default=5)  # Number of repetitions
    args = parser.parse_args()
    # If n_shots is 0, we don't need to account for random sampling of examples
    if args.n_shots == 0:
        args.n_repetitions = 1

    return args


def format_prompt(
    source_texts: List[str],
    target_texts: List[str]
):
    prompt_template = (
        "Tekst: {source_text}\nKorreksjon:{target_text}"
    )

    examples = [
        prompt_template.format(
            source_text=source_text,
            target_text=" " + target_text
            if i < len(source_texts) - 1
            else "",  # Add space before target text (except for the last example,
            # which should be empty)
        )
        for i, (source_text, target_text) in enumerate(zip(source_texts, target_texts))
    ]
    prompt = "\n\n".join(
        examples
    )  # Join the examples with two newlines (https://arxiv.org/abs/2302.01398)

    # Add an instructional prefix for 0-shot and 1-shot
    if len(source_texts) <= 2:
        prefix = "Her er eksempler på perfekt korrigering av grammatiske feil:"
        prompt = f"{prefix}\n\n{prompt}"

    return prompt


def load_model(model_path: str):
    # Load the pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda").eval()

    eos_token_id = tokenizer("\n").input_ids[-1]

    return {
        "name": model_path.split("/")[-1],
        "tokenizer": tokenizer,
        "model": model,
        "eos_token_id": eos_token_id,
    }


def load_data():
    dataset = load_dataset(
        "ltg/ask-gec",
        split="test",
    )

    # Deduplicate source texts in the dataset
    dataset = [
        {
            "source_text": sample["source"].strip(),
            "target_text": sample["correction"].strip(),
        }
        for sample in dataset
    ]

    return dataset


def sample_random_examples(dataset: List[dict], example_index: int, n_shots: int):
    # Sample n_shots different examples from the dataset (excluding the example at example_index)
    sequence = list(range(0, example_index)) + list(
        range(example_index + 1, len(dataset))
    )
    random_indices = random.sample(sequence, n_shots)
    return [dataset[j] for j in random_indices]


@torch.no_grad()
def generate(text: str, model: dict):
    # Generate text using the pre-trained model
    input_ids = model["tokenizer"](text, return_tensors="pt").input_ids.cuda()
    input_ids = input_ids[:, -(2048-256):]  # Truncate the input to 1024 tokens
    prediction = model["model"].generate(
        input_ids,
        max_new_tokens=256,
        num_beams=1,  # Greedy decoding
        do_sample=False,
        eos_token_id=model["eos_token_id"],
    )
    prediction = model["tokenizer"].decode(prediction[0, input_ids.size(1) :]).strip()

    # NB-GPT-J-specific post-processing because it doesn't want to generate newline tokens
    if "nb-gpt" in model["name"]:
        if ". o" in prediction:
            prediction = prediction.split(". o")[0] + "."
        elif "? o" in prediction:
            prediction = prediction.split("? o")[0] + "?"
        elif "! o" in prediction:
            prediction = prediction.split("! o")[0] + "!"

    return prediction


def main():
    args = parse_args()
    random.seed(42)

    model = load_model(args.model_name_or_path)
    dataset = load_data()

    log_file = open(
        f"eval_{args.model_name_or_path.split('/')[-1]}_{args.n_shots}-shots.txt",
        "w",
    )
    sources, targets, predictions = [], [], []
    f_scores = []

    for _ in range(args.n_repetitions):
        for i, example in enumerate(tqdm(dataset)):
            shots = sample_random_examples(dataset, i, args.n_shots)
            source_texts = [example["source_text"] for example in shots] + [
                example["source_text"]
            ]
            target_texts = [example["target_text"] for example in shots] + [
                example["target_text"]
            ]

            prompt = format_prompt(
                source_texts, target_texts
            )
            prediction = generate(prompt, model)

            if i < 5:
                print(f"Prompt:\n{prompt}\n")
                print(f"Target:\n{example['target_text']}\n")
                print(f"Prediction:\n{prediction}\n")

            sources.append(example["source_text"])
            targets.append(example["target_text"])
            predictions.append(prediction)


        # evaluate
            
        unique_tmp_name = uuid.uuid4().hex

        with open(f"tmp/{unique_tmp_name}_sources.txt", "w") as f:
            f.write("\n".join(sources))
        with open(f"tmp/{unique_tmp_name}_targets.txt", "w") as f:
            f.write("\n".join(targets))
        with open(f"tmp/{unique_tmp_name}_predictions.txt", "w") as f:
            f.write("\n".join(predictions))

        subprocess.run(
            f"export PATH=~/.local/bin:$PATH;errant_parallel -orig tmp/{unique_tmp_name}_sources.txt -cor tmp/{unique_tmp_name}_targets.txt -out tmp/{unique_tmp_name}_targets.m2 -lev -tok",
            shell=True
        )
        subprocess.run(
            f"export PATH=~/.local/bin:$PATH;errant_parallel -orig tmp/{unique_tmp_name}_sources.txt -cor tmp/{unique_tmp_name}_predictions.txt -out tmp/{unique_tmp_name}_predictions.m2 -lev -tok",
            shell=True
        )

        output = subprocess.check_output(
            f"export PATH=~/.local/bin:$PATH;errant_compare -ref tmp/{unique_tmp_name}_targets.m2 -hyp tmp/{unique_tmp_name}_predictions.m2",
            shell=True,
        )
        f_05 = float(output.decode().strip().split("\n")[-2].split()[-1].strip())
        f_scores.append(f_05)

        log_file.write(f"{f_05:.2%}\n")
        log_file.flush()

    log_file.write(
        f"F_{0.5} score: {mean(f_scores):.2%} ± {stdev(f_scores) if len(f_scores) > 1 else 0:.2%}\n"
    )
    log_file.close()

    subprocess.run(f"rm tmp/{unique_tmp_name}_*", shell=True)


if __name__ == "__main__":
    print("\nWARNING: make sure you have ERRANT installed to run the evaluation! Available here: https://github.com/chrisjbryant/errant\n\n", flush=True)
    main()
