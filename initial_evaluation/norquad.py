import argparse
import random
import os
from statistics import mean, stdev
from typing import List

import torch
import torchmetrics
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Path to the pre-trained model")
    parser.add_argument("--n_shots", type=int, default=5, help="Number of examples to sample")
    parser.add_argument("--n_repetitions", type=int, default=5, help="Number of repetitions to average over")
    args = parser.parse_args()

    if args.n_shots == 0:
        args.n_repetitions = 1  # If n_shots is 0, we don't need to account for random sampling of examples

    return args


def format_prompt(samples):
    # Format the prompt, using prompt from the GPT3 paper (https://arxiv.org/abs/2005.14165, Figure G.28)
    prompt_template = "Tittel: {title}\n\nTekst: {text}\n\nSpørsmål: {question}\n\nSvar:{answer}"
    examples = [
        prompt_template.format(
            title=sample['title'],
            text=sample['text'],
            question=sample['question'],
            answer=' ' + sample['answer'] if i < len(samples) - 1 else ""  # Add space before target text (except for the last example, which should be empty)
        )
        for i, sample in enumerate(samples)
    ]
    prompt = '\n\n\n'.join(examples)  # Join the examples with two newlines

    return prompt


def load_model(model_path: str):
    # Load the pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).cuda().eval()
    if "gpt-sw3" in model_path:
        eos_token_id = tokenizer('\n').input_ids[-1]  # SentencePiece weirdness
    else:
        eos_token_id = tokenizer('\n').input_ids[0]  # The generation should stop when the model predicts the newline token

    if hasattr(model.config, "n_positions"):
        max_length = model.config.n_positions
    elif hasattr(model.config, "max_position_embeddings"):
        max_length = model.config.max_position_embeddings
    elif hasattr(model.config, "max_length"):
        max_length = model.config.max_length
    elif hasattr(model.config, "n_ctx"):
        max_length = model.config.n_ctx
    else:
        max_length = 2048  # Default value

    return {
        "name": model_path.split('/')[-1],
        "tokenizer": tokenizer,
        "model": model,
        "eos_token_id": eos_token_id,
        "max_length": max_length
    }


def load_data():
    dataset = load_dataset("ltg/norquad", split="test")
    dataset = [
        {
            "text": '\n'.join(sample["context"].strip().split('\n')[1:]).strip(),
            "title": sample["context"].strip().split('\n')[0].strip(),
            "question": ' '.join(sample["question"].strip().split()),
            "answer": ' '.join(sample["answers"]["text"][0].strip().split()),
        }
        for sample in dataset
    ]
    return dataset


@torch.no_grad()
def generate(model: dict, text: str):
    # Generate text using the pre-trained model
    input_ids = model["tokenizer"](text, return_tensors='pt').input_ids.cuda()
    input_ids = input_ids[:, -(model["max_length"] - 32):]  # Truncate input to 2048 tokens
    prediction = model["model"].generate(
        input_ids,
        max_new_tokens=32,
        num_beams=1,  # Greedy decoding
        do_sample=False,
        eos_token_id=model["eos_token_id"]
    )
    prediction = model["tokenizer"].decode(prediction[0, input_ids.size(1):]).strip()

    # NB-GPT-J-specific post-processing because it can't generate a newline token
    if "nb-gpt" in model["name"]:
        if ". o" in prediction:
            prediction = prediction.split(". o")[0] + "."
        elif "? o" in prediction:
            prediction = prediction.split("? o")[0] + "?"
        elif "! o" in prediction:
            prediction = prediction.split("! o")[0] + "!"

    return prediction


def sample_random_examples(dataset: List[dict], example_index: int, n_shots: int):
    # Sample n_shots different examples from the dataset (excluding the example at example_index)
    sequence = list(range(0, example_index)) + list(range(example_index + 1, len(dataset)))
    random_indices = random.sample(sequence, n_shots)
    return [dataset[j] for j in random_indices]


def main():
    args = parse_args()
    random.seed(42)

    model = load_model(args.model_name_or_path)
    dataset = load_data()

    log_file = open(f"eval_norquad_{model['name']}_{args.n_shots}-shots.txt", "w")

    f1_scores, em_scores = [], []
    for repetition in range(args.n_repetitions):
        squad_metric = torchmetrics.text.SQuAD()
        for i, sample in enumerate(tqdm(dataset)):
            shots = sample_random_examples(dataset, i, args.n_shots)
            examples = shots + [sample]

            input_text = format_prompt(examples)

            predicted_answer = generate(model, input_text)
            gold_answer = sample["answer"]

            squad_metric.update(
                {"prediction_text": predicted_answer, "id": f"{i}_{repetition}"},
                {"answers": {"answer_start": [0], "text": [gold_answer]}, "id": f"{i}_{repetition}"}
            )
        
        scores = squad_metric.compute()
        f1_scores.append(scores["f1"].item())
        em_scores.append(scores["exact_match"].item())

        print(f"F1: {scores['f1']}, EM: {scores['exact_match']}")
        log_file.write(f"{scores['f1']}\t{scores['exact_match']}\n")
        log_file.flush()

    log_file.write(f"\nMean F1: {mean(f1_scores)} ± {stdev(f1_scores) if len(f1_scores) > 1 else 0}\n")
    log_file.write(f"Mean EM: {mean(em_scores)} ± {stdev(em_scores) if len(em_scores) > 1 else 0}\n")
    log_file.close()


if __name__ == "__main__":
    main()
