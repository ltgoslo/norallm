import argparse
import random
from statistics import mean, stdev
from typing import List
import torch
import torchmetrics
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default=None
    )  # Path to the pre-trained model
    parser.add_argument(
        "--source_language", type=str, default="en"
    )  # Source language code
    parser.add_argument(
        "--target_language", type=str, default="nb"
    )  # Target language code
    parser.add_argument(
        "--n_shots", type=int, default=5
    )  # Number of random examples to sample
    parser.add_argument("--n_repetitions", type=int, default=5)  # Number of repetitions
    args = parser.parse_args()
    # If n_shots is 0, we don't need to account for random sampling of examples
    if args.n_shots == 0:
        args.n_repetitions = 1

    return args


def format_prompt(
    source_texts: List[str],
    target_texts: List[List[str]],
    source_language: str,
    target_language: str,
):
    # Format the prompt with source and target texts
    source_language = {"en": "Engelsk", "nb": "Bokmål", "nn": "Nynorsk"}[
        source_language
    ]
    target_language = {"en": "Engelsk", "nb": "Bokmål", "nn": "Nynorsk"}[
        target_language
    ]

    # As in "The unreasonable effectiveness of few-shot learning for machine translation"
    # (https://arxiv.org/abs/2302.01398)
    prompt_template = (
        "{source_language}: {source_text}\n{target_language}:{target_text}"
    )

    examples = [
        prompt_template.format(
            source_language=source_language,
            target_language=target_language,
            source_text=source_text,
            target_text=" " + target_text[0]
            if i < len(source_texts) - 1
            else "",  # Add space before target text (except for the last example,
            # which should be empty)
        )
        for i, (source_text, target_text) in enumerate(zip(source_texts, target_texts))
    ]
    prompt = "\n\n".join(
        examples
    )  # Join the examples with two newlines (https://arxiv.org/abs/2302.01398)

    return prompt


def load_model(model_path: str):
    # Load the pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda").eval()

    if "gpt-sw3" in model_path:
        eos_token_id = tokenizer("\n").input_ids[-1]  # SentencePiece weirdness
    else:
        eos_token_id = tokenizer("\n").input_ids[
            0
        ]  # The generation should stop when the model predicts the newline token

    return {
        "name": model_path.split("/")[-1],
        "tokenizer": tokenizer,
        "model": model,
        "eos_token_id": eos_token_id,
    }


def load_data(source_language: str, target_language: str, reverse: bool = False):
    # Load the dataset for the given source and target languages
    source_language_id = {"en": "eng", "nb": "nob", "nn": "nno"}[source_language]
    target_language_id = {"en": "eng", "nb": "nob", "nn": "nno"}[target_language]

    try:
        dataset = load_dataset(
            "Helsinki-NLP/tatoeba_mt",
            f"{source_language_id}-{target_language_id}",
            split="test",
        )
    except ValueError:
        return load_data(target_language, source_language, reverse=True)

    # Deduplicate source texts in the dataset
    deduplicated_dataset = {}
    for sample in dataset:
        source_text = " ".join(
            sample["sourceString" if not reverse else "targetString"].strip().split()
        )  # Remove leading/trailing whitespace and normalize whitespace
        target_text = " ".join(
            sample["targetString" if not reverse else "sourceString"].strip().split()
        )  # Remove leading/trailing whitespace and normalize whitespace
        if source_text not in deduplicated_dataset:
            deduplicated_dataset[source_text] = [target_text]
        elif target_text not in deduplicated_dataset[source_text]:
            deduplicated_dataset[source_text].append(target_text)

    deduplicated_dataset = sorted(
        deduplicated_dataset.items(), key=lambda x: (x[0], x[1])
    )
    deduplicated_dataset = [
        {"source_text": source_text, "target_texts": target_texts}
        for source_text, target_texts in deduplicated_dataset
    ]

    return deduplicated_dataset


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
    dataset = load_data(args.source_language, args.target_language)

    log_file = open(
        f"eval_{args.source_language}_{args.target_language}_{args.model_name_or_path.split('/')[-1]}_{args.n_shots}-shots.txt",
        "w",
    )

    bleu_scores, bert_scores, chrf_scores = [], [], []
    for _ in range(args.n_repetitions):
        bleu_score = torchmetrics.text.SacreBLEUScore()
        chrf_score = torchmetrics.text.CHRFScore()
        bert_score = torchmetrics.text.BERTScore(
            "bert-base-multilingual-cased", num_layers=9
        )  # set according to
        # https://docs.google.com/spreadsheets/d/1RKOVpselB98Nnh_EOC4A2BYn8_201tmPODpNWu4w7xI/edit#gid=0

        for i, example in enumerate(tqdm(dataset)):
            shots = sample_random_examples(dataset, i, args.n_shots)
            source_texts = [example["source_text"] for example in shots] + [
                example["source_text"]
            ]
            target_texts = [example["target_texts"] for example in shots] + [
                example["target_texts"]
            ]

            prompt = format_prompt(
                source_texts, target_texts, args.source_language, args.target_language
            )
            prediction = generate(prompt, model)

            bleu_score.update([prediction], [example["target_texts"]])
            chrf_score.update([prediction], [example["target_texts"]])
            bert_score.update([prediction], [example["target_texts"][0]])

            if i == 0:
                print(f"Prompt:\n{prompt}\n")
                print(f"Target:\n{example['target_texts'][0]}\n")
                print(f"Prediction:\n{prediction}\n")

        bleu_scores.append(bleu_score.compute().item())
        chrf_scores.append(chrf_score.compute().item())
        bert_scores.append(bert_score.compute()["f1"].mean().item())

        print(f"BLEU score: {bleu_scores[-1]:.2f}")
        print(f"CHRF score: {chrf_scores[-1]:.2f}")
        print(f"BERT score: {bert_scores[-1]:.2f}\n")
        log_file.write(f"{bleu_scores[-1]}\t{bert_scores[-1]}\n")
        log_file.flush()

    log_file.write(
        f"\nBLEU score: {mean(bleu_scores)} ± {stdev(bleu_scores) if len(bleu_scores) > 1 else 0}\n"
    )
    log_file.write(
        f"CHRF score: {mean(chrf_scores)} ± {stdev(chrf_scores) if len(chrf_scores) > 1 else 0}\n"
    )
    log_file.write(
        f"BERT score: {mean(bert_scores)} ± {stdev(bert_scores) if len(bert_scores) > 1 else 0}"
    )

    log_file.close()


if __name__ == "__main__":
    main()
