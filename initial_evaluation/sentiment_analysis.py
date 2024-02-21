import argparse
import random
from statistics import mean, stdev
from typing import List
from datasets import load_dataset
from sklearn import metrics
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Path to the pre-trained model",
    )
    parser.add_argument(
        "--n_shots", type=int, default=5, help="Number of examples to sample"
    )
    parser.add_argument(
        "--n_repetitions",
        type=int,
        default=5,
        help="Number of repetitions to average over",
    )
    args = parser.parse_args()
    # If n_shots is 0, we don't need to account for random sampling of examples
    if args.n_shots == 0:
        args.n_repetitions = 1

    return args


def format_prompt(input_texts: List[str], labels: List[str]):
    # Format the prompt
    prompt_template = "Tekst: {text}\nSentiment:{label}"
    examples = [
        prompt_template.format(
            text=text,
            label=" " + label if i < len(input_texts) - 1 else ""
            # Add space before target text (except for the last example, which should be empty)
        )
        for i, (text, label) in enumerate(zip(input_texts, labels))
    ]
    prompt = "\n\n".join(examples)  # Join the examples with two newlines

    return prompt


def load_model(model_path: str):
    # Load the pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).cuda().eval()

    labels = (
        ["positiv", "negativ"] if "sw3" in model_path else [" positiv", " negativ"]
    )  # Add space before positive and negative for the sentencepiece models
    labels = [tokenizer(el, return_tensors="pt").input_ids for el in labels]
    positive_id = labels[0][0, 0].item()
    negative_id = labels[1][0, 0].item()

    return {
        "tokenizer": tokenizer,
        "model": model,
        "positive_id": positive_id,
        "negative_id": negative_id,
    }


def load_data():
    dataset = load_dataset("ltg/norec_sentence", "ternary", split="test")
    dataset = [
        {
            "text": sample["review"],
            "label": "positiv" if sample["sentiment"] == 2 else "negativ",
        }
        for sample in dataset
        if sample["sentiment"] != 1  # Exclude neutral samples
    ]
    return dataset


def predict(model, input_text):
    input_ids = model["tokenizer"](input_text, return_tensors="pt").input_ids.cuda()
    input_ids = input_ids[:, -2047:]

    # perform inference to obtain logits
    output = model["model"].generate(
        input_ids,
        max_new_tokens=1,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
    )

    # get logits
    logits = output.scores[-1]
    positive_logits = logits[0, model["positive_id"]].item()
    negative_logits = logits[0, model["negative_id"]].item()

    if positive_logits > negative_logits:
        return "positiv"
    else:
        return "negativ"


def sample_random_examples(dataset: List[dict], example_index: int, n_shots: int):
    # Sample n_shots different examples from the dataset (excluding the example at example_index)
    sequence = list(range(0, example_index)) + list(
        range(example_index + 1, len(dataset))
    )
    random_indices = random.sample(sequence, n_shots)
    return [dataset[j] for j in random_indices]


def main():
    args = parse_args()
    random.seed(42)

    model = load_model(args.model_name_or_path)
    dataset = load_data()

    log_file = open(
        f"eval_sa_{args.model_name_or_path.split('/')[-1]}_{args.n_shots}-shots.txt",
        "w",
    )

    f1_scores = []
    for _ in range(args.n_repetitions):
        gold_labels, predictions = [], []
        for i, sample in enumerate(tqdm(dataset)):
            shots = sample_random_examples(dataset, i, args.n_shots)
            examples = shots + [sample]

            input_text = format_prompt(
                input_texts=[s["text"] for s in examples],
                labels=[s["label"] for s in examples],
            )

            predicted_answer = predict(model, input_text)
            predictions.append(predicted_answer)

            gold_answer = sample["label"]
            gold_labels.append(gold_answer)

        # Calculate metrics
        macro_f1 = metrics.f1_score(gold_labels, predictions, average="macro")
        f1_scores.append(macro_f1)

        print(f"Macro F1: {macro_f1:.4f}")
        log_file.write(f"{macro_f1}\n")
        log_file.flush()

    log_file.write(
        f"\nMean macro F1: {mean(f1_scores)} ± {stdev(f1_scores) if len(f1_scores) > 1 else 0}\n"
    )
    log_file.close()


if __name__ == "__main__":
    main()
