import json
import random
import torch
from tqdm import tqdm
from collections import Counter
from safetensors import safe_open
from safetensors.torch import save_file
from tokenizers import Tokenizer, normalizers, pre_tokenizers, processors


def load_tokenizers():
    """Load and prepare tokenizer configurations from JSON files."""
    # Load tokenizer JSON files
    with open('tokenizer_nemo.json', 'r') as f:
        target_tokenizer_json = json.load(f)

    with open('mistral_tokenizer.json', 'r') as f:
        mistral_tokenizer_json = json.load(f)

    # Extract vocabularies
    mistral_vocabulary = mistral_tokenizer_json["model"]["vocab"]
    target_vocabulary = target_tokenizer_json["model"]["vocab"]

    # Create id-to-token mappings
    mistral_id_to_token = {v: k for k, v in mistral_vocabulary.items()}
    target_id_to_token = {v: k for k, v in target_vocabulary.items()}

    # Initialize tokenizers
    target_tokenizer = Tokenizer.from_file("tokenizer_nemo.json")
    mistral_tokenizer = Tokenizer.from_file("mistral_tokenizer.json")

    # Configure mistral tokenizer
    mistral_tokenizer.normalizer = normalizers.Sequence([])
    mistral_tokenizer.post_processor = processors.Sequence([])
    mistral_tokenizer.pre_tokenizer = pre_tokenizers.Sequence([])

    return (
        mistral_vocabulary,
        target_vocabulary,
        mistral_id_to_token,
        target_id_to_token,
        target_tokenizer,
        mistral_tokenizer
    )

def create_match_dictionary(target_vocabulary, mistral_tokenizer):
    """Create a mapping dictionary between target and source tokens."""
    match_dict = {}

    for subword, subword_id in tqdm(target_vocabulary.items()):
        # Handle special tokens
        if subword_id == 3:
            match_dict[subword_id] = [10]
            continue

        if subword_id < 5:
            match_dict[subword_id] = [subword_id]
            continue

        if subword_id < 16:
            match_dict[subword_id] = [14 + subword_id - 5]
            continue

        # Process regular tokens
        mistral_tokens = mistral_tokenizer.encode(subword).ids
        match_dict[subword_id] = mistral_tokens

    return match_dict

def analyze_matches(match_dict):
    """Analyze the distribution of token matches."""
    counter = Counter([0 if v is None else len(v) for v in match_dict.values()])
    print("Match length distribution:", counter.most_common())

def show_example_matches(target_id_to_token, mistral_id_to_token, match_dict, num_examples=10):
    """Display random examples of token matches."""
    print("\nRandom examples of matched subwords:")
    for _ in range(num_examples):
        i = random.randint(0, len(target_id_to_token) - 1)
        print(f"{target_id_to_token[i]} -> {[mistral_id_to_token[j] for j in match_dict[i]]}")

def process_embeddings(match_dict, target_vocabulary, model_path):
    """Process and update model embeddings based on token matches."""
    # Load model tensors
    with safe_open(model_path, framework="pt", device="cpu") as f:
        tensors = {key: f.get_tensor(key) for key in f.keys()}

    embedding = tensors["model.embed_tokens.weight"]
    dtype = embedding.dtype

    # Update embeddings
    for target_id, source_ids in tqdm(match_dict.items()):
        embedding[target_id] = torch.mean(embedding[source_ids].float(), dim=0).to(dtype)

    # Trim embedding to vocabulary size
    embedding = embedding[:len(target_vocabulary)].contiguous()
    tensors["model.embed_tokens.weight"] = embedding

    # Save updated tensors
    save_file(tensors, model_path)

def main():
    """Main execution function."""
    # Load and prepare tokenizers
    (
        mistral_vocabulary,
        target_vocabulary,
        mistral_id_to_token,
        target_id_to_token,
        target_tokenizer,
        mistral_tokenizer
    ) = load_tokenizers()

    # Create token matching dictionary
    match_dict = create_match_dictionary(target_vocabulary, mistral_tokenizer)

    # Analyze and display results
    analyze_matches(match_dict)
    show_example_matches(target_id_to_token, mistral_id_to_token, match_dict)

    # Process embeddings
    model_path = "model-00001-of-00005.safetensors"
    process_embeddings(match_dict, target_vocabulary, model_path)

if __name__ == "__main__":
    main()