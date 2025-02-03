import argparse
import json
from tqdm import tqdm
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers import Tokenizer, pre_tokenizers, decoders, normalizers, Regex, processors


def initialize_tokenizer(args):
    start_of_text_symbol = "<s>"
    end_of_text_symbol = "</s>"
    unk_symbol = "<unk>"
    mask_symbol = "<mask>"
    pad_symbol = "<pad>"

    special_tokens = [unk_symbol, start_of_text_symbol, end_of_text_symbol, pad_symbol, mask_symbol]
    special_tokens += [f"<special_{i}>" for i in range(11)]

    tokenizer = Tokenizer(BPE(
        unk_token=unk_symbol,
        byte_fallback=False,
        fuse_unk=False,
        ignore_merges=True
    ))

    tokenizer.normalizer = normalizers.Sequence([
        normalizers.Prepend(" "),
        normalizers.NFKC(),
        normalizers.Replace(Regex("\n"), '\n '),
        normalizers.Replace(Regex(" *\n"), '\n'),
    ])

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(
            Regex("[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"),
            behavior="isolated",
            invert=False
        ),
        pre_tokenizers.ByteLevel(
            add_prefix_space=False, use_regex=False, trim_offsets=True
        ),
        pre_tokenizers.Split(
            Regex(".{1,24}"),
            behavior="isolated",
            invert=False
        )
    ])

    tokenizer.decoder = decoders.Sequence([
        decoders.ByteLevel(add_prefix_space=False, use_regex=False),
        decoders.Strip(' ', 1, 0),
        decoders.Replace("\n ", "\n")
    ])

    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{start_of_text_symbol} $A",
        pair=f"{start_of_text_symbol} $A {start_of_text_symbol} $B",
        special_tokens=[
            (start_of_text_symbol, 1),
        ]
    )

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True
    )

    return tokenizer, trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BERT sharding')
    parser.add_argument('--validation_path', type=str, default="../data/dev.jsonl", help='Specify the validation filename')
    parser.add_argument('--vocab_path', type=str, default="tokenizer_nemo_faroese_reduced_2.json", help='Specify the output filename')
    parser.add_argument('--vocab_size', type=int, default=51_200, help='Number of subwords in the trained tokenizer')
    args = parser.parse_args()

    print(f"Initializing a BPE tokenizer", flush=True)
    tokenizer, trainer = initialize_tokenizer(args)

    reducer = 1

    def iterator(paths):
        for path, (divider, multiplier) in tqdm(paths.items()):
            divider *= reducer
            for i, document in enumerate(open(path, mode='rt')):
                if i % divider != 0:
                    continue

                try:
                    document = json.loads(document)
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON: {document}")
                    continue

                if not isinstance(document, dict):
                    text = document
                else:
                    text = document["text"]

                text = text.strip()
                if text is None or len(text) == 0:
                    continue

                for _ in range(multiplier):
                    yield text

    paths = {
        "/pfs/lustrep1/scratch/project_465000144/dasamuel/normistral/norwegian/bokmaal.jsonl": (10 * 2, 1),
        "/pfs/lustrep1/scratch/project_465000144/dasamuel/normistral/norwegian/nynorsk.jsonl": (1 * 2, 1),
        "/pfs/lustrep1/scratch/project_465000144/dasamuel/looped-lm/culturax/sv.jsonl": (36 * 2, 1),
        "/pfs/lustrep1/scratch/project_465000144/dasamuel/looped-lm/culturax/da.jsonl": (20 * 2, 1),
        "/pfs/lustrep1/scratch/project_465000144/dasamuel/looped-lm/culturax/is.jsonl": (37 * 2, 1),
        "/pfs/lustrep1/scratch/project_465000144/dasamuel/normistral/faroese/all.jsonl": (16 * 2, 1),
        "/pfs/lustrep1/scratch/project_465000144/dasamuel/normistral/sami/all.jsonl": (1, 1),
        "/pfs/lustrep1/scratch/project_465000144/dasamuel/normistral/code/stack.jsonl": (100 * 2, 1),
        "/pfs/lustrep1/scratch/project_465000144/dasamuel/normistral/english/fineweb.jsonl": (30 * 2, 1),
    }

    tokenizer.train_from_iterator(iterator(paths), trainer)

    print("Saving the tokenizer", flush=True)
    tokenizer.save(args.vocab_path)

    print("TEST")
    print("Trying to load the tokenizer...")
    tokenizer = Tokenizer.from_file(args.vocab_path)
    print("Success!")

    print("Samples from the tokenizer:")

    def test(tokenizer, text):
        subwords = tokenizer.encode(text).tokens
        return ' '.join(subwords)

    texts = [
        """One of the most impressive long term hobby projects is Robert's Rocket Project. He started building a 100 lbf liquid engine in 2001, fired a regeneratively cooled version in 2007, started building a regen 250 lbf in 2008.""",
        """what are examples of interfaces that allow you to manage sets of queries (SQL, splunk, lucene/elastic, xpath, whatever other language)?""",
        """### Increasingly seeing a big schism between what I think my research is & what others think it is. I don't do qualitative work and I'm not trained in anthro or theories of race or gender. I can't supervise students with these interests!\n\nI'm a sociophonetician who works on prosody!""",
        """The Northern Lights season is here... Taking these pictures is an art itself and requires preparation, so The Local spoke to an expert to find out how to take awe-inspiring snaps of the Northern Lights.""",
        """Some people have SOTA facial recognition abilities: "At the very upper end of the performance scale, a cohort of just 1-2% of the population are 'super-recognisers'-people who can memorise and recall unfamiliar faces, even after the briefest glimpse.\""""
    ]

    for text in texts:
        print(f"INPUT:  {text}\nTOKENS: {test(tokenizer, text)}\n", flush=True)
