#!/usr/bin/env python
"""
Script to convert a tokenizer from trust_remote_code=True to trust_remote_code=False
and validate that outputs are identical using the XNLI dataset.

Usage:
    python scripts/convert_tokenizer.py moonshotai/Kimi-K2.5
    python scripts/convert_tokenizer.py moonshotai/Kimi-K2.5 --push-to-hub
    python scripts/convert_tokenizer.py moonshotai/Kimi-K2.5 --save-dir ./converted_tokenizer
"""

import argparse
from datasets import load_dataset
from tqdm import tqdm

from transformers import AutoTokenizer


def validate_tokenizers(tok_original, tok_converted, dataset, num_samples=1000):
    """Validate that both tokenizers produce identical outputs on the dataset."""
    mismatches = []
    count = 0

    for i, example in enumerate(tqdm(dataset, total=num_samples, desc="Validating")):
        if count >= num_samples:
            break

        # XNLI has nested structure with translations - get English text
        premise = example.get("premise", {})
        hypothesis = example.get("hypothesis", {})

        # Handle both flat and nested structures
        if isinstance(premise, dict):
            premise = premise.get("en", premise.get("translation", [""])[0] if isinstance(premise.get("translation"), list) else "")
        if isinstance(hypothesis, dict):
            hypothesis = hypothesis.get("en", hypothesis.get("translation", [""])[0] if isinstance(hypothesis.get("translation"), list) else "")

        for field, text in [("premise", premise), ("hypothesis", hypothesis)]:
            if not text or not isinstance(text, str):
                continue

            ids_original = tok_original.encode(text)
            ids_converted = tok_converted.encode(text)

            if ids_original != ids_converted:
                mismatches.append({
                    "index": i,
                    "field": field,
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "original": ids_original[:20],
                    "converted": ids_converted[:20],
                })

        count += 1

    return mismatches


def main():
    parser = argparse.ArgumentParser(description="Convert and validate tokenizer")
    parser.add_argument("model_name", type=str, help="HuggingFace model name (e.g., moonshotai/Kimi-K2.5)")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save converted tokenizer")
    parser.add_argument("--push-to-hub", action="store_true", help="Push converted tokenizer to Hub as PR")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of XNLI samples to validate")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation step")
    args = parser.parse_args()

    print(f"Loading original tokenizer from {args.model_name} (trust_remote_code=True)...")
    tok_original = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    print(f"  Type: {type(tok_original).__name__}")
    print(f"  Vocab size: {tok_original.vocab_size}")

    print(f"\nLoading converted tokenizer from {args.model_name} (trust_remote_code=False)...")
    tok_converted = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=False)
    print(f"  Type: {type(tok_converted).__name__}")
    print(f"  Vocab size: {tok_converted.vocab_size}")

    if not args.skip_validation:
        print(f"\nLoading XNLI dataset for validation...")
        dataset = load_dataset("facebook/xnli", "all_languages", split="validation", trust_remote_code=True)

        print(f"Validating on {args.num_samples} samples...")
        mismatches = validate_tokenizers(tok_original, tok_converted, dataset, args.num_samples)

        if mismatches:
            print(f"\n❌ Found {len(mismatches)} mismatches!")
            for m in mismatches[:5]:  # Show first 5
                print(f"  Sample {m['index']} ({m['field']}): {m['text']}")
                print(f"    Original:  {m['original']}")
                print(f"    Converted: {m['converted']}")
            if len(mismatches) > 5:
                print(f"  ... and {len(mismatches) - 5} more")
            return 1
        else:
            print(f"\n✓ All {args.num_samples} samples match!")

    if args.save_dir:
        print(f"\nSaving converted tokenizer to {args.save_dir}...")
        tok_converted.save_pretrained(args.save_dir)
        print(f"  Saved!")

    if args.push_to_hub:
        print(f"\nPushing to Hub as PR...")
        result = tok_converted.push_to_hub(args.model_name, create_pr=True)
        print(f"  PR URL: {result.pr_url}")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
