#!/usr/bin/env python3
"""
Quick test: download only 100 samples to verify the pipeline works.
Use this before running the full 200K download.
"""
from datasets import load_dataset, Dataset

MAX_SAMPLES = 100
SAVE_PATH = "/path/to/data/openthoughts3_test_100"

print(f"Quick test: downloading {MAX_SAMPLES} samples from OpenThoughts3-1.2M...")
print(f"Target: {SAVE_PATH}")

dataset = load_dataset(
    "open-thoughts/OpenThoughts3-1.2M",
    split="train",
    streaming=True,
)

data = []

for idx, item in enumerate(dataset):
    conversations = item.get("conversations", [])

    if len(conversations) < 2:
        continue

    human_turns = [c["value"] for c in conversations if c["from"] == "human"]
    assistant_turns = [c["value"] for c in conversations if c["from"] == "assistant"]

    if not human_turns or not assistant_turns:
        continue

    data.append({
        "conversations": conversations,
        "problem": human_turns[0],
        "answer": assistant_turns[-1],
    })

    if len(data) >= MAX_SAMPLES:
        break

    if (len(data)) % 10 == 0:
        print(f"  Collected {len(data)} samples...")

print(f"\nCollected {len(data)} samples")

dataset_filtered = Dataset.from_list(data)
print(f"Saving to {SAVE_PATH}...")
dataset_filtered.save_to_disk(SAVE_PATH)

print("\n✓ Quick test completed!")
print(f"\nSample 0:")
print(f"  Problem: {data[0]['problem'][:100]}...")
print(f"  Answer:  {data[0]['answer'][:100]}...")
print(f"\nTo verify: python verify_openthoughts3.py")
print(f"(You'll need to update DATA_PATH in that script to: {SAVE_PATH})")
