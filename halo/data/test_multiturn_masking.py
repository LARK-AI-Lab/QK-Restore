#!/usr/bin/env python3
"""
Test script to verify multi-turn conversation masking.
Tests both single-turn and multi-turn scenarios.
"""
from transformers import AutoTokenizer

# Test data
test_cases = [
    {
        "name": "Single-turn conversation",
        "conversations": [
            {"from": "human", "value": "What is 2+2?"},
            {"from": "assistant", "value": "2+2 equals 4."},
        ]
    },
    {
        "name": "Multi-turn conversation",
        "conversations": [
            {"from": "human", "value": "What is 2+2?"},
            {"from": "assistant", "value": "2+2 equals 4."},
            {"from": "human", "value": "What about 3+3?"},
            {"from": "assistant", "value": "3+3 equals 6."},
        ]
    },
]

def build_messages(conversations):
    """Convert conversations to messages format."""
    messages = []
    for turn in conversations:
        role = "user" if turn["from"] == "human" else "assistant"
        messages.append({"role": role, "content": turn["value"]})
    return messages

def tokenize_fn(messages, tokenizer, max_length=512):
    """Tokenize and mask user turns."""
    # Apply chat template
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Tokenize full conversation
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
    )

    input_ids = tokenized["input_ids"]
    labels = input_ids.copy()

    # Mask all user turns
    current_messages = []
    for i, msg in enumerate(messages):
        current_messages.append(msg)

        # Tokenize up to this point
        partial_text = tokenizer.apply_chat_template(
            current_messages,
            tokenize=False,
            add_generation_prompt=(msg["role"] == "user"),
        )

        partial_ids = tokenizer(
            partial_text,
            truncation=True,
            max_length=max_length,
        )["input_ids"]

        # If this is a user turn, mask these tokens
        if msg["role"] == "user":
            n_mask = min(len(partial_ids), len(labels))
            labels[:n_mask] = [-100] * n_mask

    return {
        "input_ids": input_ids,
        "labels": labels,
        "full_text": full_text,
    }

def main():
    print("=" * 70)
    print("Multi-turn Conversation Masking Test")
    print("=" * 70)

    # Load tokenizer (use a small model for testing)
    tok_path = "/path/to/results/MixedData_hypenet_CoT_SFT/hypenet-9b-mix_old_selection_Miro/ckpt_100"

    print(f"\nLoading tokenizer from: {tok_path}")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    for test_case in test_cases:
        print("\n" + "=" * 70)
        print(f"Test: {test_case['name']}")
        print("=" * 70)

        conversations = test_case["conversations"]
        messages = build_messages(conversations)

        print(f"\nConversations ({len(conversations)} turns):")
        for i, conv in enumerate(conversations):
            print(f"  {i+1}. {conv['from']:10s}: {conv['value'][:50]}...")

        result = tokenize_fn(messages, tokenizer)

        input_ids = result["input_ids"]
        labels = result["labels"]
        full_text = result["full_text"]

        print(f"\nFull text:")
        print(f"  {full_text[:200]}...")

        print(f"\nTokenization:")
        print(f"  Total tokens: {len(input_ids)}")
        print(f"  Masked tokens (labels=-100): {sum(1 for l in labels if l == -100)}")
        print(f"  Unmasked tokens (assistant): {sum(1 for l in labels if l != -100)}")

        # Show token-by-token breakdown
        print(f"\nToken breakdown (first 50 tokens):")
        print(f"  {'Idx':<5} {'Token':<20} {'Label':<10} {'Status'}")
        print(f"  {'-'*5} {'-'*20} {'-'*10} {'-'*10}")

        for i in range(min(50, len(input_ids))):
            token = tokenizer.decode([input_ids[i]])
            label = labels[i]
            status = "MASKED" if label == -100 else "TRAIN"
            print(f"  {i:<5} {repr(token):<20} {label:<10} {status}")

        # Verify masking correctness
        print(f"\n✓ Verification:")
        masked_ratio = sum(1 for l in labels if l == -100) / len(labels)
        print(f"  Masked ratio: {masked_ratio:.1%}")

        if len(conversations) == 2:
            # Single-turn: expect ~30-50% masked (user turn + prompt)
            expected_range = (0.2, 0.6)
        else:
            # Multi-turn: depends on turn lengths
            expected_range = (0.1, 0.8)

        if expected_range[0] <= masked_ratio <= expected_range[1]:
            print(f"  ✓ PASS: Masked ratio within expected range {expected_range}")
        else:
            print(f"  ⚠️  WARNING: Masked ratio outside expected range {expected_range}")

    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70)

if __name__ == "__main__":
    main()
