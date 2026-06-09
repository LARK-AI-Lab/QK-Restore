import os
import random
from datasets import load_from_disk

def sample_arrow_dataset(
    input_path: str,
    output_path: str,
    num_samples: int,
    seed: int = 42,
):
    # Load dataset from Arrow folder
    dataset = load_from_disk(input_path)

    total_size = len(dataset)
    print(f"Total examples: {total_size}")

    if num_samples > total_size:
        raise ValueError("num_samples exceeds dataset size")

    # Reproducible sampling
    random.seed(seed)
    indices = random.sample(range(total_size), num_samples)

    # Select subset
    subset = dataset.select(indices)

    print(f"Subset size: {len(subset)}")

    # Save new Arrow dataset
    subset.save_to_disk(output_path)
    print(f"Saved subset to: {output_path}")


if __name__ == "__main__":
    sample_arrow_dataset(
        input_path="/path/to/data/fineweb-edu",
        output_path="/path/to/data/fineweb-edu-subset",
        num_samples=10000,  # change this
    )

# import os
# from datasets import load_from_disk

# def sample_by_token_count(
#     input_path: str,
#     output_path: str,
#     target_tokens: int = 320_000_000,
#     seed: int = 42,
#     token_column: str = "input_ids",
# ):
#     dataset = load_from_disk(input_path)
#     print(f"Total examples in dataset: {len(dataset)}")

#     # Shuffle once for uniform sampling
#     dataset = dataset.shuffle(seed=seed)

#     total_tokens = 0
#     selected_indices = []

#     for idx, example in enumerate(dataset):
#         n_tokens = len(example[token_column])
#         total_tokens += n_tokens
#         selected_indices.append(idx)

#         if idx % 10000 == 0 and idx > 0:
#             print(f"Processed {idx} examples | Tokens so far: {total_tokens:,}")

#         if total_tokens >= target_tokens:
#             break

#     print(f"\nFinal token count: {total_tokens:,}")
#     print(f"Selected examples: {len(selected_indices)}")

#     subset = dataset.select(selected_indices)
#     subset.save_to_disk(output_path)

#     print(f"Saved subset to: {output_path}")


# if __name__ == "__main__":
#     sample_by_token_count(
#         input_path="data/fineweb_edu_sample-10BT",
#         output_path="data/fineweb_edu_subset_320M",
#         target_tokens=320_000_000,
#     )