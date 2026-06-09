from datasets import load_dataset, Dataset, concatenate_datasets

# ds1 = load_dataset("openbmb/UltraData-Math", "UltraData-Math-L3-Multi-Style-Synthetic")
# ds2 = load_dataset("openbmb/UltraData-Math", "UltraData-Math-L3-Textbook-Exercise-Synthetic")

# ds1 = ds1['train']
# ds2 = ds2['train']
# # Combine datasets
# combined = concatenate_datasets([ds1, ds2])

# # Save locally (Arrow format)
# combined.save_to_disk("/path/to/data/ultramath")

MAX_SAMPLES = 200000
SAVE_PATH = "/path/to/data/MiroMind_M1_200K"

dataset = load_dataset(
    "miromind-ai/MiroMind-M1-SFT-719K",
    split="train"
)

data = []

for item in dataset:

    problem = item["question"]
    generations = item["response"]
    # flags = item["is_reasoning_complete"]

    # for gen, flag in zip(generations, flags):

        # if flag:
    data.append({
        "problem": problem,
        "answer": generations
    })


    if len(data) >= MAX_SAMPLES:
        break


dataset_filtered = Dataset.from_list(data)

print("Collected samples:", len(dataset_filtered))

dataset_filtered.save_to_disk(SAVE_PATH)