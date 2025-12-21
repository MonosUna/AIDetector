from datasets import concatenate_datasets, load_dataset

raid = load_dataset("liamdugan/raid")
train = raid["train"]
reddit = train.filter(lambda x: x["domain"] == "reddit")
human_part = reddit.filter(lambda x: x["model"] == "human")
not_human_part = reddit.filter(lambda x: x["model"] != "human")
human_attacks = list(set(human_part["attack"]))
total_human_samples = 20000
samples_per_human_attack = total_human_samples // len(human_attacks)

balanced_human_train = []
balanced_human_test = []

for attack in human_attacks:
    attack_data = human_part.filter(lambda x: x["attack"] == attack).shuffle(
        seed=42
    )

    test_size = int(0.2 * samples_per_human_attack)
    train_size = samples_per_human_attack - test_size

    test_part = attack_data.select(range(test_size))
    train_part = attack_data.select(range(test_size, test_size + train_size))

    balanced_human_test.append(test_part)
    balanced_human_train.append(train_part)

human_test = concatenate_datasets(balanced_human_test)
human_train = concatenate_datasets(balanced_human_train)

non_human_models = list(set(not_human_part["model"]))
non_human_attacks = list(set(not_human_part["attack"]))

total_non_human_samples = len(human_test) + len(human_train)
samples_per_combo = total_non_human_samples // (
    len(non_human_models) * len(non_human_attacks)
)

balanced_non_human_train = []
balanced_non_human_test = []

for model in non_human_models:
    model_data = not_human_part.filter(lambda x: x["model"] == model)
    for attack in non_human_attacks:
        combo_data = model_data.filter(
            lambda x: x["attack"] == attack
        ).shuffle(seed=42)

        n_samples = min(samples_per_combo, len(combo_data))
        test_size = int(0.2 * n_samples)
        train_size = n_samples - test_size

        test_part = combo_data.select(range(test_size))
        train_part = combo_data.select(
            range(test_size, test_size + train_size)
        )

        balanced_non_human_test.append(test_part)
        balanced_non_human_train.append(train_part)

test_models = concatenate_datasets(balanced_non_human_test)
train_models = concatenate_datasets(balanced_non_human_train)

test = concatenate_datasets([human_test] + balanced_non_human_test)
train = concatenate_datasets([human_train] + balanced_non_human_train)

print(
    "Общий размер датасета test:", len(test)
)  # Общий размер датасета test: 7956
print(
    "Общий размер датасета: train", len(train)
)  # Общий размер датасета: train 31968

final_train_df = train.to_pandas()
final_test_df = test.to_pandas()

final_train_df = final_train_df[["model", "attack", "generation"]]
final_test_df = final_test_df[["model", "attack", "generation"]]

final_train_df.to_csv("train.csv", index=False)
final_test_df.to_csv("test.csv", index=False)
