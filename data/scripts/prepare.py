import json
from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig


def load_raw_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert {"model", "attack", "generation"}.issubset(
        df.columns
    ), f"Unexpected columns in {path}"
    return df


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    # создаём столбец text
    df["text"] = df["generation"]

    # labels: human → 0, AI → 1
    df["labels"] = (df["model"] != "human").astype(int)

    return df[["text", "labels"]]


def save_json(df: pd.DataFrame, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            df.to_dict(orient="records"), f, ensure_ascii=False, indent=2
        )


@hydra.main(
    version_base="1.1", config_path="../../config", config_name="config"
)
def main(cfg: DictConfig):
    prep_cfg = cfg.data
    cwd = Path(get_original_cwd())

    print("Loading raw CSVs...")
    df_train = load_raw_csv(cwd / prep_cfg.raw_train)
    df_test = load_raw_csv(cwd / prep_cfg.raw_test)

    print("Preparing...")
    df_train_prep = prepare_df(df_train)
    df_test_prep = prepare_df(df_test)

    print("Saving...")
    save_json(df_train_prep, cwd / prep_cfg.processed_train)
    save_json(df_test_prep, cwd / prep_cfg.processed_test)

    print("Done!")
    print(f"Train samples: {len(df_train_prep)}")
    print(f"Test samples: {len(df_test_prep)}")


if __name__ == "__main__":
    main()
