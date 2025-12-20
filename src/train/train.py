import os
import logging
from pathlib import Path
import random
import numpy as np

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import wandb
import torch
from datasets import Dataset, load_dataset

from transformers import AutoTokenizer

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import sys
sys.path.append(str(Path(__file__).parent.parent))
from modules.module import HFClassifier


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset_from_path(path: str) -> Dataset:
    p = Path(path)
    if p.suffix == ".json":
        return Dataset.from_json(str(p))
    elif p.suffix == ".csv":
        return load_dataset("csv", data_files=str(p))["train"]
    else:
        raise ValueError(f"Unsupported format {p.suffix}")


@hydra.main(version_base="1.1", config_path="../../config", config_name="config")
def main(cfg: DictConfig):

    cwd = Path(get_original_cwd())
    log.info("Loaded config:\n" + OmegaConf.to_yaml(cfg))
    set_seed(cfg.train.seed)

    wandb_logger = None
    if cfg.logging.enabled:
        wandb_logger = WandbLogger(
            project=cfg.logging.project,
            entity=cfg.logging.entity,
            name=cfg.logging.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

    dataset = load_dataset_from_path(cwd / cfg.data.processed_train)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=cfg.model.max_length
        )

    dataset = dataset.map(tokenize_fn, batched=True)
    dataset = dataset.rename_column("generation", "text") if "generation" in dataset.column_names else dataset
    dataset = dataset.rename_column("model", "labels") if "model" in dataset.column_names else dataset

    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    split = dataset.train_test_split(test_size=cfg.train.test_size, seed=cfg.train.seed)
    train_ds, val_ds = split["train"], split["test"]

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=cfg.train.batch_size)

    model = HFClassifier(
        model_name=cfg.model.name,
        num_labels=cfg.model.num_labels,
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )

    out_dir = cwd / cfg.train.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="cpu",
        logger=wandb_logger,
        log_every_n_steps=cfg.train.eval_steps,
        default_root_dir=str(out_dir)
    )

    trainer.fit(model, train_dl, val_dl)

    model.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    if cfg.logging.enabled:
        wandb.finish()

    log.info("Training complete.")


if __name__ == "__main__":
    main()
