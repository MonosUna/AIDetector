import logging
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from datasets import Dataset, load_dataset
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_dataset_from_path(path: Path) -> Dataset:
    if path.suffix == ".csv":
        return load_dataset("csv", data_files=str(path))["train"]
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


@hydra.main(
    version_base="1.1", config_path="../../config", config_name="config"
)
def main(cfg: DictConfig):

    cwd = Path(get_original_cwd())
    log.info("Loaded config:\n" + OmegaConf.to_yaml(cfg))

    device = torch.device("cpu")

    if cfg.logging.enabled:
        wandb.init(
            project=cfg.logging.project,
            entity=cfg.logging.entity,
            name=cfg.logging.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    model_dir = cwd / cfg.test.model_dir

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()

    dataset = load_dataset_from_path(cwd / cfg.data.processed_test)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=cfg.model.max_length,
        )

    dataset = dataset.map(tokenize_fn, batched=True)
    dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    dataloader = DataLoader(
        dataset, batch_size=cfg.test.batch_size, shuffle=False
    )

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    results = {}

    if cfg.metrics.accuracy:
        results["accuracy"] = accuracy_score(y_true, y_pred)

    if cfg.metrics.f1:
        results["f1"] = f1_score(y_true, y_pred)

    if cfg.metrics.roc_auc:
        results["roc_auc"] = roc_auc_score(y_true, y_pred)

    log.info("Inference metrics:")
    for k, v in results.items():
        log.info(f"{k}: {v:.4f}")

    if cfg.logging.enabled:
        wandb.log(results)
        wandb.finish()


if __name__ == "__main__":
    main()
