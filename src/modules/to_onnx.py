import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig


@hydra.main(version_base="1.1", config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    cwd = Path(get_original_cwd())

    model_dir = cwd / cfg.train.output_dir
    output_path = cwd / cfg.onnx.output_path
    max_length = cfg.model.max_length

    print(f"Starting ONNX export")
    print(f"Model dir: {model_dir}")
    print(f"Output path: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, attn_implementation="eager")
    model.eval()

    dummy = tokenizer(
        "dummy text",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    torch.onnx.export(
        model,
        (
            dummy["input_ids"],
            dummy["attention_mask"],
        ),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=13,
        dynamo=False,
    )

    print(f"ONNX model saved to {output_path}")


if __name__ == "__main__":
    main()
