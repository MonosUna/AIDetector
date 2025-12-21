import logging
from pathlib import Path

import hydra
import numpy as np
import tritonclient.http as httpclient
import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from transformers import AutoTokenizer

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@hydra.main(
    version_base="1.1",
    config_path="../../config",
    config_name="config",
)
def main(cfg: DictConfig):

    cwd = Path(get_original_cwd())

    tokenizer = AutoTokenizer.from_pretrained(cwd / cfg.test.model_dir)

    client = httpclient.InferenceServerClient(url=cfg.web.url)

    app = FastAPI()
    templates = Jinja2Templates(
        directory=str(Path(__file__).parent / "templates")
    )

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    @app.post("/predict", response_class=HTMLResponse)
    async def predict(request: Request, text: str = Form(...)):
        inputs = tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            padding="max_length",
            max_length=cfg.model.max_length,
        )

        input_ids = inputs["input_ids"].astype(np.int64)
        attention_mask = inputs["attention_mask"].astype(np.int64)

        triton_inputs = [
            httpclient.InferInput("input_ids", input_ids.shape, "INT64"),
            httpclient.InferInput(
                "attention_mask", attention_mask.shape, "INT64"
            ),
        ]

        triton_inputs[0].set_data_from_numpy(input_ids)
        triton_inputs[1].set_data_from_numpy(attention_mask)

        outputs = [httpclient.InferRequestedOutput("logits")]

        response = client.infer(
            model_name=cfg.web.model_name,
            model_version=cfg.web.model_version,
            inputs=triton_inputs,
            outputs=outputs,
        )

        logits = response.as_numpy("logits")
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        ai_prob = float(probs[0, 1])

        return templates.TemplateResponse(
            "index.html",
            {"request": request, "result": {"probability": f"{ai_prob:.3f}"}},
        )

    uvicorn.run(app, host=cfg.web.host, port=cfg.web.port, log_level="info")


if __name__ == "__main__":
    main()
