import os
import torch
import neptune
import numpy as np
import nltk

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

from flask import Flask, request
from bert import MLBERT
from preprocess import preprocess
from tokenizer import tokenizer
from celery import shared_task
from utils import get_best_run, get_model_weights_name, download_model_weights
from celery_init_app import celery_init_app
from config import API_TOKEN, PROJECT_NAME, BROKER_URL

app = Flask(__name__)
app.config.from_mapping(
    CELERY=dict(
        broker_url=BROKER_URL,
        task_ignore_result=True,
    ),
)
celery_init_app(app)

categories = [
    "Beauty & Hygiene",
    "Kitchen",
    "Garden & Pets",
    "Cleaning & Household",
    "Gourmet & World Food",
    "Foodgrains",
    "Oil & Masala",
    "Snacks & Branded Foods",
    "Beverages",
    "Bakery",
    "Cakes & Dairy",
    "Baby Care",
    "Fruits & Vegetables",
    "Eggs",
    "Meat & Fish",
]

project = neptune.init_project(
    project=PROJECT_NAME,
    mode="read-only",
    api_token=API_TOKEN,
)

# On server initialize we are going to download
# model weights for default value
default_run_id = download_model_weights(project)


@shared_task(ignore_result=True)
def download_model_weights_task() -> None:
    download_model_weights(project)


@app.route("/predict", methods=["POST"])
def predict():
    best_run = get_best_run(project)
    best_run_id = best_run["sys/id"].fetch()
    best_model_params = best_run["config/parameters"].fetch()
    model = MLBERT(n_classes=best_model_params["num_classes"])

    download_model_weights_task.delay()

    if os.path.exists(f"{os.getcwd()}" + f"/{get_model_weights_name(best_run_id)}"):
        model.load_state_dict(
            torch.load(
                f"{os.getcwd()}" + f"/{get_model_weights_name(best_run_id)}",
                map_location=torch.device("cpu"),
            )
        )
    else:
        model.load_state_dict(
            torch.load(
                f"{os.getcwd()}" + f"/f{get_model_weights_name(default_run_id)}",
                map_location=torch.device("cpu"),
            )
        )

    text = preprocess(request.json["text"])

    inputs = tokenizer.encode_plus(
        text,
        None,
        truncation=True,
        add_special_tokens=True,
        max_length=best_model_params["max_len"],
        padding="max_length",
        return_token_type_ids=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]
    output = model(ids, mask, token_type_ids)
    output = torch.sigmoid(output).cpu().detach().numpy().round()
    output = np.array(output, dtype=np.int32).squeeze(0).tolist()
    result = [true for pred, true in zip(output, categories) if pred]
    return {"predicted_categories": result}
