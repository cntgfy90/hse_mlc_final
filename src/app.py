import os
import torch
import neptune
import numpy as np
from flask import Flask, request
from bert import MLBERT
from preprocess import preprocess
from tokenizer import tokenizer

app = Flask(__name__)

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
    project="stepangrigorov/test",
    mode="read-only",
    api_token="<API_TOKEN>",
)


@app.route("/predict", methods=["POST"])
def predict():
    best_run_id = project.fetch_runs_table(tag="best").to_pandas()["sys/id"][0]
    best_run = neptune.init_run(
        project="stepangrigorov/test",
        mode="read-only",
        with_id=best_run_id,
        api_token="<API_TOKEN>",
    )
    best_model_params = best_run["config/parameters"].fetch()
    model = MLBERT(n_classes=best_model_params["num_classes"])

    if not os.path.exists(f"{os.getcwd()}" + "/src/model_weights.bin"):
        best_run["model/weights"].download(
            destination=f"{os.getcwd()}" + "/src/model_weights.bin"
        )

    model.load_state_dict(
        torch.load(
            f"{os.getcwd()}" + "/src/model_weights.bin",
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
