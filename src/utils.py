import neptune
import os
from config import API_TOKEN, PROJECT_NAME


def get_model_weights_name(id):
    return f"model_weights_{id}.bin"


def get_best_run(project):
    best_run_id = project.fetch_runs_table(tag="best").to_pandas()["sys/id"][0]
    best_run = neptune.init_run(
        project=PROJECT_NAME,
        mode="read-only",
        with_id=best_run_id,
        api_token=API_TOKEN,
    )
    return best_run


def download_model_weights(project) -> str:
    best_run = get_best_run(project)
    best_run_id = best_run["sys/id"].fetch()
    if not os.path.exists(f"{os.getcwd()}" + f"/{get_model_weights_name(best_run_id)}"):
        best_run["model/weights"].download(
            destination=f"{os.getcwd()}" + f"/{get_model_weights_name(best_run_id)}"
        )
    return best_run_id
