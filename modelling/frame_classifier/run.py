import click
from model import (
    build_randomforest_model,
    build_mlp_classifier,
    get_data,
    show_permutation_imp,
)
import time

# run this file from the root folder!
#     python modelling/frame_classifier/run.py

base_agent_map_folder = "agent_maps"
normal_folder = ["nominal_recordings"]
anomalous_folders = [
    "oncoming_car_recordings",
    "tl_sl_recordings",
    "debris_avoidance_recordings",
]


@click.command()
@click.option("--test", is_flag=True)
def main(test):
    t0 = time.time()
    global normal_folder
    global anomalous_folders
    if test:
        normal_folder = [f"test_{r}" for r in normal_folder]
        anomalous_folders = [f"test_{r}" for r in anomalous_folders]

    X_train, X_test, y_train, y_test = get_data(
        normal_folders=normal_folder,
        anomalous_folders=anomalous_folders,
        agent_map_folder=base_agent_map_folder,
    )
    t1 = time.time()

    print("Time to get_data =", t1 - t0)
    print(
        "Dataset stats: ",
        "\n",
        f"X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}, ",
    )
    model, score = build_randomforest_model(
        normal_folders=normal_folder,
        anomalous_folders=anomalous_folders,
        agent_map_folder=base_agent_map_folder,
        data=(X_train, X_test, y_train, y_test),
    )
    print("RandomForest Results:", score)
    show_permutation_imp(model, X_train, y_train)
    show_permutation_imp(model, X_test, y_test)
    # breakpoint()


if __name__ == "__main__":
    main()
