from model import (
    build_randomforest_model,
    build_mlp_classifier,
    get_data,
    show_permutation_imp,
)

# run this file from the root folder!
#     python modelling/frame_classifier/run.py

base_agent_map_folder = "agent_maps"
normal_folder = ["normal_recordings"]
anomalous_folders = [
    "oncoming_car_recordings",
    "tl_sl_recordings",
    "debris_avoidance_recordings",
]

X_train, X_test, y_train, y_test = get_data(
    normal_folders=normal_folder,
    anomalous_folders=anomalous_folders,
    agent_map_folder=base_agent_map_folder,
)
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
