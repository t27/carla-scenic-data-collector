import pandas as pd
import os
import glob
import click
from tqdm import tqdm
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
import matplotlib.pyplot as plt


class ScenarioFrameDataprep:
    def __init__(self, recording_folder, useful_cols) -> None:
        self.recording_folder = recording_folder
        self.round_names = self._get_round_names(recording_folder)
        self.useful_cols = useful_cols

    def _get_round_names(self, recording_folder):
        rounds = glob.glob(f"{recording_folder}/*.csv")
        round_names = [os.path.split(round)[-1].replace(".csv", "") for round in rounds]
        return round_names

    def get_per_round_df(self):
        dataframe_list = []
        print(f"Processing {len(self.round_names)} rounds for {self.recording_folder}")
        for round_name in tqdm(self.round_names):
            df = pd.read_csv(os.path.join(self.recording_folder, round_name) + ".csv")
            df.index.name = "index"
            vehicles = df.query('type_id.str.contains("vehicle")', engine="python")
            frame_wise_idxs = vehicles.groupby("frame_id").groups
            result_df = self._get_stats_dataframe(df, frame_wise_idxs)
            result_df["round_name"] = round_name
            dataframe_list.append(result_df)
        return dataframe_list

    def get_full_scenario_df(self):
        dataframe_list = self.get_per_round_df()
        scenario_dataframe = pd.concat(dataframe_list)
        return scenario_dataframe

    def _get_stats_dataframe(self, df, frame_wise_idxs):

        frame_results = []
        for key in frame_wise_idxs:
            # for each frame, get the min/max values of the above entries
            idxs = frame_wise_idxs[key]
            frame_df = df.iloc[idxs]
            frame_stats = [
                frame_df.velocity_x.max(),
                frame_df.velocity_y.max(),
                frame_df.velocity_z.max(),
                frame_df.velocity_x.min(),
                frame_df.velocity_y.min(),
                frame_df.velocity_z.min(),
                frame_df.angular_vel_x.max(),
                frame_df.angular_vel_y.max(),
                frame_df.angular_vel_z.max(),
                frame_df.angular_vel_x.min(),
                frame_df.angular_vel_y.min(),
                frame_df.angular_vel_z.min(),
                frame_df.acc_x.max(),
                frame_df.acc_y.max(),
                frame_df.acc_z.max(),
                frame_df.acc_x.min(),
                frame_df.acc_y.min(),
                frame_df.acc_z.min(),
            ]
            frame_results.append(frame_stats)
        return pd.DataFrame(frame_results, columns=self.useful_cols)


def dataprep(normal_folder, anomalous_folders, useful_cols, val_ratio):
    normal = []
    anomalous = []

    for rec in normal_folder:
        sfc = ScenarioFrameDataprep(rec, useful_cols)
        df = sfc.get_full_scenario_df()
        print(f"Found {len(df)} frames for {rec}")
        normal.append(df)

    for rec in anomalous_folders:
        sfc = ScenarioFrameDataprep(rec, useful_cols)
        df = sfc.get_full_scenario_df()
        anomalous.append(df)
        print(f"Found {len(df)} frames for {rec}")

    normal = pd.concat(normal)
    anomalous = pd.concat(anomalous)

    X = pd.concat([normal[useful_cols], anomalous[useful_cols]])
    y = np.hstack([np.zeros(normal.shape[0]), np.ones(anomalous.shape[0])])

    if val_ratio == 0:
        return X, None, y, None
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_ratio, random_state=42
        )

        return X_train, X_val, y_train, y_val


def evaluate_dataframe(model, dataframe_list, useful_cols, is_anomalous=False):
    """Evaluate a list of dataframes using the given frame classifier model and a majority vote operation
    Here, one dataframe represents one round.
    For each round, we evaluate each row. If majority rows are True(anomaly positive), then we mark that round as anomalousdatetime A combination of a date and a time. Attributes: ()

    Args:
        model (): Classifier model
        dataframe_list ([list]): list of pd.DataFrames, containing one dataframe per round of data
        is_anomalous (bool, optional): GT Value for the current list of dataframes. Defaults to False.
    """
    result = []
    for df in dataframe_list:
        X = df[useful_cols]
        if is_anomalous:
            y = np.ones(df.shape[0])
        else:
            y = np.zeros(df.shape[0])
        score = model.score(X, y)
        # result.append(score)
        if (
            score > 0.8
        ):  # if accuracy is above 50% we have classified correctly, hence return True for the count
            result.append(True)
        else:
            result.append(False)

    return result


def get_test_scores(model, normal_folder, anomalous_folders, useful_cols):
    normal_results = []
    for rec in normal_folder:
        sfc = ScenarioFrameDataprep(rec, useful_cols)
        dfs = sfc.get_per_round_df()
        temp_results = evaluate_dataframe(model, dfs, useful_cols, is_anomalous=False)
        normal_accuracy = sum(temp_results) / len(temp_results)
        print(f"The model classifies {rec} with {normal_accuracy} accuracy")
        normal_results += temp_results

    anomalous_results = []
    for rec in anomalous_folders:
        sfc = ScenarioFrameDataprep(rec, useful_cols)
        dfs = sfc.get_per_round_df()
        temp_results = evaluate_dataframe(model, dfs, useful_cols, is_anomalous=True)
        anomalous_accuracy = sum(temp_results) / len(temp_results)
        print(f"The model classifies {rec} with {anomalous_accuracy} accuracy")
        anomalous_results += temp_results

    # normal_results = np.array(normal_results)
    # anomalous_results = np.array(anomalous_results)
    # breakpoint()
    # predictions = np.hstack([normal_results, anomalous_results])
    # y_test = np.hstack([np.zeros_like(normal_results), np.ones_like(anomalous_results)])

    # precision, recall, _ = precision_recall_curve(y_test, predictions)
    # disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    # disp.plot()

    normal_accuracy = sum(normal_results) / len(normal_results)
    anomalous_accuracy = sum(anomalous_results) / len(anomalous_results)
    print(f"Overall Test results")
    print(f"The model classifies normal scenarios with {normal_accuracy} accuracy")
    print(
        f"The model classifies anomalous scenarios with {anomalous_accuracy} accuracy"
    )
    # breakpoint()


def show_permutation_imp(RF, X, y):
    """Show the permutation importance plot for the train/test set

    Args:
        dataset_type (str, optional): This can be "train" or "test". 
            Runs the permutation importance for either set. Defaults to "train".
    """
    result = permutation_importance(RF, X, y, n_repeats=10, random_state=42, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()

    fig, ax = plt.subplots()
    ax.boxplot(
        result.importances[sorted_idx].T, vert=False, labels=X.columns[sorted_idx]
    )
    ax.set_title("Permutation Importances")
    fig.tight_layout()
    plt.show()


def main():
    val_ratio = 0.3
    useful_cols = [
        "max_velocity_x",
        "max_velocity_y",
        "max_velocity_z",
        "min_velocity_x",
        "min_velocity_y",
        "min_velocity_z",
        "max_ang_velocity_x",
        "max_ang_velocity_y",
        "max_ang_velocity_z",
        "min_ang_velocity_x",
        "min_ang_velocity_y",
        "min_ang_velocity_z",
        "max_acc_x",
        "max_acc_y",
        "max_acc_z",
        "min_acc_x",
        "min_acc_y",
        "min_acc_z",
    ]
    normal_folder = ["nominal_recordings"]
    anomalous_folders = [
        "oncoming_car_recordings",
        "tl_sl_recordings",
        "debris_avoidance_recordings",
    ]

    X_train, X_val, y_train, y_val = dataprep(
        normal_folder, anomalous_folders, useful_cols, val_ratio
    )
    print("Training Random Forest Model for frames")
    RF = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=0)
    RF_res_model = RF.fit(X_train, y_train)
    RF_res_model.predict(X_val)
    print(f"Train Score:{RF_res_model.score(X_train, y_train)}")
    print(f"Val Score:{RF_res_model.score(X_val, y_val)}")
    test_normal_folder = [f"test_{r}" for r in normal_folder]
    test_anomalous_folders = [f"test_{r}" for r in anomalous_folders]
    show_permutation_imp(RF_res_model, X_train, y_train)
    show_permutation_imp(RF_res_model, X_val, y_val)
    get_test_scores(
        RF_res_model, test_normal_folder, test_anomalous_folders, useful_cols
    )


if __name__ == "__main__":
    main()

