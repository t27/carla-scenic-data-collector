#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from collections import defaultdict
from math import cos, sin, pi
import os
from tqdm import tqdm
from pathlib import Path
import glob


class CarlaCsvParser:
    def __init__(self, recording_folder, round_name, agent_map_folder) -> None:
        self.round_name = round_name
        self.recording_folder = recording_folder
        print("Loading Dataframe...,", round_name)
        self.df = pd.read_csv(os.path.join(recording_folder, round_name) + ".csv")
        self.df.index.name = "index"
        self.agent_maps_dir = agent_map_folder
        # breakpoint()
        vehicles = self.df.query('type_id.str.contains("vehicle")', engine="python")
        self.agent_wise_idxs = vehicles.groupby("id").groups

    # For each frame, for each vehicle, get the transform and generate the required feature map

    # 2D R,T matrix
    def _get_tf_mat(self, x, y, t):
        t = t * pi / 180
        rot = np.array([[cos(-t), -sin(-t), 0], [sin(-t), cos(-t), 0], [0, 0, 1]])
        trans = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]])
        return rot @ trans  # first translate then rotate

    def _generate_tf_matrices(self):
        """ For each vehicle, generate an R and T and a TF matrix per frame"""
        # frame_wise_vehicle_tfs = {} => {frame_id:{veh_id:[tf_mat]}}
        self.frame_wise_vehicle_tf = defaultdict(defaultdict)

        for agent_id in self.agent_wise_idxs:
            agent_df = self.df.iloc[self.agent_wise_idxs[agent_id]]
            for idx, row in agent_df.iterrows():
                frame_id = row.frame_id
                self.frame_wise_vehicle_tf[frame_id][agent_id] = {
                    "mat": self._get_tf_mat(row.pos_x, row.pos_y, row.yaw),
                    "x": row.pos_x,
                    "y": row.pos_y,
                    "z": row.pos_z,
                    "roll": row.roll,
                    "pitch": row.pitch,
                    "yaw": row.yaw,
                }

    def _transform_frame_df(self, frame_df, frame_id, center_agent_id, save_file=True):
        """ For each vehicle, store a transformed frame_wise map of all other vehicles.
            Saves files as round_0_vehicle_0_frame_0.csv.gz
        """
        mat = self.frame_wise_vehicle_tf[frame_id][center_agent_id]["mat"]
        frame_df_local = frame_df.copy()
        xy_hmg = frame_df_local[["pos_x", "pos_y"]].copy()
        xy_hmg["z"] = 1  # creating homogenous coordinate
        frame_df_local[["pos_x", "pos_y"]] = (mat @ xy_hmg.T).T.values[:, :2]
        if save_file:
            # frame_df_local.to_parquet(
            #     f"{self.agent_maps_dir}/{self.round_name}_vehicle_{center_agent_id}_frame_{frame_id}.parquet"
            # )
            Path(f"{self.agent_maps_dir}/{self.recording_folder}").mkdir(exist_ok=True)
            frame_df_local.to_csv(
                f"{self.agent_maps_dir}/{self.recording_folder}/{self.round_name}_vehicle_{center_agent_id}_frame_{frame_id}.csv.gz",
                compression="gzip",
            )

    def run(self):
        # print("Generating Transforms...")
        self._generate_tf_matrices()
        n_frames = len(self.df.frame_id.unique())
        n_agents = len(self.agent_wise_idxs.keys())
        # print(f"Total unique frames = {n_frames}. Total Agents = {n_agents}")
        # print(f"Projecting a max of {n_frames*n_agents} agent maps")
        for agent_id in self.agent_wise_idxs:
            # get all rows containing the agent
            agent_df = self.df.iloc[self.agent_wise_idxs[agent_id]]
            # get frame ids where agent was present
            frame_ids = list(agent_df.frame_id)
            for frame_id in frame_ids:
                # get full df containing all agents for the given frame
                frame_df = self.df[self.df.frame_id == frame_id]
                # TODO: append static objects(from frame 0 here)
                static_objects = self.df.query(
                    'frame_id==0 and not (type_id.str.contains("vehicle") or type_id.str.contains("traffic_light"))',
                    engine="python",
                ).copy()
                # update the frame id here for static objects
                static_objects.frame_id = frame_id
                frame_df = pd.concat([frame_df, static_objects])

                # transform the current frame's agents relative to current agent
                self._transform_frame_df(frame_df, frame_id, agent_id)


from concurrent.futures import ProcessPoolExecutor, as_completed

if __name__ == "__main__":
    # This step is pretty long, takes a few hours per round as we pre calculate all the
    # agent transforms and transform the other nearby agents for each agent in each frame,
    # more the agents more the time this step takes
    # rounds = ["round_1","round_2","round_3","round_4"]
    # folder_rounds = [
    #     ("debris_avoidance_recordings", "scenario1"),
    #     ("oncoming_car_recordings", "scenario1"),
    # ]
    print("Generating Agent Maps...")
    agent_map_folder = "./agent_maps"  # output
    Path(agent_map_folder).mkdir(exist_ok=True)

    folders = [
        "debris_avoidance_recordings",
        "oncoming_car_recordings",
        "normal_recordings",
        "tl_sl_recordings",
    ]
    pool = ProcessPoolExecutor(6)

    def job(folder, roundname, agent_map_folder):
        converter = CarlaCsvParser(folder, roundname, agent_map_folder)
        converter.run()

    futures = []
    for folder in folders:
        rounds = glob.glob(f"{folder}/*.csv")
        round_names = []
        Path(f"{agent_map_folder}/{folder}").mkdir(exist_ok=True)
        for round in rounds:
            round_name = os.path.split(round)[-1].replace(".csv", "")
            futures.append(pool.submit(job, folder, round_name, agent_map_folder))
            round_names.append(round_name)
        with open(f"{agent_map_folder}/{folder}/round_names.txt", "w") as f:
            f.write("\n".join(round_names))
    total = len(futures)
    for x in tqdm(as_completed(futures), total=total):
        x.result()
        # print("Done")
    # for round in rounds:
    #     converter = CarlaCsvParser("./recordings", round)
    #     converter.run()

