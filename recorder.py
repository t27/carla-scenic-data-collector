# Recorder that records agent states as dataframes and also stores a carla recording, in synchronous mode


#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
import pandas as pd
from tqdm import tqdm
from pathlib import Path

try:
    sys.path.append("./libs/carla-0.9.9-py3.7-linux-x86_64.egg")
except IndexError:
    pass

import carla

import argparse
import random
import time
import logging

import pathlib

current_dir = pathlib.Path(__file__).parent.absolute()


def get_metadata(actor, frame_id):
    type_id = actor.type_id

    def splitCarlaVec(vect):
        return vect.x, vect.y, vect.z

    id = actor.id
    # clsname = ClientSideBoundingBoxes.get_class_name(actor)
    tf = actor.get_transform()
    roll, pitch, yaw = tf.rotation.roll, tf.rotation.pitch, tf.rotation.yaw
    loc = actor.get_location()
    pos_x, pos_y, pos_z = splitCarlaVec(loc)
    try:
        bbox3d = actor.bounding_box
        bbox3d_offset_x, bbox3d_offset_y, bbox3d_offset_z = splitCarlaVec(
            bbox3d.location
        )
        bbox3d_extent_x, bbox3d_extent_y, bbox3d_extent_z = splitCarlaVec(bbox3d.extent)
    except:
        bbox3d_offset_x, bbox3d_offset_y, bbox3d_offset_z = None, None, None
        bbox3d_extent_x, bbox3d_extent_y, bbox3d_extent_z = None, None, None

    velocity_x, velocity_y, velocity_z = splitCarlaVec(actor.get_velocity())
    acc_x, acc_y, acc_z = splitCarlaVec(actor.get_acceleration())
    angular_vel_x, angular_vel_y, angular_vel_z = splitCarlaVec(
        actor.get_angular_velocity()
    )

    try:
        # need to do this because Carla's Actor object doesnt support getattr
        traffic_light_state = actor.state.name
    except:
        traffic_light_state = None

    return (
        frame_id,
        id,
        type_id,
        pos_x,
        pos_y,
        pos_z,
        roll,
        pitch,
        yaw,
        velocity_x,
        velocity_y,
        velocity_z,
        acc_x,
        acc_y,
        acc_z,
        angular_vel_x,
        angular_vel_y,
        angular_vel_z,
        bbox3d_offset_x,
        bbox3d_offset_y,
        bbox3d_offset_z,
        bbox3d_extent_x,
        bbox3d_extent_y,
        bbox3d_extent_z,
        traffic_light_state,
    )


def run(client, round_name, dest_folder="", session_duration_sec=10):

    # num_vehicles = 70
    # safe = True  # avoid spawning vehicles prone to accidents"

    actor_list = []
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
    dest_folder = Path(dest_folder)
    try:
        SESSION_DURATION = session_duration_sec  # seconds # TODO is it possible to read this from the carla recording file? or an external config?
        FPS = 5
        DELTA_T = 1 / FPS
        # RECORDING_FILENAME = ""
        START_TIME = 0.0
        DURATION = 0.0

        # client.set_timeout(2.0)
        world = client.get_world()
        # blueprints = world.get_blueprint_library().filter("vehicle.*")
        # traffic_manager = client.get_trafficmanager()
        settings = client.get_world().get_settings()
        if not settings.synchronous_mode:
            # traffic_manager.set_synchronous_mode(True)
            synchronous_master = True
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = DELTA_T
            client.get_world().apply_settings(settings)
        else:
            synchronous_master = False
        session_recording = f"replay_{round_name}"
        destination_filename = dest_folder / Path(session_recording)

        world.tick()
        # fmt: off
        df_columns = [
            "frame_id", "id", "type_id", "pos_x", "pos_y", "pos_z", "roll", "pitch", "yaw", 
            "velocity_x", "velocity_y", "velocity_z", "acc_x", "acc_y", "acc_z", 
            "angular_vel_x", "angular_vel_y", "angular_vel_z", 
            "bbox3d_offset_x", "bbox3d_offset_y", "bbox3d_offset_z", 
            "bbox3d_extent_x", "bbox3d_extent_y", "bbox3d_extent_z", "traffic_light_color",
        ]
        # fmt: on
        # get all non vehicle agents
        actors = world.get_actors()
        non_vehicles = [
            x
            for x in actors
            if ("vehicle" not in x.type_id and "traffic_light" not in x.type_id)
        ]  # signs, traffic lights etc
        frame_id = 0
        df_arr = []
        non_vehicle_arr = [get_metadata(actor, frame_id) for actor in non_vehicles]
        df_arr += non_vehicle_arr
        pbar = tqdm(total=FPS * SESSION_DURATION)
        while frame_id < (FPS * SESSION_DURATION):
            actors = world.get_actors()
            vehicles_and_lights = [
                x
                for x in actors
                if "vehicle" in x.type_id or "traffic_light" in x.type_id
            ]
            metadata_arr = [
                get_metadata(actor, frame_id) for actor in vehicles_and_lights
            ]
            df_arr += metadata_arr
            frame_id += 1
            pbar.update(1)
            world.tick()
        df = pd.DataFrame(df_arr, columns=df_columns)
        pbar.close()
        print("Saving CSV")
        # df.to_parquet("session_data.parquet")
        df.to_csv(str(destination_filename) + ".csv", index=False)
        print("Saving Parquet")
        # df.to_parquet("session_data.parquet")
        df.to_csv(str(destination_filename) + ".parquet", index=False)
        world.tick()
        # if args.recorder_time > 0:
        #     time.sleep(args.recorder_time)
        # else:
        #     while True:
        #         world.wait_for_tick()
        #         # time.sleep(0.1)

    finally:
        if synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        print("\ndestroying %d actors" % len(actor_list))
        # client.apply_batch_sync([carla.command.DestroyActor(x) for x in vehicles_list])

        # print("Stop recording")
        # client.stop_recorder()


# iterate through all files and save recordings


def convert_recording(carla_recording, prefix="", dest_folder=""):
    # use a prefix to specifically tag a roundname
    if dest_folder:
        Path(dest_folder).mkdir(exist_ok=True)
    try:
        host = "127.0.0.1"  # IP of the host server (default: 127.0.0.1)
        port = 2000  # TCP port to listen to (default: 2000)",
        client = carla.Client(host, port)
        # extracts the file name = scenario1.log
        roundname = prefix + os.path.split(carla_recording)[-1]

        client2 = carla.Client(host, port)
        client2.set_timeout(60.0)

        client2.set_replayer_time_factor(1.0)
        fullpath = pathlib.Path(carla_recording).absolute()

        print(client2.replay_file(str(fullpath), 0.0, 0.0, 0,))

        run(client, roundname, dest_folder=dest_folder, session_duration_sec=10)
    except KeyboardInterrupt:
        pass
    finally:
        print("\ndone.")


if __name__ == "__main__":

    folders = ["oncoming_car_recordings", "debris_avoidance_recordings"]

    for folder in folders:
        files = glob.glob(f"./{folder}/*.log")
        for fil in files:
            # TODO: identify format of prefix for each diff anomaly
            convert_recording(fil, dest_folder=folder)

