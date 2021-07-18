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
import math


CARLA_VERSION = "0.9.11"
try:
    # sys.path.append("./libs/carla-0.9.9-py3.7-linux-x86_64.egg")
    if CARLA_VERSION == "0.9.9":
        sys.path.append("./libs/carla-0.9.9-py3.7-linux-x86_64.egg")
    elif CARLA_VERSION == "0.9.11":
        sys.path.append("./libs/carla-0.9.11-py3.7-linux-x86_64.egg")
except IndexError:
    pass

import carla

import argparse
import random
import time
import logging
import click
import pathlib

import spawn

current_dir = pathlib.Path(__file__).parent.absolute()
SEED = 27
random.seed(SEED)


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


global_collision = False


def collision_detect_callback(event):
    actor_we_collide_against = event.other_actor
    impulse = event.normal_impulse
    intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
    if "vehicle." in actor_we_collide_against.type_id:
        global global_collision
        global_collision = True


def attach_collision_sensor(actor, world):
    blueprint_library = world.get_blueprint_library()

    collision_sensor = world.spawn_actor(
        blueprint_library.find("sensor.other.collision"),
        carla.Transform(),
        attach_to=actor,
    )

    collision_sensor.listen(lambda event: collision_detect_callback(event))

    return collision_sensor


def run(
    client,
    round_name,
    recording_dir,
    speed_violation_prob=60,
    tl_violation_prob=70,
    perc_speed_diff=-30,
    num_vehicles=25,
    SESSION_DURATION=60,
):
    safe = True  # avoid spawning vehicles whose geometry is not ideal for carla

    actor_list = []
    sensors = []

    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    try:
        FPS = 5
        DELTA_T = 1 / FPS

        world = client.get_world()
        blueprints = world.get_blueprint_library().filter("vehicle.*")
        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        if CARLA_VERSION == "0.9.11":
            print("Using deterministic Traffic Manager")
            traffic_manager.set_random_device_seed(SEED)
        settings = client.get_world().get_settings()
        if not settings.synchronous_mode:
            traffic_manager.set_synchronous_mode(True)
            synchronous_master = True
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = DELTA_T
            client.get_world().apply_settings(settings)
        else:
            synchronous_master = False

        recording_dir_path = pathlib.Path(recording_dir)
        recording_dir_path.mkdir(exist_ok=True)
        session_recording = str(recording_dir_path / f"{round_name}.csv")
        carla_session_recording = str(
            recording_dir_path.absolute() / f"{round_name}_carla_recording"
        )
        print("Recording on file: %s" % client.start_recorder(carla_session_recording))
        vehicles_list, walkers_list, all_actors = spawn.spawn(
            client, world, num_vehicles, 0, safe
        )
        world.tick()
        print("spawned %d vehicles, press Ctrl+C to exit." % len(actor_list))
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
        global global_collision
        global_collision = False
        actors = world.get_actors()
        for actor in actors:
            if "vehicle." in actor.type_id:
                sensors.append(attach_collision_sensor(actor, world))
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
        max_frames = FPS * SESSION_DURATION
        collision_detected_once = False
        while frame_id < max_frames:
            if global_collision and not collision_detected_once:
                # Todo, if detected, start a countdown of N frames and break only after N iterations
                print("detected collision, exiting!")
                collision_detected_once = True
                max_frames = frame_id + 5
                # continue

            actors = world.get_actors()
            for actor in actors:
                if "vehicle." in actor.type_id:
                    # print(actor.type_id)
                    tm_port = traffic_manager.get_port()
                    actor.set_autopilot(True, tm_port)
                    traffic_manager.ignore_lights_percentage(actor, tl_violation_prob)
                    traffic_manager.distance_to_leading_vehicle(actor, 3)
                    if random.random() * 100 < speed_violation_prob:
                        traffic_manager.vehicle_percentage_speed_difference(
                            actor, perc_speed_diff
                        )

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
        print(f"Saving CSV({len(df.frame_id.unique())} frames)")
        # df.to_parquet(f"session_data.parquet")
        df.to_csv(session_recording, index=False)
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
        print("\ndestroying %d actors" % (len(sensors) + len(vehicles_list)))
        # all_agents = sensors + vehicles_list
        for s in sensors:
            s.destroy()
        client.apply_batch_sync([carla.command.DestroyActor(x) for x in vehicles_list])

        print("Stop recording")
        client.stop_recorder()


@click.command()
@click.option(
    "-s",
    "--scenario_type",
    type=click.Choice(["tl_sl", "nominal"], case_sensitive=False),
    required=True,
)
@click.option("-n", "--num_rounds", default=100)
@click.option("--test", is_flag=True)
def main(scenario_type, num_rounds, test):
    # print(scenario_type, test, num_rounds)
    if test:
        random.seed(72)

    if scenario_type.lower() == "tl_sl":
        SPEED_VIOLATION_PROB = 60
        TL_VIOLATION_PROB = 70
        PERC_SPEED_DIFF = -30
        SCENARIO_NAME = "tl_sl"
        # NUM_ROUNDS = 100
    elif scenario_type.lower() == "nominal":
        SPEED_VIOLATION_PROB = 0
        TL_VIOLATION_PROB = 0
        PERC_SPEED_DIFF = 0
        SCENARIO_NAME = "nominal"
        # NUM_ROUNDS = 200
    NUM_ROUNDS = num_rounds
    print(f"Recording {SCENARIO_NAME} data")
    try:
        host = "127.0.0.1"  # IP of the host server (default: 127.0.0.1)
        port = 2000  # TCP port to listen to (default: 2000)",
        client = carla.Client(host, port)
        if test:
            scenario_dir = f"test_{SCENARIO_NAME}_recordings"
        else:
            scenario_dir = f"{SCENARIO_NAME}_recordings"

        round_names = []
        for i in range(NUM_ROUNDS):
            run(
                client,
                f"{scenario_type}_round_{i}",
                scenario_dir,
                SPEED_VIOLATION_PROB,
                TL_VIOLATION_PROB,
                PERC_SPEED_DIFF,
            )
            round_names.append(f"{scenario_type}_round_{i}")
            # client.reload_world()
    except KeyboardInterrupt:
        pass
    finally:
        print("\ndone.")


if __name__ == "__main__":
    main()
