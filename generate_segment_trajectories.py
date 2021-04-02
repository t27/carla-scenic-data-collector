import glob
import pandas as pd
import cv2
import numpy as np
import os
from dtw import dtw

# Here, we use L2 norm as the element comparison distance
l2_norm = lambda x, y: np.linalg.norm((x - y))


def get_agent_ids(folder, round_name):
    files = glob.glob(f"{folder}/{round_name}*")

    def parse_and_get_vehicle_id(st):
        # This function assumes a specific convention on the agent map filenames, that another module will generate

        _, filename = os.path.split(st)
        # breakpoint()
        # tl_sl2_round_0_vehicle_388_frame_184.csv.gz
        return int(
            filename[filename.rfind("vehicle_") + 8 : filename.rfind("frame_") - 1]
        )

    agent_ids = set([parse_and_get_vehicle_id(s) for s in files])
    return list(agent_ids)


def get_max_frame_value(folder, round_name):
    files = glob.glob(f"{folder}/{round_name}*")

    def parse_and_get_frame_id(st):
        _, filename = os.path.split(st)
        # print(filename)
        return int(filename[filename.rfind("frame_") + 6 : filename.rfind("csv") - 1])
        # return int(filename.split("_")[5].split(".")[0])

    frame_ids = set([parse_and_get_frame_id(s) for s in files])
    return max(frame_ids)


def get_basic_dataframe(
    subfolders=["round_0"], agent_map_folder="../agent_maps", max_agents=100,
):
    FRAME_LENGTH = 25  # how long should one unit of the scenario be/ what duration is enough to identify an anomaly
    RADIUS = 15  # how far should we look
    result = []
    for folder in subfolders:
        round_name_file = f"{agent_map_folder}/{folder}/round_names.txt"
        round_agent_map_folder = f"{agent_map_folder}/{folder}"
        with open(round_name_file) as f:
            round_names = f.read().split("\n")
        for round_id in round_names:
            agent_ids = get_agent_ids(round_agent_map_folder, round_name=round_id)
            max_frame = get_max_frame_value(round_agent_map_folder, round_name=round_id)
            for agent_id in agent_ids:
                # the below range function samples all non overlapping segments in the recording, can have some smart sampling here
                for start_index in range(0, max_frame, FRAME_LENGTH):
                    dtw_maps = []
                    if start_index + FRAME_LENGTH > max_frame:
                        continue
                    gen = GenerateSegmentTrajectories(
                        round_name=round_id,
                        agent_id=agent_id,
                        radius=RADIUS,
                        start_index=start_index,
                        frame_length=FRAME_LENGTH,
                        agent_map_folder=round_agent_map_folder,
                    )
                    gen.generate_basic_frame_level_stats()
                    result.append(gen.basic_frame_df)
                    if len(result) == max_agents:
                        return result

    return result


def get_dtw_maps(
    subfolders=["round_0"],
    agent_map_folder="../agent_maps",
    max_channels=10,
    max_dtw_maps=100,
):
    FRAME_LENGTH = 25  # how long should one unit of the scenario be/ what duration is enough to identify an anomaly
    RADIUS = 15  # how far should we look
    dtw_result = []
    for folder in subfolders:
        print(agent_map_folder, folder)
        round_name_file = f"{agent_map_folder}/{folder}/round_names.txt"
        round_agent_map_folder = f"{agent_map_folder}/{folder}"
        with open(round_name_file) as f:
            round_names = f.read().split("\n")
        # breakpoint()
        for round_id in round_names:
            agent_ids = get_agent_ids(round_agent_map_folder, round_name=round_id)
            max_frame = get_max_frame_value(round_agent_map_folder, round_name=round_id)
            print("Getting Data for - ", round_id, agent_ids, max_frame)
            for agent_id in agent_ids:
                # the below range function samples all non overlapping segments in the recording, can have some smart sampling here
                for start_index in range(0, max_frame, FRAME_LENGTH):
                    dtw_maps = []
                    if start_index + FRAME_LENGTH > max_frame:
                        continue
                    # print(start_index, max_frame)
                    # breakpoint()
                    gen = GenerateSegmentTrajectories(
                        round_name=round_id,
                        agent_id=agent_id,
                        radius=RADIUS,
                        start_index=start_index,
                        frame_length=FRAME_LENGTH,
                        agent_map_folder=round_agent_map_folder,
                    )
                    # print(start_index)
                    try:
                        gen.generate()
                    except Exception as exp:
                        print("error in generate", exp)
                        breakpoint()
                    data = gen.get_trajectory_data()
                    for other_agent in data["sorted_agent_ids"]:
                        if other_agent == agent_id:
                            continue
                        if len(dtw_maps) == max_channels:
                            break  # since sorted_agent_ids is sorted by agent's length, we will get the N longest dtw maps
                        dist, cost_matrix, acc_cost_matrix, path = dtw(
                            data["agent_tracks"][agent_id],
                            data["agent_tracks"][other_agent],
                            dist=l2_norm,
                        )
                        dtw_maps.append(np.expand_dims(acc_cost_matrix.T, axis=0))
                    # merge 10 dtw maps and make a 10,w,h tensor here and append to parent data list
                    if len(dtw_maps) > 0:
                        dtw_tensor = np.vstack(dtw_maps)
                        if dtw_tensor.shape[0] < max_channels:
                            # padding missing channels(each channel is one agent-ego pair) with zeros here
                            pad_size = max_channels - dtw_tensor.shape[0]
                            pad_obj = np.zeros(
                                (pad_size, dtw_tensor.shape[1], dtw_tensor.shape[2])
                            )
                            dtw_tensor = np.vstack([dtw_tensor, pad_obj])
                        dtw_result.append(dtw_tensor)
                        if len(dtw_result) > max_dtw_maps:
                            return dtw_result
    return dtw_result


class GenerateSegmentTrajectories:
    def __init__(
        self,
        round_name: str,
        agent_id: int,
        radius: float,
        start_index: int,
        frame_length: int,
        agent_map_folder: str = "agent_maps",
        save_video_file: str = None,
        show_window=True,
    ) -> None:
        self.round = round_name
        self.agent_id = agent_id
        self.agent_str = f"vehicle_{agent_id}"
        self.agent_maps = glob.glob(
            f"{agent_map_folder}/{self.round}_{self.agent_str}*.csv.gz"
        )
        # `st[st.rfind('frame_')+6:st.rfind('csv')-1]`` gets the frame_id from a
        # string that looks like 'agent_maps/round_0_vehicle_260_frame_0123.csv.gz'
        self.agent_maps.sort(
            key=lambda st: int(st[st.rfind("frame_") + 6 : st.rfind("csv") - 1])
        )
        self.start_index = start_index
        self.frame_length = frame_length
        self.radius = radius
        # for scaling the image (use if the image vis is too small)
        self.scale_factor = 10
        self.image_shape = (
            self.radius * 2 * self.scale_factor,
            self.radius * 2 * self.scale_factor,
        )
        self.car_color = (255, 0, 0)  # blue
        self.tl_color_map = {
            "Red": (0, 0, 255),
            "Green": (0, 255, 0),
            "Yellow": (0, 180, 180),
        }
        if save_video_file is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.videowriter = cv2.VideoWriter(
                save_video_file, fourcc, 20.0, self.image_shape
            )
        else:
            self.videowriter = None
        self.show_window = show_window
        self.reset_base_map()

    def reset_base_map(self):
        # create empty opencv image of radius*1.2 size ->self.base_img
        self.base_map = (
            np.ones((self.image_shape[0], self.image_shape[1], 3), dtype=np.uint8) * 255
        )

    def world2d_to_img(self, xy_arr):
        xy_arr[:, 0] += self.radius  # X coord
        xy_arr[:, 1] = (xy_arr[:, 1] * -1) + self.radius  # Y coord
        xy_arr *= self.scale_factor
        return xy_arr

    def _draw_traffic_light(self, df_row: pd.Series, x: int, y: int, image: np.ndarray):
        tl_color = df_row.traffic_light_color
        color = self.tl_color_map[tl_color]
        tlwidth = 2  # size of square of TL size
        return cv2.drawMarker(
            image,
            (x, y),
            color,
            markerType=cv2.MARKER_SQUARE,
            markerSize=self.scale_factor * tlwidth,
        )

    def _draw_vehicle(self, df_row: pd.Series, x: int, y: int, image: np.ndarray):

        if df_row.pos_x == 0 and df_row.pos_y == 0:
            # this is the center vehicle
            return cv2.drawMarker(
                image,
                (x, y),
                (0, 0, 0),
                markerType=cv2.MARKER_DIAMOND,
                markerSize=self.scale_factor * 2,
            )
        else:
            return cv2.circle(
                image,
                (x, y),
                1 * self.scale_factor,
                self.car_color,
                thickness=cv2.FILLED,
            )

    def __is_traffic_light(self, type_id):
        return "traffic_light" in type_id

    def __is_vehicle(self, type_id):
        return "vehicle" in type_id

    def draw_agent(
        self, type_id: str, df_row: pd.Series, x: int, y: int, image: np.ndarray
    ):
        if self.__is_traffic_light(type_id):
            return self._draw_traffic_light(df_row, x, y, image)
        elif self.__is_vehicle(type_id):
            return self._draw_vehicle(df_row, x, y, image)
        else:
            ## draw road signs here
            return image

    def visualize_frames(self):
        # for each frame, visualize and show frames
        for frame_file in self.agent_maps:
            df = pd.read_csv(frame_file, compression="gzip")
            df = df.query(f"abs(pos_x)<{self.radius} & abs(pos_y)<{self.radius}")
            xy_arr = df[["pos_x", "pos_y"]].values
            points = self.world2d_to_img(xy_arr)
            for i in range(points.shape[0]):
                agent_type = df.iloc[i].type_id
                x = int(points[i][0])
                y = int(points[i][1])
                self.base_map = self.draw_agent(
                    agent_type, df.iloc[i], x, y, self.base_map
                )
            if self.videowriter is not None:
                print(self.base_map.shape)
                self.videowriter.write(self.base_map)
            if self.show_window:
                cv2.imshow("Visualization", self.base_map)
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break
            self.reset_base_map()

        self.videowriter.release()
        ## Reset base image

    def get_trajectory_data(self):
        return {
            "merged_df": self.df_merged,
            "agent_tracks": self.agent_tracks,
            "agent_track_lens": self.agent_track_len,
            "sorted_agent_ids": self.sorted_agent_ids,
            "agent_metadata_dfs": self.agent_metadata_dfs,
        }

    def generate_basic_frame_level_stats(self):
        ### Each row of the basic frame level stats includes information about one frame wrt other agents
        i = self.start_index
        df_array = []
        # load the dataframes from the multiple files(one file per frame_id, agent_id pair)
        for k in range(self.frame_length):  # segment length
            frame_id = i + k
            # print(frame_id)
            df = pd.read_csv(self.agent_maps[frame_id], compression="gzip")
            df = df.query(f"abs(pos_x)<{self.radius} & abs(pos_y)<{self.radius}")
            df_array.append(df)
        # merge all dataframes together
        self.df_merged = pd.concat(df_array).reset_index()
        frame_grp = self.df_merged.groupby("frame_id")

        frame_id_wise_df = {}
        for frame_id in frame_grp.groups:
            sub_df = self.df_merged.iloc[frame_grp.groups[frame_id]]
            frame_id_wise_df[frame_id] = sub_df

        self.frame_id_wise_df = frame_id_wise_df
        # fmt:off
        column_names = [ "frame_id","round_id","ego_agent_id","num_vehicles", "max_velocity_x", "max_velocity_y", 
        "max_velocity_z", "max_ang_velocity_x", "max_ang_velocity_y", "max_ang_velocity_z", "min_velocity_x", 
        "min_velocity_y", "min_velocity_z", "min_ang_velocity_x", "min_ang_velocity_y", "min_ang_velocity_z", 
        "max_acc_x", "max_acc_y", "max_acc_z", "min_acc_x","min_acc_y","min_acc_z"]
        # fmt: on
        result = []
        for frame_id in frame_id_wise_df:
            stats = self._calculate_frame_stats(frame_id_wise_df[frame_id], frame_id)
            result.append(stats)
        self.basic_frame_df = pd.DataFrame(result, columns=column_names)

    def _calculate_frame_stats(self, frame_df, frame_id):
        frame_id = frame_id
        round_id = self.round
        num_vehicles = len(frame_df.id.unique())
        max_velocity_x = frame_df.velocity_x.max()
        max_velocity_y = frame_df.velocity_y.max()
        max_velocity_z = frame_df.velocity_z.max()
        max_ang_velocity_x = frame_df.angular_vel_x.max()
        max_ang_velocity_y = frame_df.angular_vel_y.max()
        max_ang_velocity_z = frame_df.angular_vel_z.max()
        min_velocity_x = frame_df.velocity_x.min()
        min_velocity_y = frame_df.velocity_y.min()
        min_velocity_z = frame_df.velocity_z.min()
        min_ang_velocity_x = frame_df.angular_vel_x.min()
        min_ang_velocity_y = frame_df.angular_vel_y.min()
        min_ang_velocity_z = frame_df.angular_vel_z.min()
        max_acc_x = frame_df.acc_x.max()
        max_acc_y = frame_df.acc_y.max()
        max_acc_z = frame_df.acc_z.max()
        min_acc_x = frame_df.acc_x.min()
        min_acc_y = frame_df.acc_y.min()
        min_acc_z = frame_df.acc_z.min()
        # potentially add ego stats as separate columns?
        # fmt:off
        res = [
            frame_id, round_id, num_vehicles, self.agent_id, max_velocity_x, max_velocity_y, max_velocity_z, 
            max_ang_velocity_x, max_ang_velocity_y, max_ang_velocity_z, min_velocity_x, min_velocity_y, 
            min_velocity_z, min_ang_velocity_x, min_ang_velocity_y, min_ang_velocity_z, max_acc_x, max_acc_y, 
            max_acc_z, min_acc_x, min_acc_y, min_acc_z,
        ]
        # fmt:on
        return res

    def generate(self):
        """
        Uses the agent id, the start frame ids and segment length generates agent level trajectories of a fixed size segment (frame_length)
        Generates the following objects:
            self.df_merged = the merged dataframe for all frames
            self.agent_tracks = The [Nx2] sized trajectories for all agents in the scenario, 
                                if the agent wasn't there in the frame at a given time, 
                                the coordinate value is [self.radius + 1, self.radius + 1]
            self.agent_track_len = the effective length of the trajectories(only valid values)
            self.sorted_agent_ids = agent ids sorted by their trajectory length
                                    (agents with the most duration are higher in the list)
        """
        i = self.start_index
        df_array = []
        # load the dataframes from the multiple files(one file per frame_id, agent_id pair)
        for k in range(self.frame_length):  # segment length
            frame_id = i + k
            # print(frame_id)
            # breakpoint()
            if frame_id >= len(self.agent_maps):
                print(
                    f"Trying to access frame {frame_id} but only have {len(self.agent_maps)}"
                )
                break
            df = pd.read_csv(self.agent_maps[frame_id], compression="gzip")
            df = df.query(f"abs(pos_x)<{self.radius} & abs(pos_y)<{self.radius}")
            df_array.append(df)
        # merge all dataframes together
        if len(df_array) == 0:
            self.df_merged = pd.DataFrame()
            self.agent_tracks = {}
            self.agent_metadata_dfs = {}
            self.agent_track_len = {}
            self.sorted_agent_ids = []
            return
        self.df_merged = pd.concat(df_array).reset_index()
        # group all dataframes by id, so that we can get agent level metrics across time
        agent_grp = self.df_merged.groupby("id")
        self.agent_tracks = {}
        self.agent_metadata_dfs = {}
        self.agent_track_len = {}
        for agent_id in agent_grp.groups:
            sub_df = self.df_merged.iloc[agent_grp.groups[agent_id]]
            # empty template for the trajectories
            tracks = [
                [self.radius + 1, self.radius + 1] for _ in range(self.frame_length)
            ]
            track_len = 0
            # populate the empty template
            for idx, row in sub_df.iterrows():
                frame_idx = row.frame_id - self.start_index
                # if row.frame_id == 0:
                #     # check if the object is within the car position add to the current frame
                #     print("0 frame id")
                try:
                    tracks[frame_idx] = [row.pos_x, row.pos_y]
                except:
                    breakpoint()
                track_len += 1
            self.agent_tracks[agent_id] = np.array(tracks)
            self.agent_metadata_dfs[agent_id] = sub_df
            self.agent_track_len[agent_id] = track_len
        self.sorted_agent_ids = list(self.agent_track_len.keys())
        self.sorted_agent_ids.sort(key=lambda x: self.agent_track_len[x], reverse=True)


def test_visualize():
    round_name = "scenario1.log"
    folder = "agent_maps/debris_avoidance_recordings"
    agent_ids = get_agent_ids(folder, round_name)
    obj = GenerateSegmentTrajectories(
        round_name=round_name,
        agent_id=agent_ids[1],
        radius=20,
        start_index=0,
        frame_length=50,
        agent_map_folder=folder,
        save_video_file="testvideo.mp4",
    )
    obj.visualize_frames()


if __name__ == "__main__":
    test_visualize()

