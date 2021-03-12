import glob
import pandas as pd
import cv2
import numpy as np


class VisualizeAgentMaps:
    def __init__(
        self,
        round_number: int,
        agent_id: int,
        radius: float,
        agent_map_folder: str = "agent_maps",
        save_video_file: str = None,
        show_window=True,
    ) -> None:
        self.round = f"round_{round_number}"
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
            fourcc = cv2.VideoWriter_fourcc(*"vp80")
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
                # print(self.base_map.shape)
                self.videowriter.write(self.base_map)
            if self.show_window:
                cv2.imshow("Visualization", self.base_map)
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break
            self.reset_base_map()

        self.videowriter.release()
        ## Reset base image


if __name__ == "__main__":
    viz = VisualizeAgentMaps(0, 324, 15, save_video_file="test.webm")
    viz.visualize_frames()

