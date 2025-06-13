import cv2
from odometry import PinholeCamera, VisualOdometry
from dataVisualization import LanderData
import numpy as np


class TrajEstimator(LanderData):
    def __init__(self, npz_path : str, dvs_resolution : tuple = (200, 200),
                 fx=217.2, fy=277, cx=None, cy=None, k1=0, k2=0, p1=0, p2=0, k3=0,  # Camera intrisic parameters
                 filter_data : bool = False, filter_t : float = 1 / 24, filter_k : int = 1, filter_size : int = 5):  # Filter params
        
        super().__init__(npz_path, dvs_resolution, filter_data, filter_t, filter_k, filter_size)
        
        cx = dvs_resolution[1] // 2 if cx is None else cx
        cy = dvs_resolution[0] // 2 if cy is None else cy
        
        cam = PinholeCamera(fx, fy, cx, cy, k1, k2, p1, p2, k3)
        self.__vo = VisualOdometry(cam)
        
    def process_event_frames(self, t_start=0, t_end=np.inf, tau=0.1, wait=100, frame_type="standard", display=True):
        """
        Displays event frames between t_start and t_end with temporal resolution `tau`.
        - `frame_type`: str : "standard" or "exponential"
        """
        # Convert structured array to NumPy arrays
        ts = self.events["t"] * 1e-6  # Convert from µs to seconds
        xs = self.events["x"]
        ys = self.events["y"]
        ps = self.events["p"]
        ps = np.where(ps == 1, 1, -1)  # Convert to +1 / -1

        # Filter events in the time range
        mask = (ts >= t_start) & (ts <= t_end)
        ts, xs, ys, ps = ts[mask], xs[mask], ys[mask], ps[mask]

        # Display frames in time slices of `tau`
        temp_x, temp_y, temp_p, temp_ts = [], [], [], []
        start_time = ts[0]
        frame_count = 0
        estimated_traj = []

        for i in range(len(ts)):
            t = ts[i]
            temp_ts.append(ts[i])
            temp_x.append(xs[i])
            temp_y.append(ys[i])
            temp_p.append(ps[i])

            if t - start_time >= tau:
                # Generate frame
                if frame_type == "exponential":
                    frame = self.__exponantial_decay_aggregation((temp_x, temp_y, temp_p, temp_ts))
                else:  # Standard event frame
                    frame = self.__event_frame_agregation((temp_x, temp_y, temp_p, temp_ts))

                # Process frame
                self.__vo.update(frame, frame_id=frame_count)
                
                print(self.__vo.cur_t)  # Estimated postion
                
                estimated_traj.append(self.__vo.cur_t)  # TODO: reshape to lenght of trajectory and write to trajectory
                
                # Show frame
                if display:
                    frame = cv2.resize(frame, np.multiply(4, frame.shape), interpolation = cv2.INTER_LINEAR)  # Resize, żeby było większe okno?
                    cv2.imshow("Event Frame", frame)
                    key = cv2.waitKey(wait)
                    if key == ord('q'):
                        print("User exited with 'q'.")
                        break

                # Reset for next frame
                frame_count += 1
                start_time = t
                temp_x, temp_y, temp_p = [], [], []

        print(f"\nDisplayed {frame_count} event frames.")
        cv2.destroyAllWindows()