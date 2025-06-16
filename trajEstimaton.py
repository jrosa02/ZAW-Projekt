import cv2
from odometry import PinholeCamera, VisualOdometry
from dataVisualization import LanderData
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import os


def resample_2d_array(a, m):
    """
    Resamples a 2D array of shape (n, d) to shape (m, d),
    where resampling is done along axis 0 (rows).
    
    Parameters:
        a: np.ndarray of shape (n, d)
        m: int, target number of rows
        
    Returns:
        np.ndarray of shape (m, d)
    """
    if a.ndim != 2:
        raise ValueError("Input array must be 2D (n, d)")
    
    n, d = a.shape
    original_positions = np.linspace(0, 1, n)
    target_positions = np.linspace(0, 1, m)

    # Interpolujemy każdą kolumnę osobno
    resampled = np.empty((m, d))
    for i in range(d):
        resampled[:, i] = np.interp(target_positions, original_positions, a[:, i])
    
    return resampled


class TrajEstimator(LanderData):
    '''
    Read and proccess input events from given .npz file
    Parameters:
    - `npz_path` : str - path to input .npz file
    '''
    def __init__(self, npz_path : str, dvs_resolution : tuple = (200, 200),
                 fx=217.2, fy=277.0, cx=None, cy=None, k1=0, k2=0, p1=0, p2=0, k3=0,  # Camera intrisic parameters
                 filter_data : bool = False, filter_t : float = 1 / 24, filter_k : int = 1, filter_size : int = 5,  # Filter params
                 filter_output_pose : bool = False, output_filter_cutoff : float = None, # Lowpass butterworth filter params
                 vertical_scaling_factor = 1.0
                 ):  
        
        super().__init__(npz_path, dvs_resolution, filter_data, filter_t, filter_k, filter_size)
        
        cx = self.img_shape[1] // 2 if cx is None else cx
        cy = self.img_shape[0] // 2 if cy is None else cy
        
        self.__cam = PinholeCamera(self.img_shape[1], self.img_shape[0], fx, fy, cx, cy, k1, k2, p1, p2, k3)
        
        self.estimated_trajectory = {
            "position": np.array([]),
            "velocity": np.array([])
        }
        
        self.pos_cost = None
        self.vel_cost = None
        
        self.__filter_output_pose = filter_output_pose
        self.__output_filter_cutoff = output_filter_cutoff

        self.vertical_scaling_factor = vertical_scaling_factor

    def process_event_frames(self, t_start=0, t_end=np.inf, tau=0.1, wait=100, frame_type="standard", display=False):
        """
        Displays event frames between t_start and t_end with temporal resolution `tau`.
        Parameters:
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

        # Create VisualOdometry object for pose estimation
        n_frames = int((ts[-1] - ts[0]) / tau)
        vo = VisualOdometry(self.__cam, trajectory=self.trajectory, rangemeter=self.rangemeter, n_frames=n_frames, frame_rate=n_frames/self.timestamps[-1], vertical_scaling_factor = self.vertical_scaling_factor)

        # Process frames in time slices of `tau`
        temp_x, temp_y, temp_p, temp_ts = [], [], [], []
        start_time = ts[0]
        frame_count = 0
        for i in range(len(ts)):
            t = ts[i]
            temp_ts.append(ts[i])
            temp_x.append(xs[i])
            temp_y.append(ys[i])
            temp_p.append(ps[i])

            if t - start_time >= tau:
                # Generate frame
                if frame_type == "exponential":
                    frame = self._exponantial_decay_aggregation((temp_x, temp_y, temp_p, temp_ts))
                else:  # Standard event frame
                    frame = self._event_frame_agregation((temp_x, temp_y, temp_p, temp_ts))

                # Process frame
                vo.update(frame, frame_id=frame_count)
                
                # Show frame
                if display:
                    frame = cv2.resize(frame, np.multiply(4, frame.shape), interpolation = cv2.INTER_LINEAR)
                    cv2.imshow("Event Frame", frame)
                    key = cv2.waitKey(wait)
                    if key == ord('q'):
                        print("User exited with 'q'.")
                        break

                # Reset for next frame
                frame_count += 1
                start_time = t
                temp_x, temp_y, temp_p = [], [], []
        
        # Add start values if it is train data else keep it at 0
        estimated_pos = np.array(vo.position_trajectory).squeeze()
        if not np.isnan(self.trajectory["position"][0]).any():
            estimated_pos += self.trajectory["position"][0]
        
        if self.__filter_output_pose:
            fs = 1 / (self.timestamps[-1] / n_frames)  # nyquist freq
            cutoff = self.__output_filter_cutoff if self.__output_filter_cutoff is not None else ((1 / self.timestamps[-1]) * 6) 
            
            b, a = butter(N=4, Wn=cutoff, btype='low', fs=fs)
            estimated_pos = filtfilt(b, a, estimated_pos, axis=0)
        
        self.estimated_trajectory["position"] = resample_2d_array(estimated_pos, self.trajectory["position"].shape[0])
        
        # Velocity estimation - centered
        # TODO: Chujnia straszna wychodzi - to nie może tak być - dlatego dodałem filtrację dolnoprzepustową- ciągle jest do dupy, trzeba to inaczej liczyć
        estimated_vel = (np.roll(estimated_pos, -1, axis=0) - np.roll(estimated_pos, 1, axis=0)) / (2 * tau)
        estimated_vel[0] = (estimated_pos[1] - estimated_pos[0]) / tau         # forward diff
        estimated_vel[-1] = (estimated_pos[-1] - estimated_pos[-2]) / tau      # backward diff
        
        # estimated_vel = np.array(vo.velocity_trajectory).squeeze()  # jeśli się doda liczenie prędkości w vo
        
        self.estimated_trajectory["velocity"] = resample_2d_array(estimated_vel, self.trajectory["velocity"].shape[0])
        
        if display:
            print(f"\nDisplayed {frame_count} event frames.")
            cv2.destroyAllWindows()
            
        self.pos_cost = self.pos_cost_fun()
        self.vel_cost = self.vel_cost_fun()
            
        return self.vel_cost  # Docelowo niech zwraca vel_cost, bo to jest metryka, według której oceniają
            
    def pos_cost_fun(self):
        if self.estimated_trajectory["position"].size == 0:
            print("Estimated position not calculated!")
            return None
        
        error = np.mean((self.estimated_trajectory["position"] - self.trajectory["position"])**2) / self.trajectory["position"].shape[0]
        return error
    
    def vel_cost_fun(self):
        '''
        Scoring metric
        '''
        if self.estimated_trajectory["velocity"].size == 0:
            print("Estimated velocity not calculated!")
            return None
        
        error = np.sqrt(np.sum((self.estimated_trajectory["velocity"] - self.trajectory["velocity"]) ** 2, axis=1)) / (self.trajectory["velocity"].shape[0] * self.trajectory["position"][:,2])
        return error
                    
    def plot_estimated_trajectory(self):
        if self.estimated_trajectory["position"].size != 0:
            # Position
            plt.plot(self.timestamps, self.estimated_trajectory["position"])
            plt.title("Estimated Position (x, y, z)")
            plt.xlabel("Time [s]")
            plt.ylabel("position [m]")
            plt.legend(["x", "y", "z"])
            plt.grid()
            plt.show()
        else:
            print("Estimated position not calculacted!")

        if self.estimated_trajectory["velocity"].size != 0:
            # Velocity
            plt.plot(self.timestamps, self.estimated_trajectory["velocity"])
            plt.title("Estimated Velocity (vx, vy, vz)")
            plt.xlabel("Time [s]")
            plt.ylabel("Velocity [m/s]")
            plt.legend(["vx", "vy", "vz"])
            plt.grid()
        else:
            print("Estimated velocity not calculacted!")
            
    def save_result(self, out_path="out"):
        self.data["traj"][:, 0:3] = self.estimated_trajectory["position"]
        self.data["traj"][:, 3:6] = self.estimated_trajectory["velocity"]
        
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        
        np.savez_compressed(os.path.join(out_path, f"{self.npz_path.split('/')[-1].removeprefix('/')}"), self.data)