import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import butter, filtfilt


event_dtype = np.dtype([
    ('x', np.uint16),
    ('y', np.uint16),
    ('p', np.uint8),
    ('t', np.uint64)
])


class LanderData:
    '''
    Read input events from given .npz file
    @param: npz_path - path to input .npz file
    '''
    def __init__(self, npz_path : str, dvs_resolution : tuple = (200, 200),  # w zasadzie to ten zbiór danych symuluje DAVIS240, więc rozmiar ramki to powinno być 240x180, ale faktycznie w danych jest 200x200
                 filter_data : bool = False, filter_t : float = 1 / 24, filter_k : int = 1, filter_size : int = 5,  # Filter params
                 filter_rangemeter : bool = True, rangemeter_filter_cutoff : float = None):  # Filtering rangemeter signal
        self.npz_path = npz_path
        self.img_shape = (dvs_resolution[1], dvs_resolution[0])  # Swapped
        
        # Filtering parameters
        self.__filter_data = filter_data
        self.filter_t = filter_t if filter_data else None
        self.filter_k = filter_k if filter_data else None
        self.filter_size = filter_size if filter_data else None
        
        self.__filter_rangemeter = filter_rangemeter
        self.__rangemeter_filter_cutoff = rangemeter_filter_cutoff
        
        # Loading data to self.events, self.trajectory, self.timestamps and self.rangemeter
        self._load_data()

    def _load_data(self):
        self.data = np.load(self.npz_path, allow_pickle=True)
        
        events_array = self.data["events"]
        events_np = np.array(events_array, dtype=event_dtype)
        if not self.__filter_data:
            self.events = events_np
        else:  # Filtering events - not taking into account polarity of each event
            self.__sae = np.zeros(self.img_shape, dtype=float)  # surface of active events - last timestamp for each event
            self.signal_events_n = 0
            filtered_events = []
            for event in events_np:
                xi, yi, ti = event['x'], event['y'], event['t']
                # Omitting too frequent events
                if ti - self.__sae[yi, xi] < 1e-5:
                    continue
                    
                self.__sae[yi, xi] = ti
                if not self.__is_ba(event):
                    self.signal_events_n += 1
                    filtered_events.append(event)
            
            self.noise_events_n = events_np.shape[0] - self.signal_events_n
                
            self.events = np.array(filtered_events, dtype=event_dtype)

        
        self.timestamps = self.data["timestamps"]
        self.trajectory = {
            "position": self.data["traj"][:, 0:3],
            "velocity": self.data["traj"][:, 3:6],
            "euler_angles": self.data["traj"][:, 6:9],
            "angular_velocity": self.data["traj"][:, 9:12],
        }
        self.rangemeter = {
            "time": self.data["range_meter"][:, 0],
            "distance": self.data["range_meter"][:, 1],
        }
        
        if self.__filter_rangemeter:
            fs = 1 / (self.rangemeter["time"][-1] / len(self.rangemeter["time"]))  # nyquist freq
            cutoff = self.__rangemeter_filter_cutoff if self.__rangemeter_filter_cutoff is not None else ((1 / self.rangemeter["time"][-1]) * 10) 
            
            b, a = butter(N=4, Wn=cutoff, btype='low', fs=fs)
            self.rangemeter["distance"] = filtfilt(b, a, self.rangemeter["distance"], axis=0)

    def summary(self):
        print(f"Loaded LanderData from: {self.npz_path}")
        if self.__filter_data and self.signal_events_n and self.noise_events_n:
            print(f"Signal events: {self.signal_events_n}, Noise events: {self.noise_events_n}")
        else:
            print(f"Events: {len(self.events)} entries")
        print(f"Timestamps: {len(self.timestamps)} entries")
        print(f"Trajectory: {len(self.trajectory['position'])} entries")
        print(f"IMU: {len(self.trajectory['euler_angles'])} entries")
        print(f"Rangemeter: {len(self.rangemeter['time'])} entries")
    
    def plot_rangemeter(self):
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.rangemeter["time"], self.rangemeter["distance"])
        ax.grid()
        ax.set_title("Rangemeter")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Distance [m]")

    def plot_trajectory(self):
        fig, axes = plt.subplots(4, 1, figsize=(10, 20))

        # Position
        axes[0].plot(self.timestamps, self.trajectory["position"])
        axes[0].set_title("Position (x, y, z)")
        axes[0].set_xlabel("Time [s]")
        axes[0].set_ylabel("Position [m]")
        axes[0].legend(["x", "y", "z"])
        axes[0].grid()

        # Velocity
        axes[1].plot(self.timestamps, self.trajectory["velocity"])
        axes[1].set_title("Velocity (vx, vy, vz)")
        axes[1].set_xlabel("Time [s]")
        axes[1].set_ylabel("Velocity [m/s]")
        axes[1].legend(["vx", "vy", "vz"])
        axes[1].grid()

        # Euler Angles
        axes[2].plot(self.timestamps, self.trajectory["euler_angles"])
        axes[2].set_title("Euler Angles (phi, theta, psi)")
        axes[2].set_xlabel("Time [s]")
        axes[2].set_ylabel("Angle [rad]")
        axes[2].legend(["phi", "theta", "psi"])
        axes[2].grid()

        # Angular Velocity
        axes[3].plot(self.timestamps, self.trajectory["angular_velocity"])
        axes[3].set_title("Angular Velocity (p, q, r)")
        axes[3].set_xlabel("Time [s]")
        axes[3].set_ylabel("Angular Velocity [rad/s]")
        axes[3].legend(["p", "q", "r"])
        axes[3].grid()

        plt.tight_layout()
        plt.show()
    
    def display_event_frames(self, t_start=0, t_end=np.inf, tau=0.1, wait=100, frame_type="standard"):
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
       
                # Show frame
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
        
    def _exponantial_decay_aggregation(self, events) -> np.ndarray:
        '''
        Type of event frame, which consider timestamp of event
        - events: (events_x, events_y, events_polarity, events_ts) 
        '''
        frame = np.ones(self.img_shape, dtype=float) * 127.0

        if events:
            # Assuming thet events are in ascending order by timestamps
            e_x, e_y, e_p, e_ts = events
            max_ts = e_ts[-1]
            delta_t = max_ts - e_ts[0]
        
            if delta_t <= 0:
                # Too few events or timestamps are equal
                for x, y, p in zip(e_x, e_y, e_p):
                    if 0 <= y < self.img_shape[0] and 0 <= x < self.img_shape[1]:
                        frame[y, x] = 255 if p == 1 else 0
            else:
                for x, y, p, ts in zip(e_x, e_y, e_p, e_ts):
                    if 0 <= y < self.img_shape[0] and 0 <= x < self.img_shape[1]:
                        decay = np.exp(-abs(max_ts - ts) / delta_t)
                        value = (p * decay + 1) * 127.5
                        frame[y, x] = np.clip(value, 0, 255)

        return frame.astype(np.uint8)

    def _event_frame_agregation(self, events) -> np.ndarray:
        ''' 
        Standard event frame 
        @param events - (events_x, events_y, events_polarity, events_ts) 
        '''
        frame = np.ones(self.img_shape, dtype=np.uint8) * 127
        e_x, e_y, e_p, _ = events
        for x, y, p in zip(e_x, e_y, e_p):
            if 0 <= y < self.img_shape[0] and 0 <= x < self.img_shape[1]:
                frame[y, x] = 255 if p == 1 else 0

        return frame

    def __is_ba(self, event) -> bool:
        '''
        Background Activity Filter (BAF)
        Zwraca True, jeśli zdarzenie wygląda na tło (szum), False jeśli to "prawdziwe" zdarzenie.
        '''
        x, y, ts = event['x'], event['y'], event['t']
        offset = self.filter_size // 2

        l_bound_y = max(0, y - offset)
        u_bound_y = min(self.img_shape[0], y + offset + 1)
        l_bound_x = max(0, x - offset)
        u_bound_x = min(self.img_shape[1], x + offset + 1)

        sae_window = self.__sae[l_bound_y:u_bound_y, l_bound_x:u_bound_x]

        neighbors = np.sum((ts - sae_window) < self.filter_t) - 1  # odejmujemy 1, by pominąć samo zdarzenie

        return neighbors < self.filter_k