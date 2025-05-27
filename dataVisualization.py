import numpy as np
import matplotlib.pyplot as plt
import cv2


event_dtype = np.dtype([
    ('x', np.uint16),
    ('y', np.uint16),
    ('p', np.uint8),
    ('t', np.uint64)
])

class LanderData:
    def __init__(self, npz_path):
        self.npz_path = npz_path
        self._load_data()

    def _load_data(self):
        data = np.load(self.npz_path, allow_pickle=True)
        
        events_array = data["events"]
        events_np = np.array(events_array, dtype=event_dtype)
        self.events = events_np
        
        self.timestamps = data["timestamps"]
        self.trajectory = {
            "position": data["traj"][:, 0:3],
            "velocity": data["traj"][:, 3:6],
            "euler_angles": data["traj"][:, 6:9],
            "angular_velocity": data["traj"][:, 9:12],
        }
        self.rangemeter = {
            "time": data["range_meter"][:, 0],
            "distance": data["range_meter"][:, 1],
        }

    def summary(self):
        print(f"Loaded LanderData from: {self.npz_path}")
        print(f"Events: {len(self.events)} entries")
        print(f"Timestamps: {len(self.timestamps)} entries")
        print(f"Trajectory: {len(self.trajectory['position'])} entries")
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

    
    def display_event_frames(self, t_start=0, t_end=np.inf, tau=0.1, img_shape=(200, 200), wait=100):
        """
        Displays event frames between t_start and t_end with temporal resolution `tau`.
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
        temp_x, temp_y, temp_p = [], [], []
        start_time = ts[0]
        frame_count = 0

        for i in range(len(ts)):
            t = ts[i]
            temp_x.append(xs[i])
            temp_y.append(ys[i])
            temp_p.append(ps[i])

            if t - start_time >= tau:
                # Generate frame
                frame = np.ones(img_shape, dtype=np.uint8) * 127
                for x, y, p in zip(temp_x, temp_y, temp_p):
                    if 0 <= y < img_shape[0] and 0 <= x < img_shape[1]:
                        frame[y, x] = 255 if p == 1 else 0

                # Show frame
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

        print(f"\nDisplayed {frame_count} event frames.")
        cv2.destroyAllWindows()