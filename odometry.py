import numpy as np
import cv2
import imutils
import os
from os.path import join
import matplotlib.pyplot as plt

DATASET_DIR = 'data/'

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 1000  # Było 1500, ale tu mamy mały obrazek

class PinholeCamera:
    def __init__(self, width, height, fx, fy, cx, cy, 
                k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion = (abs(k1) > 0.0000001)
        self.d = [k1, k2, p1, p2, k3]
        
        
lk_params = dict(winSize  = (21, 21), criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

def featureTracking(image_ref, image_cur, px_ref):
    
    px_cur, status, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, nextPts=None, winSize=lk_params['winSize'], maxLevel=3, criteria=lk_params['criteria'])
    status = status.ravel()
    px_ref_pos_status = px_ref[status==1]
    px_cur_pos_status = px_cur[status==1]

    return px_ref_pos_status, px_cur_pos_status


def rotation_matrix_from_euler(phi, theta, psi):
    # phi = roll, theta = pitch, psi = yaw
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]
    ])

    R_y = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    R_z = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])

    R = R_z @ R_y @ R_x
    return R


class VisualOdometry:
    '''
    Visual odometry from sequence of frames and datsa from rangemeter
    
    Parameters
    - `cam`: PinholeCamera - intrisic camera parameters
    - `trajectory`: dict - trajectory dictionary from LanderData
    - `rangemeter`: dict - rangemeter dictionary from LanderData
    - `n_frames` : int - number of all frames in analizes sequence (number of frames registered during rangemeter and imu work)
    
    Note:
    Number of frames must be bigger or equal to number of `rangemeter` entries in same time period!
    To get best results it should be diviser of this value.
    '''
    def __init__(self, cam, trajectory, rangemeter, n_frames):
        self.frame_stage = 0
        self.cur_R = None
        self.cur_t = None
        
        self.__cam = cam
        self.__new_frame = None
        self.__last_frame = None
        self.__px_ref = None
        self.__px_cur = None
        self.__focal = cam.fx
        self.__pp = (cam.cx, cam.cy)

        self.__detector = cv2.FastFeatureDetector_create(
            threshold=25, nonmaxSuppression=True)

        self.__rangemeter = rangemeter
        self.__trajectory = trajectory
        
        self.__n_frames = n_frames

    def getAbsoluteScale(self, frame_id):  # scale from rangemeter and imu angles
        # rangemeter measures alnog local z axis - altitude
        range_dir_local = np.array([0, 0, 1])
        
        imu_idx = self.__trajectory_idx_by_frame(frame_id)
    
        phi = self.__trajectory["euler_angles"][imu_idx,0]  # roll
        theta = self.__trajectory["euler_angles"][imu_idx,1]  # pitch
        psi = self.__trajectory["euler_angles"][imu_idx,2]  # yaw
        
        # to global coordinates
        R = rotation_matrix_from_euler(phi, theta, psi)
        range_dir_global = R @ range_dir_local

        range_idx = self.__rangemeter_idx_by_frame(frame_id)
        range_prev_idx = self.__rangemeter_idx_by_frame(frame_id - 1)
        
        range_curr = self.__rangemeter["distance"][range_idx]
        range_prev = self.__rangemeter["distance"][range_prev_idx]
        
        delta_r = range_curr - range_prev
        movement_vector = delta_r * range_dir_global

        return np.linalg.norm(movement_vector)

    def processFirstFrame(self):
        keypoints = self.__detector.detect(self.__new_frame)
        self.__px_ref = np.array([kp.pt for kp in keypoints], np.float32)
        self.frame_stage = STAGE_SECOND_FRAME

    def processSecondFrame(self):
        keypoints1, keypoints2 = featureTracking(self.__last_frame, self.__new_frame, self.__px_ref)
        E_mat, _ = cv2.findEssentialMat(points1=keypoints2, points2=keypoints1, method=cv2.RANSAC, prob=0.999, threshold=1.0, pp=self.__pp, focal=self.__focal)
        _, self.cur_R, self.cur_t, _ = cv2.recoverPose(E_mat, keypoints2, keypoints1, focal=self.__focal, pp=self.__pp)
        self.frame_stage = STAGE_DEFAULT_FRAME
        self.__px_ref = keypoints2

    def processFrame(self, frame_id):
        keypoints1, keypoints2 = featureTracking(self.__last_frame, self.__new_frame, self.__px_ref)
        
        self.__px_cur = keypoints2
        
        E_mat, _ = cv2.findEssentialMat(points1=keypoints2, points2=keypoints1, method=cv2.RANSAC, prob=0.999, threshold=1.0, pp=self.__pp, focal=self.__focal)
        _, R, t, _ = cv2.recoverPose(E_mat, keypoints2, keypoints1, focal=self.__focal, pp=self.__pp)
        scale = self.getAbsoluteScale(frame_id)
        # Aktualizacja R i t
        if scale > 0.1:
            self.cur_t += np.dot(self.cur_R, t) * scale
            self.cur_R = np.dot(R, self.cur_R)
        
        if len(self.__px_ref) < kMinNumFeature:
            keypoints = self.__detector.detect(self.__new_frame)
            self.__px_cur = np.array([kp.pt for kp in keypoints], np.float32)
            
        self.__px_ref = self.__px_cur

    def update(self, img, frame_id):
        assert(img.ndim==2 and img.shape[0]==self.__cam.height and img.shape[1]==self.__cam.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"

        self.__new_frame = img
        if self.frame_stage == STAGE_FIRST_FRAME:
            self.processFirstFrame()
        elif self.frame_stage == STAGE_SECOND_FRAME:
            self.processSecondFrame()
        elif self.frame_stage == STAGE_DEFAULT_FRAME:
            self.processFrame(frame_id)
            
        self.__last_frame = self.__new_frame
        
    def __rangemeter_idx_by_frame(self, frame_id):
        '''
        Assuming equal periods between each frame
        '''
        rangemeter_entries = len(self.__rangemeter["distance"])
        
        return int(np.round(frame_id * (rangemeter_entries / self.__n_frames)))
        
    def __trajectory_idx_by_frame(self, frame_id):
        '''
        Assuming equal periods between each frame
        '''
        trajectory_entries = len(self.__trajectory["position"])
        
        return int(np.round(frame_id * (trajectory_entries / self.__n_frames)))



