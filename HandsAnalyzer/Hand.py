# -*- coding: utf-8 -*-
"""=== Hand Analyzer =>
    Module : Hand
    Author : Saifeddine ALOUI (ParisNeo)
    Licence : MIT
    Description :
        Hand data holder (landmarks, posture ...). Allows the extraction of multiple facial features out of landmarks.
<================"""


import re
from typing import NamedTuple, Tuple
import numpy as np
import mediapipe as mp
import cv2
from numpy import linalg
from numpy.lib.type_check import imag
from scipy.signal import butter, filtfilt
import math
import time
from PIL import Image
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as R


from .helpers.geometry.euclidian import buildCameraMatrix, get_plane_infos, get_z_line_equation, get_plane_line_intersection
from .helpers.geometry.orientation import rotateLandmarks

# Get an instance of drawing specs to be used for drawing masks on hands
DrawingSpec =  mp.solutions.drawing_utils.DrawingSpec
class Hand():    
    """Hand is the class that provides operations on hand landmarks.
    It is extracted by the hand analyzer and could then be used for multiple hand features extraction purposes
    """


    # Key landmark indices
    palm_indices = [
                        0,
                        1,
                        5,
                        9,
                        13,
                        17
                    ]
    left_palm_reference_positions = np.array([
                    [0,0,0],
                    [50,0,0],
                    [58,75,0],
                    [39,78,0],
                    [20,80,0],
                    [0,80,0],
                ])
    right_palm_reference_positions = np.array([
                    [0,0,0],
                    [-50,0,0],
                    [-58,75,0],
                    [-39,78,0],
                    [-20,80,0],
                    [0,80,0],
                ])

    def __init__(self, is_left:bool=True, landmarks:NamedTuple = None, image_shape: tuple = (640, 480)):
        """Creates an instance of Hand

        Args:
            is_left (bool): if true, this is a left hand else this is a right hand.
            landmarks (NamedTuple, optional): Landmarks object extracted by mediapipe tools
            image_shape (tuple, optional): The width and height of the image used to extract the hand. Required to get landmarks in the right pixel size (useful for hand copying and image operations). Defaults to (480, 640).
        """
        self.image_shape = image_shape

        if type(landmarks)==np.ndarray:
            self.npLandmarks=landmarks
        else:
            self.update(is_left,landmarks)

        # Left or right hands have different orientations
        self.is_left = is_left


        # Initialize hand information
        self.pos = None
        self.ori = None

        self.mp_drawing = mp.solutions.drawing_utils



    @property
    def ready(self)->bool:
        """Returns if the hand has landmarks or not

        Returns:
            bool: True if the hand has landmarks
        """
        return self.landmarks is not None

    def update(self, is_left:bool, landmarks:NamedTuple)->None:
        """Updates the landmarks of the hand

        Args:
            is_left (bool): if true, this is a left hand else this is a right hand.
            landmarks (NamedTuple): The new landmarks
        """
        self.is_left = is_left
        if landmarks is not None:
            self.landmarks = landmarks
            self.npLandmarks = np.array([[lm.x * self.image_shape[0], lm.y * self.image_shape[1], lm.z * self.image_shape[0]] for lm in landmarks.landmark])
        else:
            self.landmarks = None
            self.npLandmarks = np.array([])


    def get_landmark_pos(self, index) -> Tuple:
        """Recovers the position of a landmark from a results array

        Args:
            index (int): Index of the landmark to recover

        Returns:
            Tuple: Landmark 3D position in image space
        """

        # Assertion to verify that the hand object is ready
        assert self.ready, "Hand object is not ready. There are no landmarks extracted."

        lm = self.npLandmarks[index, ...]
        return np.array([lm[0], lm[1], lm[2]])



    def get_landmarks_pos(self, indices: list) -> np.ndarray:
        """Recovers the position of a landmark from a results array

        Args:
            indices (list): List of indices of landmarks to extract

        Returns:
            np.ndarray: A nX3 array where n is the number of landmarks to be extracted and 3 are the 3 cartesian coordinates
        """

        # Assertion to verify that the hand object is ready
        assert self.ready, "Hand object is not ready. There are no landmarks extracted."

        return self.npLandmarks[indices,...]


    def draw_landmark_by_index(self, image: np.ndarray, index: int, color: tuple = (255, 0, 0), radius: int = 5, thickness:int=1) -> np.ndarray:
        """Draw a landmark on an image from landmark index

        Args:
            image (np.ndarray): Image to draw the landmark on
            index (int): Index of the landmark
            color (tuple, optional): Color of the landmark. Defaults to (255, 0, 0).
            radius (int, optional): Radius of the circle to draw the landmark. Defaults to 5.
            thickness (int, optional): Thickness of the line to draw the landmark. Defaults to 5.

        Returns:
            np.ndarray: Output image
        """
        pos = self.npLandmarks[index,:]
        return cv2.circle(
            image,(int(pos[0]), int(pos[1])), radius, color, thickness
        )


    def draw_landmarks(self, image: np.ndarray, landmarks: np.ndarray=None, radius:int=1, color: tuple = (255, 0, 0), thickness: int = 1, link=True) -> np.ndarray:
        """Draw a list of landmarks on an image

        Args:
            image (np.ndarray): Image to draw the contour on
            landmarks (np.ndarray): a nX3 ndarray containing the positions of the landmarks. Defaults to None (use all landmarks).
            radius (int, optional): Radius of the circle to draw the landmark. Defaults to 5.
            color (tuple, optional): Color of the landmark. Defaults to (255, 0, 0).
            thickness (int, optional): Thickness of the line to draw the landmark. Defaults to 5.


        Returns:
            np.ndarray: The image with the contour drawn on it
        """
        if landmarks is None:
            landmarks = self.npLandmarks
            
        lm_l=landmarks.shape[0]
        for i in range(lm_l):
            image = cv2.circle(image, (int(landmarks[i,0]), int(landmarks[i,1])), radius,color, thickness)
            if link:
                if i not in [4,8,12,16,20]:
                    image = cv2.line(image, (int(landmarks[i,0]), int(landmarks[i,1])),(int(landmarks[(i+1)%lm_l,0]), int(landmarks[(i+1)%lm_l,1])),color, thickness)
        return image

    def draw_landmark(self, image: np.ndarray, pos: tuple, color: tuple = (255, 0, 0), radius: int = 5, thickness:int=1) -> np.ndarray:
        """Draw a landmark on an image

        Args:
            image (np.ndarray): Image to draw the landmark on
            pos (tuple): Position of the landmark
            color (tuple, optional): Color of the landmark. Defaults to (255, 0, 0).
            radius (int, optional): Radius of the circle to draw the landmark. Defaults to 5.
            thickness (int, optional): Thickness of the line to draw the landmark. Defaults to 5.

        Returns:
            np.ndarray: Output image
        """
        return cv2.circle(
            image,(int(pos[0]), int(pos[1])), radius, color, thickness
        )

    def draw_contour(self, image: np.ndarray, contour: np.ndarray, color: tuple = (255, 0, 0), thickness: int = 1, isClosed:bool = True) -> np.ndarray:
        """Draw a contour on an image

        Args:
            image (np.ndarray): Image to draw the contour on
            contour (np.ndarray): a nX3 ndarray containing the positions of the landmarks
            color (tuple, optional): Color of the landmark. Defaults to (255, 0, 0).
            thickness (int, optional): Thickness of the line to draw the landmark. Defaults to 5.
            isClosed (bool, optional): If True, the contour will be closed, otherwize it will be kept open. Defaults to True 


        Returns:
            np.ndarray: The image with the contour drawn on it
        """

        pts = np.array([[int(p[0]), int(p[1])] for p in contour.tolist()]).reshape((-1, 1, 2))
        return cv2.polylines(image, [pts], isClosed, color, thickness)

    def get_hand_posture(self, camera_matrix:np.ndarray = None, dist_coeffs:np.ndarray=np.zeros((4,1)))->tuple:
        """Gets the posture of the head (position in cartesian space and Euler angles)
        Args:
            camera_matrix (int, optional)       : The camera matrix built using buildCameraMatrix Helper function. Defaults to a perfect camera matrix 
            dist_coeffs (np.ndarray, optional)) : The distortion coefficients of the camera
        Returns:
            tuple: (position, orientation) the orientation is either in compact rodriguez format (angle * u where u is the rotation unit 3d vector representing the rotation axis). Feel free to use the helper functions to convert to angles or matrix
        """

        # Assertion to verify that the hand object is ready
        assert self.ready, "Hand object is not ready. There are no landmarks extracted."

        if camera_matrix is None:
            camera_matrix= buildCameraMatrix()

        # Use opencv's PnPsolver to solve the rotation problem

        face_2d_positions = self.npLandmarks[self.palm_indices,:2]
        if self.is_left:
            (success, face_ori, face_pos, _) = cv2.solvePnPRansac(
                                                        self.left_palm_reference_positions.astype(np.float),
                                                        face_2d_positions.astype(np.float), 
                                                        camera_matrix, 
                                                        dist_coeffs,
                                                        flags=cv2.SOLVEPNP_ITERATIVE)
        else:
            (success, face_ori, face_pos, _) = cv2.solvePnPRansac(
                                                        self.left_palm_reference_positions.astype(np.float),
                                                        face_2d_positions.astype(np.float), 
                                                        camera_matrix, 
                                                        dist_coeffs,
                                                        flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            return None, None

        # save posture
        self.pos = face_pos
        self.ori = face_ori

        return face_pos, face_ori


    def rect_contains(self, rect:tuple, point:tuple)->bool:
        """Tells whether a point is inside a rectangular region

        Args:
            rect (tuple): The rectangle coordiantes (topleft , bottomright)
            point (tuple): The point position (x,y)

        Returns:
            bool: True if the point is inside the rectangular region
        """
        if point[0] < rect[0]:
            return False
        elif point[1] < rect[1]:
            return False
        elif point[0] > rect[2]:
            return False
        elif point[1] > rect[3]:
            return False
        return True


    def getHandBox(self, image:np.ndarray, landmark_indices:list=None, margins=(0,0,0,0))->np.ndarray:
        """Gets an image of the hand extracted from the original image (simple box extraction which will extract some of the background)

        Args:
            image (np.ndarray): Image to extract the hand from
            src_triangles (list): The delaulay triangles indices (look at triangulate)
            landmark_indices (list, optional): The list of landmarks to be used (the same list used for the triangulate method that allowed the extraction of the triangles). Defaults to None.

        Returns:
            np.ndarray: Hand drawn on a black background (the size of the image is equal of that of the hand in the original image)
        """

        # Assertion to verify that the hand object is ready
        assert self.ready, "Hand object is not ready. There are no landmarks extracted."

        if landmark_indices is None:
            landmarks = self.npLandmarks[:, :2]
        else:
            landmarks = self.npLandmarks[landmark_indices, :2]
        p1 = landmarks.min(axis=0)-np.array(margins[0:2])
        p2 = landmarks.max(axis=0)+np.array(margins[2:4])
        return image[int(p1[1]):int(p2[1]),int(p1[0]):int(p2[0])]

    def draw_bounding_box(self, image:np.ndarray, color:tuple=(255,0,0), thickness:int=1, text=None):
        """Draws a bounding box around the hand

        Args:
            image (np.ndarray): The image on which we will draw the bounding box
            color (tuple, optional): The color of the bounding box. Defaults to (255,0,0).
            thickness (int, optional): The line thickness. Defaults to 1.
        """
        pt1 = self.npLandmarks.min(axis=0)
        pt2 = self.npLandmarks.max(axis=0)
        cv2.rectangle(image, (int(pt1[0]),int(pt1[1])), (int(pt2[0]),int(pt2[1])), color, thickness)
        if text is not None:
            cv2.putText(image, text, (int(pt1[0]),int(pt1[1]-20)),cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness)


    def draw_reference_frame(self, image:np.ndarray, pos: np.ndarray, ori:np.ndarray, origin:np.ndarray=None, line_length:int=50)->None:
        """Draws a reference frame at a sprecific position

        Args:
            image (np.ndarray): The image to draw the reference frame on.
            pos (np.ndarray): The real 3D position of the frame reference
            ori (np.ndarray): The orientation of the frame in compressed axis angle format
            origin (np.ndarray): The origin in camera frame where to draw the frame
            translation (np.ndarray, optional): A translation vector to draw the frame in a different position tha n the origin. Defaults to None.
            line_length (int, optional): The length of the frame lines (X:red,y:green,z:blue). Defaults to 50.
        """

        #Let's project three vectors ex,ey,ez to form a frame and draw it on the nose
        (center_point2D_x, jacobian) = cv2.projectPoints(np.array([(0, 0.0, 0.0)]), ori, pos, buildCameraMatrix(), np.zeros((4,1)))
        (end_point2D_x, jacobian) = cv2.projectPoints(np.array([(line_length, 0.0, 0.0)]), ori, pos, buildCameraMatrix(), np.zeros((4,1)))
        (end_point2D_y, jacobian) = cv2.projectPoints(np.array([(0.0, line_length, 0.0)]), ori, pos, buildCameraMatrix(), np.zeros((4,1)))
        (end_point2D_z, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, line_length)]), ori, pos, buildCameraMatrix(), np.zeros((4,1)))

        p1 = ( int(center_point2D_x[0][0][0]), int(center_point2D_x[0][0][1]))         
        p2_x = ( int(end_point2D_x[0][0][0]), int(end_point2D_x[0][0][1]))         
        p2_y = ( int(end_point2D_y[0][0][0]), int(end_point2D_y[0][0][1]))         
        p2_z = ( int(end_point2D_z[0][0][0]), int(end_point2D_z[0][0][1]))   

        """
        """
        if origin is not None:
            do = (int(origin[0]- p1[0]), int(origin[1]- p1[1]))
            p1=  (int(origin[0]), int(origin[1]))

            p2_x= (p2_x[0]+do[0],p2_x[1]+do[1])

            p2_y= (p2_y[0]+do[0],p2_y[1]+do[1])

            p2_z= (p2_z[0]+do[0],p2_z[1]+do[1])

        cv2.line(image, p1, p2_x, (255,0,0), 2)   
        cv2.line(image, p1, p2_y, (0,255,0), 2)   
        cv2.line(image, p1, p2_z, (0,0,255), 2)
    

    def is_pointing_to_2d_region(self, region:tuple, pos: np.ndarray, ori:np.ndarray):
        """Returns weather the hand or eye is pointing inside a 2d region represented by the polygon 

        Args:
            region (tuple): A list of points in form of ndarray that represent the region (all points should belong to the same plan)
            pos (np.ndarray): The position of the hand or eye
            ori (np.ndarray): The orientation of the hand or eye

        Returns:
            boolean: If true then the hand or eye is pointing to that region else false
        """
        assert(len(region)>=3,"Region should contain at least 3 points")
        # Copy stuff
        region = region.copy()
        # First find the pointing line, and the plan on which the region is selected
        pl = get_plane_infos(region[0],region[1],region[2])
        e1 = pl[2]
        e2 = pl[3]
        ln = get_z_line_equation(pos, ori)
        p, p2d = get_plane_line_intersection(pl, ln)
        # Lets put all the points of the region inside the 2d plane
        for i in range(len(region)):
            region[i]=np.array([np.dot(region[i], e1), np.dot(region[i], e2)])

        # Now let's check that the poit is inside the region
        in_range=True
        for i in range(len(region)):
            AB = region[(i+1)%len(region)]-region[i]
            AP = p2d-region[i]
            c = np.cross(AB, AP)
            if i==0:
                if c>=0:
                    pos=True
                else:
                    pos=False
            else:
                if c>=0 and pos==False:
                    in_range = False
                    break
                elif c<0 and pos==True:
                    in_range = False
                    break
        
        return in_range

        