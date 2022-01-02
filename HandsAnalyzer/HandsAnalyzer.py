# -*- coding: utf-8 -*-
"""=== Hand Analyzer =>
    Module : FaceAnalyzer
    Author : Saifeddine ALOUI (ParisNeo)
    Licence : MIT
    Description :
        Main module. FaceAnalyzer analyzes an image and extract hands by creating instances of Hand in its attribute Faces
<================"""

from typing import NamedTuple, Tuple
import numpy as np
import mediapipe as mp
import cv2
import math
import time
from PIL import Image
from scipy.signal import butter, filtfilt

from .Hand import Hand

class HandsAnalyzer():
    """A class that analyzes the facial components
    """

    def __init__(self, max_nb_hands=1, image_shape: tuple = (640, 480),  static_image_mode:bool=True):
        """Creates an instance of the FaceAnalyzer object

        Args:
            max_nb_hands (int,optional) : The maximum number of hands to be detected by the mediapipe library
            image_shape (tuple, optional): The shape of the image to be processed. Defaults to (480, 640).
        """
        self.mp_drawing = mp.solutions.drawing_utils
        self.fmd = mp.solutions.hands.Hands(static_image_mode=static_image_mode, max_num_hands=max_nb_hands)
        self.max_nb_hands = max_nb_hands

        self.hands = [Hand(image_shape=image_shape) for i in range(max_nb_hands)]
        self.image_shape = image_shape
        self.image = None
        self.results = None
        self.found_hands = False
        self.found_hands = False
        self.nb_hands = 0        

    @property
    def image_size(self)->tuple:
        """A property to image size

        Returns:
            tuple: The image size
        """
        return self.image_shape

    @image_size.setter
    def image_size(self,new_shape:tuple):
        self.image_shape=new_shape
        for hand in self.hands:
            hand.image_shape=new_shape

    def process(self, image: np.ndarray) -> NamedTuple:
        """Processes an image and extracts the hands

        Args:
            image (np.ndarray): The image to extract hands from

        Returns:
            NamedTuple: The result of extracting the image
        """
        # Process the image
        results = self.fmd.process(image)

        # Keep a local reference to the image
        self.image = image

        # If hands found
        if results.multi_hand_landmarks is not None:
            self.found_hands = True
            self.nb_hands = len(results.multi_hand_landmarks)
        else:
            self.found_hands = False
            self.nb_hands = 0
            return
    
        # Update hands
        for i, lm in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[i].classification[0].label.lower()
            if i>=len(self.hands):
                continue
            self.hands[i].update(handedness=="left", lm)
        for i in range(len(results.multi_hand_landmarks),self.max_nb_hands):
            self.hands[i].update(True, None)

        self.results = results
    @staticmethod
    def from_image(file_name:str, max_nb_hands:int=1, image_shape:tuple=(640, 480)):
        """Opens an image and extracts a hand from it

        Args:
            file_name (str)                 : The name of the image file containing one or multiple hands
            max_nb_hands (int, optional)    : The maximum number of hands to extract. Defaults to 1
            image_shape (tuple, optional)   : The image shape. Defaults to (640, 480)

        Returns:
            An instance of FaceAnalyzer: A hand analyzer containing all processed hands out of the image. Ile image can be found at fa.image
        """
        fa = HandsAnalyzer(max_nb_hands=max_nb_hands)
        image = Image.open(file_name)
        image = image.resize(image_shape)
        npImage = np.array(image)[...,:3]
        fa.process(npImage)
        return fa
