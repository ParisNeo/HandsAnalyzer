"""=== hello_hands =>

    Author : Saifeddine ALOUI
    Description :
        A code to test HandsAnalyzer: Extract hands landmarks from a realtime video input
<================"""
from HandsAnalyzer import HandsAnalyzer, Hand
import numpy as np
import cv2

# open camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Build a window
cv2.namedWindow('Face Mesh', flags=cv2.WINDOW_NORMAL)
cv2.resizeWindow('Face Mesh', (640,480))

# Build face analyzer while specifying that we want to extract just a single face
ha = HandsAnalyzer(max_nb_hands=3)

# Main Loop
while cap.isOpened():
    # Read image
    success, image = cap.read()
    # Opencv uses BGR format while mediapipe uses RGB format. So we need to convert it to RGB before processing the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Process the image to extract hands and draw the lines
    ha.process(image)
    # If there are some hands then process
    if ha.nb_hands>0:
        for i in range(ha.nb_hands):
            hand = ha.hands[i]
            # Draw the landmarks
            hand.draw_landmarks(image, thickness=3)
            # Draw a bounding box
            hand.draw_bounding_box(image,text="left" if hand.is_left else "right")
            pos, ori = hand.get_hand_posture()
            if pos is not None:
                hand.draw_reference_frame(image, pos, ori, origin=hand.get_landmark_pos(0))
    # Show the image
    try:
        cv2.imshow('Face Mesh', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    except Exception as ex:
        print(ex)
    
    # Wait for key stroke for 5 ms
    wk = cv2.waitKey(5)
    if wk & 0xFF == 27: # If escape is pressed then return
        break
    if wk & 0xFF == 27: # If escape is pressed then return
        break


