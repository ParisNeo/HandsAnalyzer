"""=== hello_hands =>

    Author : Saifeddine ALOUI
    Description :
        A code to test HandsAnalyzer: Extract hands landmarks from a realtime video input
<================"""
from HandsAnalyzer import HandsAnalyzer, Hand
from HandsAnalyzer.helpers.geometry.orientation import orientation2Euler
import numpy as np
import cv2
from pathlib import Path

# open camera
cap = cv2.VideoCapture(0)

# Build a window
cv2.namedWindow('Hello hands', flags=cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hello hands', (640,480))

# Build face analyzer while specifying that we want to extract just a single face
ha = HandsAnalyzer(max_nb_hands=3)
y,p,r=0,0,0
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
            # Find hand posture
            pos, ori = hand.get_hand_posture()
            if pos is not None:
                # Get the center of the palm
                center=hand.get_landmarks_pos(hand.palm_indices).mean(axis=0)
                # Draw the reference frame
                hand.draw_reference_frame(image, pos, ori, origin=center)
                # Get yaw puitch and roll
                y,p,r = orientation2Euler(ori)
            # Draw a bounding box
            hand.draw_bounding_box(image,text=f"left {y:.2f},{p:.2f},{r:.2f}" if hand.is_left else f"right {y:.2f},{p:.2f},{r:.2f}")

    # Show the image
    try:
        cv2.imshow('Hello hands', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    except Exception as ex:
        print(ex)
    
    # Wait for key stroke for 5 ms
    wk = cv2.waitKey(5)
    if wk & 0xFF == 27: # If escape is pressed then return
        break
    if wk & 0xFF == 115: # If s is pressed then take a snapshot
        sc_dir = Path(__file__).parent/"screenshots"
        if not sc_dir.exists():
            sc_dir.mkdir(exist_ok=True, parents=True)
        i = 1
        file = sc_dir /f"sc_{i}.jpg"
        while file.exists():
            i+=1
            file = sc_dir /f"sc_{i}.jpg"
        cv2.imwrite(str(file),cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        print(hand.get_landmarks_pos(hand.palm_indices))
        print("Shot")



