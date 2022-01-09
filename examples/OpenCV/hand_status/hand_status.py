"""=== hello_hands =>

    Author : Saifeddine ALOUI
    Description :
        A code to test HandsAnalyzer: Extract hands landmarks from a realtime video input
<================"""
from HandsAnalyzer import HandsAnalyzer, Hand
from HandsAnalyzer.helpers.geometry.orientation import orientation2Euler
from HandsAnalyzer.helpers.geometry.euclidian import get_alignment_coefficient
import numpy as np
import cv2
from pathlib import Path

# open camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Build a window
cv2.namedWindow('Hand Status', flags=cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hand Status', (640,480))

# Build face analyzer while specifying that we want to extract just a single face
ha = HandsAnalyzer(max_nb_hands=6)
hand_status_names = ["Closed","Half Closed","Opened"]
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
            status = hand.getHandStatus()

            hand.draw_bounding_box(image,text=f"left {hand_status_names[status]}" if hand.is_left else f"right {hand_status_names[status]}")
    # Show the image
    try:
        cv2.imshow('Hand Status', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
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



