# HandsAnalyzer
A library based on mediapipe to analyze hands posture and gesture.
# Examples
## OpenCv
Examples using opencv window to output stuff.
### hello_hands
A simple code to test the hands detection, showing if it is left or right hand and highliting the hands as well as determining the hand orientation based on the palm landmarks.
### fingers_counter
A code to count fingers for each hand in the image and returns the total count.
No neural nets used. Only simple geometry. So this code is very fast.
### hand_status
A code to get the hand status (closed, helf opened or opened)