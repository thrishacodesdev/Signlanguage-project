import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Setup webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands solution
with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:

    prev_gesture = ""
    last_detected_time = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip and convert the image to RGB
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and detect hands
        results = hands.process(image_rgb)

        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Drawing landmarks on hand
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get coordinates of landmarks
                landmarks = hand_landmarks.landmark

                # Detect fingers up/down logic
                finger_states = []

                # Tip IDs for 5 fingers: [thumb, index, middle, ring, pinky]
                tip_ids = [4, 8, 12, 16, 20]

                for i in tip_ids:
                    # Check if tip is higher (y-coordinate is lower) than joint below
                    if i == 4:  # Thumb (check x axis)
                        if landmarks[i].x < landmarks[i - 1].x:
                            finger_states.append(1)
                        else:
                            finger_states.append(0)
                    else:
                        if landmarks[i].y < landmarks[i - 2].y:
                            finger_states.append(1)
                        else:
                            finger_states.append(0)

                # Convert finger_states to gesture
                gesture = ""

                if finger_states == [0, 1, 0, 0, 0]:
                    gesture = "1"
                elif finger_states == [0, 1, 1, 0, 0]:
                    gesture = "2"
                elif finger_states == [0, 1, 1, 1, 0]:
                    gesture = "3"
                elif finger_states == [0, 1, 1, 1, 1]:
                    gesture = "4"
                elif finger_states == [1, 1, 1, 1, 1]:
                    gesture = "5"
                elif finger_states == [0, 0, 0, 0, 0]:
                    gesture = "Fist"
                elif finger_states == [1, 0, 0, 0, 0]:
                    gesture = "Thumbs Up"
                elif finger_states == [1, 1, 0, 0, 1]:
                    gesture = "Spiderman"
                else:
                    gesture = ""

                # Display gesture on screen
                current_time = time.time()
                if gesture and gesture != prev_gesture:
                    prev_gesture = gesture
                    last_detected_time = current_time
                elif current_time - last_detected_time > 2:
                    prev_gesture = ""

                if prev_gesture:
                    cv2.putText(image, f"Gesture: {prev_gesture}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # Show the image
        cv2.imshow('Sign Language to Text', image)

        # Exit on pressing 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
