import copy

import cv2 as cv
import csv
import time
import mediapipe as mp
import numpy as np
import itertools

from collections import deque


def main():
    # Initials ##################################################################
    webcam = 0
    cap_width = 960
    cap_height = 540

    p_time = 0
    c_time = 0

    # Camera preparation ########################################################
    cap = cv.VideoCapture(webcam)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Mediapipe hand model load #################################################
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    # Coordinate history ########################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # During capturing process ##################################################
    while True:
        # FPS measurement #######################################################
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        # Exit argument #########################################################
        if cv.waitKey(5) & 0xFF == 27:  # ESC key.
            break

        # If frame in capture is available: #####################################
        available, image = cap.read()
        if not available:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Apply Mediapipe into detection ########################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Optimize detection process ############################################
        image.flags.writeable = False
        results = hands.process(image)

        # If the hand is detected: ##############################################
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                bounding_box = calc_bounding_box(debug_image, hand_landmarks)

                # Landmark calculation by actual image size
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion into relative coordinates / normalized coordinates (wrist point)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                print(pre_processed_landmark_list)

                # Write to csv dataset
                # logging_csv("A", pre_processed_landmark_list)

                # Draw into image output
                mp_drawing.draw_landmarks(
                    debug_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        # Output ################################################################
        cv.imshow('Hand Gesture Recognition', debug_image)


def calc_bounding_box(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def logging_csv(feature, landmark_list):

    csv_path = 'model/keypoint.csv'
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([feature, *landmark_list])

    return


if __name__ == '__main__':
    main()