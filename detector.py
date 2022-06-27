import copy
import itertools
import pickle
import time

import cv2 as cv
import mediapipe as mp
import numpy as np
import pandas as pd


def main():
    # Initialization ###################################################################################################

    # Initialize camera settings
    webcam = 0
    cap = cv.VideoCapture(webcam)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)

    # Initialize FPS p_time
    p_time = 0

    # Initialize Mediapipe's hand model parameters
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.25,
    )

    # Open & import trained model ######################################################################################
    with open('model/trained_classifier.pkl', 'rb') as f:
        model = pickle.load(f)

    # While in capturing process #######################################################################################
    while True:

        # Press "ESC" key to stop the application
        if cv.waitKey(5) & 0xFF == 27:
            break

        # If frame/image in capture is available, then read it
        available, image = cap.read()
        if not available:
            break

        # Flip and copy the image for debugging
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        # Convert frame image from BGR to RGB for pre-optimization
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Optimize detection process
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        # Calculate and visualize FPS
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        debug_image = draw_fps(debug_image, fps)

        # Visualize info
        cv.putText(debug_image, "* Achmad Mahathir P. (187006041) | Universitas Siliwangi 2022", (323, 470),
                   cv.FONT_HERSHEY_SIMPLEX, 0.3,
                   (0, 0, 0), 2,
                   cv.LINE_AA)
        cv.putText(debug_image, "* Achmad Mahathir P. (187006041) | Universitas Siliwangi 2022", (323, 470),
                   cv.FONT_HERSHEY_SIMPLEX, 0.3,
                   (255, 255, 255), 1,
                   cv.LINE_AA)

        # If the hand is detected: #####################################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Calculate boundaries for bounding box
                bounding_box = calc_bounding_box(debug_image, hand_landmarks)

                # Convert pre-normalized landmark keys into pixels numbering
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Convert into relative coordinates / normalize keys from wrist point
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # Visualize complete hand landmarks
                debug_image = draw_landmarks(debug_image, landmark_list)

                # Try predict hand gesture #############################################################################
                try:
                    hand = pre_processed_landmark_list

                    x = pd.DataFrame([hand])
                    sign_language_class = model.predict(x)[0]
                    sign_language_prob = model.predict_proba(x)[0]
                    print(sign_language_class, sign_language_prob)

                    # Draw "Hand detected" if hand is on screen
                    cv.putText(debug_image, "Hand detected", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.73,
                               (0, 0, 0), 4, cv.LINE_AA)
                    cv.putText(debug_image, "Hand detected", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.73,
                               (255, 255, 255), 2, cv.LINE_AA)

                    # Implement class value in upper description box
                    cv.rectangle(debug_image, (bounding_box[0], bounding_box[1]), (bounding_box[2],
                                                                                   bounding_box[1] - 22), (0, 0, 0), -1)
                    sign_alphabet = sign_language_class.split(' ')[0]
                    cv.putText(debug_image, 'Class : ' + sign_alphabet, (bounding_box[0] + 5, bounding_box[1] - 4),
                               cv.FONT_HERSHEY_SIMPLEX,
                               0.6, (0, 0, 0), 2, cv.LINE_AA)
                    cv.putText(debug_image, 'Class : ' + sign_alphabet, (bounding_box[0] + 5, bounding_box[1] - 4),
                               cv.FONT_HERSHEY_SIMPLEX,
                               0.6, (255, 255, 255), 1, cv.LINE_AA)

                    # Implement probability value in lower description box
                    sign_prob = str(round(sign_language_prob[np.argmax(sign_language_prob)], 2) * 100)
                    cv.rectangle(debug_image, (bounding_box[2], bounding_box[3]), (bounding_box[0],
                                                                                   bounding_box[3] + 22), (0, 0, 0), -1)
                    cv.putText(debug_image, 'Prob : ' + sign_prob + "%", (bounding_box[0] + 5, bounding_box[3] + 17),
                               cv.FONT_HERSHEY_SIMPLEX,
                               0.6, (0, 0, 0), 2, cv.LINE_AA)
                    cv.putText(debug_image, 'Prob : ' + sign_prob + "%", (bounding_box[0] + 5, bounding_box[3] + 17),
                               cv.FONT_HERSHEY_SIMPLEX,
                               0.6, (255, 255, 255), 1, cv.LINE_AA)

                    # Draw bounding box
                    debug_image = draw_bounding_box(True, debug_image, bounding_box)

                # If not detected, then just pass ######################################################################
                finally:
                    pass

        # Output frame #################################################################################################
        cv.imshow('Hand (Fingerspelling) Sign Language Recognition', debug_image)


# Functions ############################################################################################################
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


def draw_fps(image, fps):
    cv.putText(image, "FPS : " + str(int(fps)), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.73, (0, 0, 0), 4,
               cv.LINE_AA)
    cv.putText(image, "FPS : " + str(int(fps)), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.73, (255, 255, 255), 2,
               cv.LINE_AA)

    return image


def draw_bounding_box(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (155, 168, 174), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (188, 202, 208), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (155, 168, 174), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[0]),
                (155, 168, 174), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (155, 168, 174), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (155, 168, 174), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (155, 168, 174), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (155, 168, 174), 2)

        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (227, 232, 234), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (188, 202, 208), 2),
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (227, 232, 234), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (188, 202, 208), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (227, 232, 234), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (188, 202, 208), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (227, 232, 234), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (188, 202, 208), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (227, 232, 234), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


if __name__ == '__main__':
    main()
