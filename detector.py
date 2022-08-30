import copy
import itertools
import pickle
import time

import cv2 as cv
import mediapipe as mp
import numpy as np
import pandas as pd

# Initialization #######################################################################################################

# Initialize colors
black = (0, 0, 0)
grey_shade1 = (155, 168, 174)
grey_shade2 = (188, 202, 208)
grey_shade3 = (227, 232, 234)
white = (255, 255, 255)

# Initialize camera settings
webcam = 0
cap = cv.VideoCapture(webcam)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)

# Initialize misc.
init_prev_time = 0
escape_key = 27

# Initialize Mediapipe's hand model parameters
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.25,
)


# Main program #########################################################################################################
def main():
    previous_time = init_prev_time

    # Open & import trained model
    with open('model/trained_classifier.pkl', 'rb') as f:
        model = pickle.load(f)

    # While in capturing process
    while True:

        # Application stops when "ESC" key is pressed
        if cv.waitKey(5) & 0xFF == escape_key:
            break

        # If frame/image in capture is not available left, then stop the application
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
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        debug_image = draw_fps(debug_image, fps)

        # Visualize student info
        debug_image = draw_student_info(debug_image)

        # If the hand is detected: #####################################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Calculate boundaries for bounding box
                bounding_box = calc_bounding_box(debug_image, hand_landmarks)

                # Convert pre-normalized landmark keys into pixels numbering
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Convert into relative coordinates / normalize keys from wrist point
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # Visualize complete hand landmarks
                debug_image = draw_landmarks(debug_image, landmark_list)

                # Try predict hand gesture and: ########################################################################
                try:
                    hand = pre_processed_landmark_list

                    data_frame = pd.DataFrame([hand])
                    sign_language_class = model.predict(data_frame)[0]
                    sign_language_prob = model.predict_proba(data_frame)[0]
                    print(sign_language_class, sign_language_prob)

                    # Draw "Hand detected" description
                    debug_image = draw_hand_detected(debug_image)

                    # Draw bounding box with descriptions
                    debug_image = draw_upper_bound_desc(debug_image, bounding_box, sign_language_class)
                    debug_image = draw_bounding_box(True, debug_image, bounding_box)
                    debug_image = draw_lower_bound_desc(debug_image, bounding_box, sign_language_prob)

                # Finally if not detected, then just pass ##############################################################
                finally:
                    pass

        # Output frame #################################################################################################
        cv.imshow('Hand (Fingerspelling) Sign Language Recognition', debug_image)


# Calculation functions ################################################################################################
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


# Cosmetic functions ###################################################################################################
def draw_student_info(image):

    x_position = 323
    y_position = 470
    font_size = 0.3
    black_thickness = 2
    white_thickness = 1

    cv.putText(image, "* Achmad Mahathir P. (187006041) | Universitas Siliwangi 2022", (x_position, y_position),
               cv.FONT_HERSHEY_SIMPLEX, font_size,
               black, black_thickness,
               cv.LINE_AA)
    cv.putText(image, "* Achmad Mahathir P. (187006041) | Universitas Siliwangi 2022", (x_position, y_position),
               cv.FONT_HERSHEY_SIMPLEX, font_size,
               white, white_thickness,
               cv.LINE_AA)

    return image


def draw_fps(image, fps):

    x_position = 10
    y_position = 30
    font_size = 0.73
    black_thickness = 4
    white_thickness = 2

    cv.putText(image, "FPS : " + str(int(fps)), (x_position, y_position), cv.FONT_HERSHEY_SIMPLEX, font_size, black,
               black_thickness,
               cv.LINE_AA)
    cv.putText(image, "FPS : " + str(int(fps)), (x_position, y_position), cv.FONT_HERSHEY_SIMPLEX, font_size, white,
               white_thickness,
               cv.LINE_AA)

    return image


def draw_hand_detected(image):
    cv.putText(image, "Hand detected", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.73,
               black, 4, cv.LINE_AA)
    cv.putText(image, "Hand detected", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.73,
               white, 2, cv.LINE_AA)

    return image


def draw_upper_bound_desc(image, brect, sign_lang_class):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
    sign_alphabet = sign_lang_class.split(' ')[0]
    cv.putText(image, 'Class : ' + sign_alphabet, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX,
               0.6, black, 2, cv.LINE_AA)
    cv.putText(image, 'Class : ' + sign_alphabet, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX,
               0.6, white, 1, cv.LINE_AA)

    return image


def draw_bounding_box(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     black, 1)

    return image


def draw_lower_bound_desc(image, bbox, sign_lang_prob):
    sign_prob = str(round(sign_lang_prob[np.argmax(sign_lang_prob)], 2) * 100)
    cv.rectangle(image, (bbox[2], bbox[3]), (bbox[0], bbox[3] + 22), (0, 0, 0), -1)
    cv.putText(image, 'Prob : ' + sign_prob + "%", (bbox[0] + 5, bbox[3] + 17),
               cv.FONT_HERSHEY_SIMPLEX,
               0.6, black, 2, cv.LINE_AA)
    cv.putText(image, 'Prob : ' + sign_prob + "%", (bbox[0] + 5, bbox[3] + 17),
               cv.FONT_HERSHEY_SIMPLEX,
               0.6, white, 1, cv.LINE_AA)

    return image


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                black, 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                grey_shade1, 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                black, 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                grey_shade2, 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                black, 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                grey_shade1, 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[0]),
                black, 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[0]),
                grey_shade1, 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                black, 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                grey_shade1, 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                black, 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                grey_shade1, 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                black, 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                grey_shade1, 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                black, 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                grey_shade1, 2)

        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                black, 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                grey_shade3, 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                black, 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                white, 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                black, 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                grey_shade2, 2),
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                black, 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                grey_shade3, 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                black, 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                white, 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                black, 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                grey_shade2, 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                black, 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                grey_shade3, 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                black, 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                white, 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                black, 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                grey_shade2, 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                black, 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                grey_shade3, 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                black, 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                white, 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                black, 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                grey_shade2, 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                black, 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                grey_shade3, 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                black, 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                white, 2)

    for index, landmark in enumerate(landmark_point):
        if index == 0:
            cv.circle(image, (landmark[0], landmark[1]), 5, white,
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, black, 1)
        if index == 1:
            cv.circle(image, (landmark[0], landmark[1]), 5, white,
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, black, 1)
        if index == 2:
            cv.circle(image, (landmark[0], landmark[1]), 5, white,
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, black, 1)
        if index == 3:
            cv.circle(image, (landmark[0], landmark[1]), 5, white,
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, black, 1)
        if index == 4:
            cv.circle(image, (landmark[0], landmark[1]), 8, white,
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, black, 1)
        if index == 5:
            cv.circle(image, (landmark[0], landmark[1]), 5, white,
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, black, 1)
        if index == 6:
            cv.circle(image, (landmark[0], landmark[1]), 5, white,
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, black, 1)
        if index == 7:
            cv.circle(image, (landmark[0], landmark[1]), 5, white,
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, black, 1)
        if index == 8:
            cv.circle(image, (landmark[0], landmark[1]), 8, white,
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, black, 1)
        if index == 9:
            cv.circle(image, (landmark[0], landmark[1]), 5, white,
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, black, 1)
        if index == 10:
            cv.circle(image, (landmark[0], landmark[1]), 5, white,
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, black, 1)
        if index == 11:
            cv.circle(image, (landmark[0], landmark[1]), 5, white,
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, black, 1)
        if index == 12:
            cv.circle(image, (landmark[0], landmark[1]), 8, white,
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, black, 1)
        if index == 13:
            cv.circle(image, (landmark[0], landmark[1]), 5, white,
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, black, 1)
        if index == 14:
            cv.circle(image, (landmark[0], landmark[1]), 5, white,
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, black, 1)
        if index == 15:
            cv.circle(image, (landmark[0], landmark[1]), 5, white,
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, black, 1)
        if index == 16:
            cv.circle(image, (landmark[0], landmark[1]), 8, white,
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, black, 1)
        if index == 17:
            cv.circle(image, (landmark[0], landmark[1]), 5, white,
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, black, 1)
        if index == 18:
            cv.circle(image, (landmark[0], landmark[1]), 5, white,
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, black, 1)
        if index == 19:
            cv.circle(image, (landmark[0], landmark[1]), 5, white,
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, black, 1)
        if index == 20:
            cv.circle(image, (landmark[0], landmark[1]), 8, white,
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, black, 1)

    return image


if __name__ == '__main__':
    main()
