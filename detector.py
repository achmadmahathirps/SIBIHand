# Import libraries #####################################################################################################
from itertools import chain
from pickle import load
from time import time
from copy import deepcopy
from google.protobuf.json_format import MessageToDict
from pandas import DataFrame
from mediapipe import solutions
from numpy import \
    empty, array, append, argmax
from cv2 import \
    VideoCapture, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, waitKey, flip, cvtColor, COLOR_BGR2RGB, imshow, \
    boundingRect, putText, FONT_HERSHEY_SIMPLEX, LINE_AA, rectangle, line, circle, CAP_DSHOW

import keyboard


# Main program #########################################################################################################
def main():
    # Initializations #########################################

    # Initialize camera settings
    webcam = 1  # <- (0 = built-in webcam, 1 = external webcam)

    from_capture = VideoCapture(webcam, CAP_DSHOW)
    from_capture.set(CAP_PROP_FRAME_WIDTH, 640)
    from_capture.set(CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize Mediapipe hand model parameters
    mp_hands = solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )
    drawing = solutions.drawing_utils
    drawing_styles = solutions.drawing_styles

    # Initialize misc.
    read_pkl = 'rb'
    previous_time = 0
    on = False
    key_release = True

    # #########################################################

    # Open & import trained model
    with open('model/svm_trained_classifier_test.pkl', read_pkl) as model_file:
        model = load(model_file)

    # While in capturing process
    while True:

        # Application stops when "ESC" key is pressed
        if waitKey(3) & keyboard.is_pressed('ESC'):
            print("Exited through ESC key.")
            break

        # If frame/image in capture is not available left, then stop the application
        available, image = from_capture.read()
        if not available:
            print("Video/image frame not available left.")
            break

        # Toggle Mediapipe hand landmark visuals
        if keyboard.is_pressed('v'):  # <- Toggle by pressing "v" key
            while keyboard.is_pressed('v'):
                pass
            if on:
                key_release = False  # <- Visualizer turned off by default.
            elif not on:
                key_release = True
            on = not on

        # Flip (if built-in webcam is detected) and copy the image for debugging
        image = flip(image, 1)
        debug_image = deepcopy(image)

        # Convert frame image from BGR to RGB for pre-optimization
        image = cvtColor(image, COLOR_BGR2RGB)

        # Optimize before detection process
        image.flags.writeable = False
        detection_results = hands.process(image)  # <- (Main hand detection)
        image.flags.writeable = True

        # Calculate and visualize FPS indicator
        current_time = time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        debug_image = draw_fps(debug_image, fps)

        # Visualize student info
        debug_image = draw_student_info(debug_image)

        # If the hand is detected:
        if detection_results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(detection_results.multi_hand_landmarks,
                                                  detection_results.multi_handedness):
                # 0. Return whether it is Right or Left Hand
                detected_hand = MessageToDict(handedness)['classification'][0]['label']

                # 1. Extract & convert pre-normalized landmark keys from hand into absolute pixel value
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # 2a. If right hand is detected
                if detected_hand == 'Right':
                    # 3a. Convert into relative coordinates / normalize keys from wrist point
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # 2b. else (if) left hand is detected
                else:
                    # 3b. Convert & invert x coordinates into relative coordinates / normalize keys from wrist point
                    pre_processed_landmark_list = pre_process_landmark_x_inverted(landmark_list)

                # If hand detection is confirmed, try :
                try:
                    # 4. Compare dataset with processed landmarks from detected hand
                    data_frame = DataFrame([pre_processed_landmark_list])
                    sign_language_class = model.predict(data_frame)[0]
                    sign_language_prob = model.predict_proba(data_frame)[0]

                    # Draw "Hand detected" description
                    debug_image = draw_hand_detected(debug_image, sign_language_class)

                    # Calculate boundaries for bounding box
                    bounding_box = calc_bounding_box(debug_image, hand_landmarks)

                    # Draw bounding box with descriptions
                    debug_image = draw_upper_bound_desc(debug_image, bounding_box, sign_language_class)
                    debug_image = draw_bounding_box(True, debug_image, bounding_box)
                    debug_image, prob_percentage = draw_lower_bound_desc(debug_image, bounding_box, sign_language_prob)

                    # Show output in terminal
                    print('Sign : ' + sign_language_class)
                    print(sign_language_prob)
                    print(prob_percentage)

                # Finally if hand is not detected, just bypass to the below code
                finally:
                    pass

                # Draw complete hand landmarks
                if not key_release:
                    debug_image = draw_outlines(debug_image, landmark_list)
                    drawing.draw_landmarks(  # (Mediapipe default visualizer)
                        debug_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        drawing_styles.get_default_hand_landmarks_style(),
                        drawing_styles.get_default_hand_connections_style())

        # Output frame
        imshow('Hand (Fingerspelling) Sign Language Recognition', debug_image)

# Main Program #########################################################################################################


# Calculation functions ################################################################################################

# Calculate bounding box size
def calc_bounding_box(image, hand_landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = empty((0, 2), int)

    for _, landmark in enumerate(hand_landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [array((landmark_x, landmark_y))]

        landmark_array = append(landmark_array, landmark_point, axis=0)

    x_axis, y_axis, width, height = boundingRect(landmark_array)

    return [x_axis, y_axis, x_axis + width, y_axis + height]


# Extract & convert default-normalized landmark keys into absolute pixel value
def calc_landmark_list(image, hand_landmarks):

    # Initialize image size & new landmark list
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_list = []

    # Extract & for each landmark keys from detected hand
    for _, landmark in enumerate(hand_landmarks.landmark):

        # Convert pre-normalized landmark keys into absolute pixel value
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        # ! landmark_z is unused due to a bug from Mediapipe Hands when detecting the depth of the hand
        # landmark_z = landmark.z

        # Put the converted landmark keys inside the new landmark list
        landmark_list.append([landmark_x, landmark_y])

    return landmark_list


# Convert into wrist-relative point coordinates & normalize keys
def pre_process_landmark(landmark_list):

    # Receive landmark list from calc_landmark_list function
    temp_landmark_list = deepcopy(landmark_list)

    # Initialize reference key
    base_x, base_y = 0, 0

    # For each detected landmark keys in landmark list
    for index, landmark_point in enumerate(temp_landmark_list):
        # If the first index of the landmark list (wrist) is detected,
        # set the corresponding landmark keys to reference key
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        # for other landmarks, subtract with set reference key value
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = base_y - temp_landmark_list[index][1]

    # Convert to a one-dimensional matrix list
    temp_landmark_list = list(
        chain.from_iterable(temp_landmark_list))

    # Find the max value inside the one-dimensional landmark list
    max_value = max(list(map(abs, temp_landmark_list)))

    # Normalize the relative keys based from the max value
    def normalize_(n):
        return n / max_value

    # Place & replace landmark list key with new normalized value
    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    # Output : return with the new temp_landmark_list value
    return temp_landmark_list


# Convert into wrist-relative point coordinates & normalize keys for left hand 
def pre_process_landmark_x_inverted(landmark_list):

    # Receive landmark list from calc_landmark_list function
    temp_landmark_list = deepcopy(landmark_list)

    # Initialize reference key
    base_x, base_y = 0, 0

    # For each detected landmark keys in landmark list
    for index, landmark_point in enumerate(temp_landmark_list):

        # If the first index of the landmark list (wrist) is detected,
        # set the corresponding landmark keys as 0 for reference key
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        # for other landmarks in left hand, subtract with reference key value and multiply it with
        # negative value. This process makes the left hand can be detected as right hand so
        # it can detect sign language without adding a new dataset.
        temp_landmark_list[index][0] = (temp_landmark_list[index][0] - base_x) * -1
        temp_landmark_list[index][1] = (base_y - temp_landmark_list[index][1])

    # Convert to a one-dimensional matrix list (remove internal square brackets in arrays)
    temp_landmark_list = list(
        chain.from_iterable(temp_landmark_list))

    # Find the max value inside the one-dimensional landmark list
    max_value = max(list(map(abs, temp_landmark_list)))

    # Normalize the relative keys based from the max value
    def normalize_(n):
        return n / max_value

    # Place & replace landmark list key with new normalized value
    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list
# Calculation functions ################################################################################################


# Decorative functions #################################################################################################
def draw_student_info(image):
    # Text & text position
    text = "* Achmad Mahathir P. (187006041) | Universitas Siliwangi 2022"
    x_position, y_position = 10, 470

    # Font settings
    font_size = 0.3
    black, white = (0, 0, 0), (255, 255, 255)
    outline_thickness = 2
    white_thickness = 1

    putText(image, text, (x_position, y_position),
            FONT_HERSHEY_SIMPLEX, font_size,
            black, outline_thickness,
            LINE_AA)
    putText(image, text, (x_position, y_position),
            FONT_HERSHEY_SIMPLEX, font_size,
            white, white_thickness,
            LINE_AA)

    return image


def draw_fps(image, fps):
    # Text & text position
    text = " ".join(["FPS :", str(int(fps))])
    x_position, y_position = 10, 30

    # Font settings
    font_size = 0.75
    black, white = (0, 0, 0), (255, 255, 255)
    outline_thickness = 4
    white_thickness = 1

    putText(image, text, (x_position, y_position), FONT_HERSHEY_SIMPLEX, font_size, black,
            outline_thickness,
            LINE_AA)
    putText(image, text, (x_position, y_position), FONT_HERSHEY_SIMPLEX, font_size, white,
            white_thickness,
            LINE_AA)

    return image


def draw_hand_detected(image, sign_language_class):
    # Text & text position
    text = " ".join(["Hand detected :", sign_language_class])
    # text = "Hand detected : " + sign_language_class
    x_position, y_position = 10, 90

    # Font settings
    font_size = 0.75
    black, green = (0, 0, 0), (0, 255, 0)
    outline_thickness = 4
    green_thickness = 1

    putText(image, text, (x_position, y_position), FONT_HERSHEY_SIMPLEX, font_size,
            black, outline_thickness, LINE_AA)
    putText(image, text, (x_position, y_position), FONT_HERSHEY_SIMPLEX, font_size,
            green, green_thickness, LINE_AA)

    return image


def draw_upper_bound_desc(image, bbox, sign_lang_class):
    sign_alphabet = sign_lang_class.split(' ')[0]

    # Text & tracking position
    text = " ".join(["Sign :", sign_alphabet])
    top, left, bottom = 0, 1, 2
    offset = 25

    # Font settings
    font_size = 0.7
    black, white = (0, 0, 0), (255, 255, 255)
    outline_thickness = 5
    white_thickness = 2

    rectangle(image, (bbox[top], bbox[left]), (bbox[bottom], bbox[left] - offset), black, -1)
    putText(image, text, (bbox[top] + 5, bbox[left] - 4),
            FONT_HERSHEY_SIMPLEX,
            font_size, black, outline_thickness, LINE_AA)
    putText(image, text, (bbox[top] + 5, bbox[left] - 4),
            FONT_HERSHEY_SIMPLEX,
            font_size, white, white_thickness, LINE_AA)

    return image


def draw_bounding_box(use_bbox, image, bbox):
    top, left, bottom, right = 0, 1, 2, 3
    black = (0, 0, 0)

    if use_bbox:
        # Outer rectangle
        rectangle(image, (bbox[top], bbox[left]), (bbox[bottom], bbox[right]),
                  black, 1)

    return image


def draw_lower_bound_desc(image, bbox, sign_lang_prob):

    # Find max value in probability list & convert to percentage
    sign_prob = str(round(sign_lang_prob[argmax(sign_lang_prob)], 2) * 100)

    # Text & tracking position
    text = " ".join(["Prob :", sign_prob, "%"])
    top, bottom, right = 0, 2, 3
    offset = 22

    # Font settings
    font_size = 0.6
    black, white = (0, 0, 0), (255, 255, 255)
    outline_thickness = 2
    white_thickness = 1

    rectangle(image, (bbox[bottom], bbox[right]), (bbox[top], bbox[right] + offset), black, -1)
    putText(image, text, (bbox[0] + 5, bbox[3] + 17),
            FONT_HERSHEY_SIMPLEX,
            font_size, black, outline_thickness, LINE_AA)
    putText(image, text, (bbox[0] + 5, bbox[3] + 17),
            FONT_HERSHEY_SIMPLEX,
            font_size, white, white_thickness, LINE_AA)

    return image, text


def draw_outlines(image, landmark_point):
    black = (0, 0, 0)
    grey_shade3 = (227, 232, 234)

    if len(landmark_point) > 0:
        # Palm
        line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
             black, 6)
        line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
             black, 6)
        line(image, tuple(landmark_point[1]), tuple(landmark_point[5]),
             black, 6)
        line(image, tuple(landmark_point[1]), tuple(landmark_point[5]),
             grey_shade3, 3)
        line(image, tuple(landmark_point[5]), tuple(landmark_point[0]),
             black, 6)
        line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
             black, 6)
        line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
             black, 6)
        line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
             black, 6)
        line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
             black, 6)

        # Thumb
        line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
             black, 6)
        line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
             black, 6)

        # Index finger
        line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
             black, 6)
        line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
             black, 6)
        line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
             black, 6)

        # Middle finger
        line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
             black, 6)
        line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
             black, 6)
        line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
             black, 6)

        # Ring finger
        line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
             black, 6)
        line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
             black, 6)
        line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
             black, 6)

        # Little finger
        line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
             black, 6)
        line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
             black, 6)
        line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
             black, 6)

    # Keypoint shadows
    for index, landmark in enumerate(landmark_point):
        if index == index:
            circle(image, (landmark[0], landmark[1]), 7, black, 1)

    return image
# Decorative functions #################################################################################################


# Run main program
if __name__ == '__main__':
    main()
