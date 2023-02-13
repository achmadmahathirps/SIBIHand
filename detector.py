# Import libraries =====================================================================================================
from itertools import chain
from pickle import load
from time import time
from copy import deepcopy
from pandas import DataFrame
from mediapipe import solutions
from google.protobuf.json_format import MessageToDict
from numpy import \
    empty, array, append, argmax
from cv2 import \
    VideoCapture, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, waitKey, flip, cvtColor, COLOR_BGR2RGB, imshow, \
    boundingRect, putText, FONT_HERSHEY_SIMPLEX, LINE_AA, rectangle, line, circle, CAP_DSHOW

import keyboard
# Also import scikit-learn to do predictions
# and pyinstaller to make and executable file (if necessary).

# Main program : BEGIN =================================================================================================
def main():

    # Initialize camera inputs
    print('- Below are camera sources that you can use -')
    print('=============================================')
    print(' ')
    print(' [0] Main Camera')
    print(' [1] Alternative Camera')
    print(' ')
    print('=============================================')

    webcam = input('[Please select your camera source] : ')
    webcam = int(webcam)

    print(' ')
    print('Running . . .')
    print(' ')

    # Initialize camera settings
    from_capture = VideoCapture(webcam, CAP_DSHOW)
    from_capture.set(CAP_PROP_FRAME_WIDTH, 640)
    from_capture.set(CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize Mediapipe hand model parameters
    mediapipe_hands = solutions.hands
    hands = mediapipe_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    )

    # Initialize Mediapipe hand visualizer
    drawing = solutions.drawing_utils
    drawing_styles = solutions.drawing_styles

    # Initialize misc. variables
    read_pkl = 'rb'
    previous_time = 0
    on = False
    key_release = True

    # Open & load trained .pkl model
    with open('model/svm_trained_classifier_test.pkl', read_pkl) as model_file:
        model = load(model_file)
    
    # While in capturing process
    while True:

        # Program stops if camera source is greater than 1 or lesser than 0
        if webcam > 1 or webcam < 0:
            print('(!) Camera source out of bounds. Please select another camera.')
            break

        # Program stops when "ESC" key is pressed
        if waitKey(3) & keyboard.is_pressed('ESC'):
            print(' ')
            print("(!) Exited through ESC key.")
            break

        # If frame/image in capture is not available left, then stop the program
        available, image = from_capture.read()
        if not available:
            print("(!) Video/image frame not available left.")
            break

        # Show Mediapipe hand landmark visuals through keyboard switch
        if keyboard.is_pressed('v'):  # <- Toggle by pressing "v" key
            while keyboard.is_pressed('v'):
                pass
            if on:
                key_release = False  # <- Visualizer turned off by default.
            elif not on:
                key_release = True
            on = not on

        # Flip the image to mirror mode and copy the image for further debugging
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

                # 1. Retrieve information whether it is Right or Left Hand
                detected_hand = MessageToDict(handedness)['classification'][0]['label']

                # 2. Extract & convert pre-normalized landmark keys from hand into absolute pixel value
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)


                # 3. Convert the processed landmarks into relative coordinates from wrist point
                #  - At the same time, invert the x values if a left hand is detected.
                #  - This makes sure that the left hand can be detected as right hand, so we don't have to add another
                #    another datasets specifically for the left hand.
                final_processed_landmark_list = pre_process_landmark(landmark_list, detected_hand)

                # If hand detection is confirmed, try :
                try:
                    # Compare trained dataset from .pkl model with processed landmarks from detected hand
                    data_frame = DataFrame([final_processed_landmark_list])
                    sign_language_class = model.predict(data_frame)[0]
                    sign_language_prob = model.predict_proba(data_frame)[0]

                    # Calculate boundaries for bounding box
                    bounding_box = calc_bounding_box(debug_image, hand_landmarks)

                    # Draw bounding box with descriptions
                    debug_image = draw_upper_bound_desc(debug_image, bounding_box, sign_language_class,
                                                        sign_language_prob)
                    debug_image = draw_bounding_box(True, debug_image, bounding_box)
                    debug_image, prob_percentage = draw_lower_bound_desc(debug_image, bounding_box,
                                                                         sign_language_prob)

                    # Show output in terminal
                    # print(' ')
                    # print('Handedness : ' + detected_hand)
                    # print('Sign : ' + sign_language_class)
                    # print(prob_percentage)

                # Finally if hand is not detected, just bypass to the below code
                finally:
                    pass

                # Draw complete hand landmarks
                if not key_release:
                    debug_image = draw_outlines(debug_image, landmark_list)
                    drawing.draw_landmarks(  # (Mediapipe default visualizer)
                        debug_image,
                        hand_landmarks,
                        mediapipe_hands.HAND_CONNECTIONS,
                        drawing_styles.get_default_hand_landmarks_style(),
                        drawing_styles.get_default_hand_connections_style())

        # Output frame
        imshow('Hand (Fingerspelling) Sign Language Recognition', debug_image)

# Main Program : END ===================================================================================================


# Feature extraction functions =========================================================================================

# Extract & convert default-normalized landmark keys into absolute pixel value
def calc_landmark_list(image, hand_landmarks):

    # Initialize image size & new landmark list
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_list = []

    # Extract & for each landmark keys from detected hand
    for _, landmark in enumerate(hand_landmarks.landmark):

        # Convert the default normalized landmark keys into original pixel value
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        # ! landmark_z is unused due to a bug from Mediapipe Hands when detecting the depth of the hand
        # ! and detecting 2 dimensional datas are much easier to read and to explain.
        # landmark_z = landmark.z

        # Put the converted landmark keys inside the new landmark list
        landmark_list.append([landmark_x, landmark_y])

    return landmark_list


# Convert into wrist-relative point coordinates & normalize keys
def pre_process_landmark(landmark_list, handedness):

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
        if handedness == 'Right':
            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        else:
            temp_landmark_list[index][0] = (temp_landmark_list[index][0] - base_x) * -1

        temp_landmark_list[index][1] = base_y - temp_landmark_list[index][1]

    # Convert to a one-dimensional matrix list
    temp_landmark_list = list(
        chain.from_iterable(temp_landmark_list))

    # Find the max value inside the one-dimensional landmark list
    max_value = max(list(map(abs, temp_landmark_list)))

    # Normalize the relative keys based from the max value
    def normalize_value(n):
        return n / max_value

    # Place & replace landmark list key with new normalized value
    temp_landmark_list = list(map(normalize_value, temp_landmark_list))

    # Output : return with the new temp_landmark_list value
    return temp_landmark_list


# Description UI functions =====================================================================================

def draw_student_info(image):
    # Text & text position
    text = "* Achmad Mahathir P. (187006041) | Universitas Siliwangi 2023"
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


# Bounding box functions ===============================================================================================

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
    offset = 25

    return [x_axis - offset, y_axis - offset, x_axis + width + offset, y_axis + height + offset]


def draw_upper_bound_desc(image, bbox, sign_lang_class, sign_lang_prob):
    sign_alphabet = sign_lang_class.split(' ')[0]

    # Text & tracking position
    if (round(sign_lang_prob[argmax(sign_lang_prob)], 2) * 100) > 59:
        text = " ".join(["Sign :", sign_alphabet])
    else:
        text = " "
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
    putText(image, text, (bbox[top] + 5, bbox[right] + 17),
            FONT_HERSHEY_SIMPLEX,
            font_size, black, outline_thickness, LINE_AA)
    putText(image, text, (bbox[top] + 5, bbox[right] + 17),
            FONT_HERSHEY_SIMPLEX,
            font_size, white, white_thickness, LINE_AA)

    return image, text

# Decorative functions =================================================================================================

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


# Executor =============================================================================================================

# Run the program from main function
if __name__ == '__main__':
    main()
