import os
import json
import shutil
import numpy as np
import cv2
import imutils
import pytesseract as pytesseract
# https://stackoverflow.com/questions/50951955/pytesseract-tesseractnotfound-error-tesseract-is-not-installed-or-its-not-i/53672281
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


DEFAULT_WIDTH = 500
DEFAULT_HEIGHT = 500


def read_image(path_to_image):
    return cv2.imread(path_to_image)


def resize_image(img, width, height):
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)


def save_image(path, name_with_extension, img):
    cv2.imwrite(os.path.join(path, name_with_extension), img)


def initialize_folders(folder_path_list):
    for folder_path in folder_path_list:
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

# --------------------------- ELŐFELDOLGOZÁS---------------------------


# https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html
def rgb_to_gray_scale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html
def bilateral_filter(img):
    return cv2.bilateralFilter(img, 11, 17, 17)


# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html
def find_edges(img, min_val=170, max_val=300):
    return cv2.Canny(img, min_val, max_val)


# main:
# 1. átméretezés,
# 2. szürkeárnyalatos,
# 3. zaj szűrése,(bilateral filter)
# 4. élek detektálása
def pre_process_raw_image(img, width, height):
    resized_img = resize_image(img, width, height)
    save_image(RESIZED_PATH, 'resized.jpg', resized_img)

    gray_scale_img = rgb_to_gray_scale(resized_img)
    save_image(GRAY_SCALE_PATH, 'gray_scaled.jpg', gray_scale_img)

    bilateral_img = bilateral_filter(gray_scale_img)
    save_image(BILATERAL_PATH, 'bilateral.jpg', bilateral_img)

    edged_img = find_edges(bilateral_img)
    save_image(EDGED_PATH, 'edged.jpg', edged_img)

    return resized_img, bilateral_img, gray_scale_img, edged_img,


# ------------------------------------------------------------------------------


# ------------------------------ KONTÚR MEGTALÁLÁSA -------------------------------

# https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?#findcontours
def find_contours(preprocessed_edged_image):
    contours = cv2.findContours(preprocessed_edged_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # Sorting contours based on there AREA KEEPING MINIMUM (In this case: 10)
    # Anything smaller than this will not be considered)

    return sorted(contours, key=cv2.contourArea, reverse=True)[:10]

# ------------------------------------------------------------------------------


# ------------------------------- NORMALIZÁLÁS --------------------------------
# top_left, top_right, bottom_right, bottom_left
def order_rectangle_points(points):
    rctngl = np.zeros((4, 2), dtype="float32")


    sum_points = points.sum(axis=1)

    rctngl[0] = points[np.argmin(sum_points)]

    rctngl[2] = points[np.argmax(sum_points)]


    diff_points = np.diff(points, axis=1)

    rctngl[1] = points[np.argmin(diff_points)]

    rctngl[3] = points[np.argmax(diff_points)]

    return rctngl


def transform_based_on_four_points(img, aprx_curve):
    ordered_rectangle = order_rectangle_points(aprx_curve)
    (top_left, top_right, bottom_right, bottom_left) = ordered_rectangle

    bottom_width = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
    top_width = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))

    max_width = max(int(bottom_width), int(top_width))

    rigth_height = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
    left_height = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))

    max_height = max(int(rigth_height), int(left_height))

    top_down_view_points = np.array([[0,0], [max_width-1, 0], [max_width-1, max_height-1], [0, max_height-1]], dtype = "float32")

    transformation_matrix = cv2.getPerspectiveTransform(ordered_rectangle, top_down_view_points)
    transformed_image = cv2.warpPerspective(img, transformation_matrix, (max_width, max_height))

    return transformed_image


def apply_threshold(img):
    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
    _, normalized_img = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    return normalized_img


def normalization(img, aprx_curve):
    normalized_img = transform_based_on_four_points(img, aprx_curve.reshape(4, 2))
    save_image(TRANSFORMED_PATH,'transformed.jpg', normalized_img)

    normalized_img = apply_threshold(normalized_img)
    save_image(NORMALIZED_PATH, 'normalized.jpg', normalized_img)

    return normalized_img

# ------------------------------------------------------------------------------


if __name__ == '__main__':

    with open('results.json', 'w') as f:
        json.dump({'results': []}, f, default=str, sort_keys=True, indent=4)

    RAW_IMAGES_PATH = './raw_images'
    BASE_OUT_PATH = './out'

    if os.path.exists(BASE_OUT_PATH):
        shutil.rmtree(BASE_OUT_PATH)

    for image_name in os.listdir(RAW_IMAGES_PATH):
        OUT_IMAGE_PATH = os.path.join(BASE_OUT_PATH, image_name.split('.')[0])
        RESIZED_PATH = os.path.join(OUT_IMAGE_PATH, 'resized')
        GRAY_SCALE_PATH = os.path.join(OUT_IMAGE_PATH, 'gray_scaled')
        BILATERAL_PATH = os.path.join(OUT_IMAGE_PATH, 'bilateral')
        EDGED_PATH = os.path.join(OUT_IMAGE_PATH, 'edged')
        TRANSFORMED_PATH = os.path.join(OUT_IMAGE_PATH, 'transformed')
        NORMALIZED_PATH = os.path.join(OUT_IMAGE_PATH, 'threshold_normalized')
        CONTOURED_PATH = os.path.join(OUT_IMAGE_PATH, 'with_contours')

        initialize_folders([BASE_OUT_PATH, OUT_IMAGE_PATH, GRAY_SCALE_PATH, BILATERAL_PATH, EDGED_PATH, TRANSFORMED_PATH, NORMALIZED_PATH,
                            CONTOURED_PATH])

        image = read_image(os.path.join(RAW_IMAGES_PATH, image_name))

        preprocessed_resized_image, \
        preprocessed_image_bilateral, preprocessed_gray_scaled_image, \
        preprocessed_image_edged = pre_process_raw_image(image, DEFAULT_WIDTH, DEFAULT_HEIGHT)

        found_contours = find_contours(preprocessed_image_edged)

        for cntr in found_contours:
            perimeter = cv2.arcLength(cntr, closed=True)

            # https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?#findcontours
            approximated_curve = cv2.approxPolyDP(cntr, 0.01 * perimeter, True)

            if len(approximated_curve) == 4:

                # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_begin/py_contours_begin.html
                cv2.drawContours(preprocessed_resized_image, [approximated_curve], -1, (0, 255, 0), 3)

                save_image(CONTOURED_PATH, 'contoured.jpg', preprocessed_resized_image)

                normalized_image = normalization(preprocessed_image_bilateral, approximated_curve)

                # https://tesseract-ocr.github.io/tessdoc/ImproveQuality
                # https://github.com/tesseract-ocr/tesseract/blob/master/doc/tesseract.1.asc
                text = pytesseract.image_to_string(normalized_image, config='-l eng --oem 3 --psm 13')

                print('image name: {} text: {}'.format(image_name, text))

                with open('results.json', 'r') as f:
                    data = json.load(f)
                    data['results'].append({'image_name': str(image_name), 'recognized plate number': text})
                    with open('results.json', 'w') as f:
                        json.dump(data, f, default=str, sort_keys=True,
                                  indent=4)

                break
