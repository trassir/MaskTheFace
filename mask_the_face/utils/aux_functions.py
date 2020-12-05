# Author: aqeelanwar
# Created: 27 April,2020, 10:21 PM
# Email: aqeel.anwar@gatech.edu
import bz2
import math
import os
import random
import shutil
from collections import namedtuple
from configparser import ConfigParser
from pathlib import Path

import cv2
import requests
from PIL import Image, ImageDraw
from imutils import face_utils

from mask_the_face.utils.create_mask import texture_the_mask, color_the_mask
from mask_the_face.utils.fit_ellipse import *
from mask_the_face.utils.read_cfg import read_cfg

MASK_TEMPLATES_CACHE = {}


def download_dlib_model(destination: Path):
    print_orderly("Get dlib model", 60)
    dlib_model_link = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    print("Downloading dlib model...")
    with requests.get(dlib_model_link, stream=True) as r:
        print("Zip file size: ", np.round(len(r.content) / 1024 / 1024, 2), "MB")
        destination_bz2 = f'{destination}.bz2'
        if not os.path.exists(str(destination_bz2).rsplit(os.path.sep, 1)[0]):
            os.mkdir(destination_bz2.rsplit(os.path.sep, 1)[0])
        print("Saving dlib model...")
        with open(destination_bz2, "wb") as fd:
            for chunk in r.iter_content(chunk_size=32678):
                fd.write(chunk)
    print("Extracting dlib model...")
    with bz2.BZ2File(destination_bz2) as fr, open(str(destination), "wb") as fw:
        shutil.copyfileobj(fr, fw)
    print("Saved: ", destination)
    print_orderly("done", 60)
    os.remove(destination_bz2)


def get_line(face_landmark, image_shape, type="eye"):
    left_eye = face_landmark["left_eye"]
    right_eye = face_landmark["right_eye"]
    left_eye_mid = np.mean(np.array(left_eye), axis=0)
    right_eye_mid = np.mean(np.array(right_eye), axis=0)
    eye_line_mid = (left_eye_mid + right_eye_mid) / 2.

    if type == "eye":
        left_point = left_eye_mid
        right_point = right_eye_mid
        mid_point = eye_line_mid

    elif type == "nose_mid":
        nose_length = (face_landmark["nose_bridge"][-1][1] - face_landmark["nose_bridge"][0][1])
        left_point = [left_eye_mid[0], left_eye_mid[1] + nose_length / 2.]
        right_point = [right_eye_mid[0], right_eye_mid[1] + nose_length / 2.]

        mid_pointY = (face_landmark["nose_bridge"][-1][1] + face_landmark["nose_bridge"][0][1]) / 2.
        mid_pointX = (face_landmark["nose_bridge"][-1][0] + face_landmark["nose_bridge"][0][0]) / 2.
        mid_point = (mid_pointX, mid_pointY)

    elif type == "nose_tip":
        nose_length = (face_landmark["nose_bridge"][-1][1] - face_landmark["nose_bridge"][0][1])
        left_point = [left_eye_mid[0], left_eye_mid[1] + nose_length]
        right_point = [right_eye_mid[0], right_eye_mid[1] + nose_length]
        mid_point = (face_landmark["nose_bridge"][-1][1] + face_landmark["nose_bridge"][0][1]) / 2.

    elif type == "bottom_lip":
        bottom_lip = face_landmark["bottom_lip"]
        bottom_lip_mid = np.max(np.array(bottom_lip), axis=0)
        shiftY = bottom_lip_mid[1] - eye_line_mid[1]
        left_point = [left_eye_mid[0], left_eye_mid[1] + shiftY]
        right_point = [right_eye_mid[0], right_eye_mid[1] + shiftY]
        mid_point = bottom_lip_mid

    elif type == "perp_line":
        bottom_lip = face_landmark["bottom_lip"]
        bottom_lip_mid = np.mean(np.array(bottom_lip), axis=0)

        left_point = face_landmark["nose_bridge"][0]
        right_point = bottom_lip_mid

        mid_point = bottom_lip_mid

    elif type == "nose_long":
        nose_bridge = face_landmark["nose_bridge"]
        left_point = [nose_bridge[0][0], nose_bridge[0][1]]
        right_point = [nose_bridge[-1][0], nose_bridge[-1][1]]

        mid_point = left_point

    else:
        assert False

    y = [left_point[1], right_point[1]]
    x = [left_point[0], right_point[0]]
    eye_line = fit_line(x, y, image_shape)

    # Perpendicular Line
    # (midX, midY) and (midX - y2 + y1, midY + x2 - x1)
    y = [
        (left_point[1] + right_point[1]) / 2.,
        (left_point[1] + right_point[1]) / 2. + right_point[0] - left_point[0],
    ]
    x = [
        (left_point[0] + right_point[0]) / 2.,
        (left_point[0] + right_point[0]) / 2. - right_point[1] + left_point[1],
    ]
    perp_line = fit_line(x, y, image_shape)
    return eye_line, perp_line, left_point, right_point, mid_point


def get_points_on_chin(line, face_landmark, chin_type="chin"):
    chin = face_landmark[chin_type]
    points_on_chin = []
    for i in range(len(chin) - 1):
        chin_first_point = [chin[i][0], chin[i][1]]
        chin_second_point = [chin[i + 1][0], chin[i + 1][1]]

        flag, x, y = line_intersection(line, (chin_first_point, chin_second_point))
        if flag:
            points_on_chin.append((x, y))

    return points_on_chin


def plot_lines(face_line, image, debug=False):
    pil_image = Image.fromarray(image)
    if debug:
        d = ImageDraw.Draw(pil_image)
        d.line(face_line, width=4, fill="white")
        pil_image.show()


def line_intersection(line1, line2):
    start = 0
    end = -1
    line1 = ([line1[start][0], line1[start][1]], [line1[end][0], line1[end][1]])

    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    x = []
    y = []
    flag = False

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return flag, x, y

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    segment_minX = min(line2[0][0], line2[1][0])
    segment_maxX = max(line2[0][0], line2[1][0])

    segment_minY = min(line2[0][1], line2[1][1])
    segment_maxY = max(line2[0][1], line2[1][1])

    if segment_maxX + 1 >= x >= segment_minX - 1 and segment_maxY + 1 >= y >= segment_minY - 1:
        flag = True

    return flag, x, y


def fit_line(x, y, image_shape):
    if x[0] == x[1]:
        x[0] += 0.1
    coefficients = np.polyfit(x, y, 1)
    polynomial = np.poly1d(coefficients)
    x_axis = np.linspace(0, image_shape[1], 50)
    y_axis = polynomial(x_axis)
    eye_line = []
    for i in range(len(x_axis)):
        eye_line.append((x_axis[i], y_axis[i]))

    return eye_line


def get_six_points(face_landmark, image_shape):
    _, perp_line1, _, _, m = get_line(face_landmark, image_shape, type="nose_mid")
    face_b = m

    perp_line, _, _, _, _ = get_line(face_landmark, image_shape, type="perp_line")
    points1 = get_points_on_chin(perp_line1, face_landmark)
    points = get_points_on_chin(perp_line, face_landmark)
    if not points1:
        face_e = tuple(np.asarray(points[0]))
    elif not points:
        face_e = tuple(np.asarray(points1[0]))
    else:
        face_e = tuple((np.asarray(points[0]) + np.asarray(points1[0])) / 2)
    nose_mid_line, _, _, _, _ = get_line(face_landmark, image_shape, type="nose_long")

    angle = get_angle(perp_line, nose_mid_line)
    nose_mid_line, _, _, _, _ = get_line(face_landmark, image_shape, type="nose_tip")
    points = get_points_on_chin(nose_mid_line, face_landmark)
    if len(points) < 2:
        face_landmark = get_face_ellipse(face_landmark)
        points = get_points_on_chin(nose_mid_line, face_landmark, chin_type="chin_extrapolated")
        if len(points) < 2:
            points = [face_landmark["chin"][0], face_landmark["chin"][-1]]
    face_a = points[0]
    face_c = points[-1]

    nose_mid_line, _, _, _, _ = get_line(face_landmark, image_shape, type="bottom_lip")
    points = get_points_on_chin(nose_mid_line, face_landmark)
    face_d = points[0]
    face_f = points[-1]

    six_points = np.float32([face_a, face_b, face_c, face_f, face_e, face_d])
    return six_points, angle


def get_angle(line1, line2):
    delta_y = line1[-1][1] - line1[0][1]
    delta_x = line1[-1][0] - line1[0][0]
    perp_angle = math.degrees(math.atan2(delta_y, delta_x))
    if delta_x < 0:
        perp_angle = perp_angle + 180
    if perp_angle < 0:
        perp_angle += 360
    if perp_angle > 180:
        perp_angle -= 180

    delta_y = line2[-1][1] - line2[0][1]
    delta_x = line2[-1][0] - line2[0][0]
    nose_angle = math.degrees(math.atan2(delta_y, delta_x))

    if delta_x < 0:
        nose_angle = nose_angle + 180
    if nose_angle < 0:
        nose_angle += 360
    if nose_angle > 180:
        nose_angle -= 180

    angle = nose_angle - perp_angle
    return angle


def mask_face(image, six_points, angle, args, type="surgical"):
    global MASK_TEMPLATES_CACHE, MASK_TEXTURES_CACHE

    debug = False

    # Find the face angle
    threshold = 13
    if angle < -threshold:
        type += "_right"
    elif angle > threshold:
        type += "_left"

    # Read appropriate mask image
    h, w = image.shape[:2]
    module_path = Path(__file__).parent.parent
    if not "empty" in type and not "inpaint" in type:
        cfg = read_cfg(config_filename=module_path / "masks/masks.cfg", mask_type=type, verbose=False)
    else:
        if "left" in type:
            mask_str = "surgical_blue_left"
        elif "right" in type:
            mask_str = "surgical_blue_right"
        else:
            mask_str = "surgical_blue"
        cfg = read_cfg(config_filename=module_path / "masks/masks.cfg", mask_type=mask_str, verbose=False)

    img_filename = str(module_path / cfg.template)
    if img_filename not in MASK_TEMPLATES_CACHE:
        img = cv2.imread(str(module_path / cfg.template), cv2.IMREAD_UNCHANGED)
        MASK_TEMPLATES_CACHE[img_filename] = img
    else:
        img = MASK_TEMPLATES_CACHE[img_filename]

    # Process the mask if necessary
    if args.pattern:
        # Apply pattern to mask
        img = texture_the_mask(img, args.pattern, args.pattern_weight)

    if args.color:
        # Apply color to mask
        img = color_the_mask(img, args.color, args.color_weight)

    mask_line = np.float32([cfg.mask_a, cfg.mask_b, cfg.mask_c,
                            cfg.mask_f, cfg.mask_e, cfg.mask_d])
    # Warp the mask
    M, mask = cv2.findHomography(mask_line, six_points)
    dst_mask = cv2.warpPerspective(img, M, (w, h))
    dst_mask_points = cv2.perspectiveTransform(mask_line.reshape(-1, 1, 2), M)
    mask = dst_mask[:, :, 3]
    image_face = image

    # Adjust Brightness
    mask_brightness = get_avg_brightness(img)
    img_brightness = get_avg_brightness(image_face)
    delta_b = 1 + (img_brightness - mask_brightness) / 255
    dst_mask = change_brightness(dst_mask, delta_b)

    # Adjust Saturation
    mask_saturation = get_avg_saturation(img)
    img_saturation = get_avg_saturation(image_face)
    delta_s = 1 - (img_saturation - mask_saturation) / 255
    dst_mask = change_saturation(dst_mask, delta_s)

    # Apply mask
    mask_inv = cv2.bitwise_not(mask)
    img_bg = cv2.bitwise_and(image, image, mask=mask_inv)
    img_fg = cv2.bitwise_and(dst_mask, dst_mask, mask=mask)
    out_img = cv2.add(img_bg, img_fg[:, :, 0:3])
    if "empty" in type or "inpaint" in type:
        out_img = img_bg

    if "inpaint" in type:
        out_img = cv2.inpaint(out_img, mask, 3, cv2.INPAINT_TELEA)

    if debug:
        for i in six_points:
            cv2.circle(out_img, (i[0], i[1]), radius=4, color=(0, 0, 255), thickness=-1)

        for i in dst_mask_points:
            cv2.circle(out_img, (i[0][0], i[0][1]), radius=4, color=(0, 255, 0), thickness=-1)

    return out_img, mask


def draw_landmarks(face_landmarks, image):
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)
    for facial_feature in face_landmarks.keys():
        d.line(face_landmarks[facial_feature], width=5, fill="white")
    pil_image.show()


def get_face_ellipse(face_landmark):
    chin = face_landmark["chin"]
    x = []
    y = []
    for point in chin:
        x.append(point[0])
        y.append(point[1])

    x = np.asarray(x)
    y = np.asarray(y)

    a = fitEllipse(x, y)
    center = ellipse_center(a)
    phi = ellipse_angle_of_rotation(a)
    axes = ellipse_axis_length(a)
    a, b = axes

    arc = 2.2
    R = np.arange(0, arc * np.pi, 0.2)
    xx = center[0] + a * np.cos(R) * np.cos(phi) - b * np.sin(R) * np.sin(phi)
    yy = center[1] + a * np.cos(R) * np.sin(phi) + b * np.sin(R) * np.cos(phi)
    chin_extrapolated = []
    for i in range(len(R)):
        chin_extrapolated.append((xx[i], yy[i]))
    face_landmark["chin_extrapolated"] = chin_extrapolated
    return face_landmark


def get_avg_brightness(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    return np.mean(v)


def get_avg_saturation(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    return np.mean(v)


def change_brightness(img, value=1.0):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    v = value * v
    v[v > 255] = 255
    v = np.asarray(v, dtype=np.uint8)
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def change_saturation(img, value=1.0):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    s = value * s
    s[s > 255] = 255
    s = np.asarray(s, dtype=np.uint8)
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def check_path(path):
    is_directory = False
    is_file = False
    is_other = False
    if os.path.isdir(path):
        is_directory = True
    elif os.path.isfile(path):
        is_file = True
    else:
        is_other = True

    return is_directory, is_file, is_other


def shape_to_landmarks(shape):
    face_landmarks = {"left_eyebrow": [
        tuple(shape[17]),
        tuple(shape[18]),
        tuple(shape[19]),
        tuple(shape[20]),
        tuple(shape[21]),
    ], "right_eyebrow": [
        tuple(shape[22]),
        tuple(shape[23]),
        tuple(shape[24]),
        tuple(shape[25]),
        tuple(shape[26]),
    ], "nose_bridge": [
        tuple(shape[27]),
        tuple(shape[28]),
        tuple(shape[29]),
        tuple(shape[30]),
    ], "nose_tip": [
        tuple(shape[31]),
        tuple(shape[32]),
        tuple(shape[33]),
        tuple(shape[34]),
        tuple(shape[35]),
    ], "left_eye": [
        tuple(shape[36]),
        tuple(shape[37]),
        tuple(shape[38]),
        tuple(shape[39]),
        tuple(shape[40]),
        tuple(shape[41]),
    ], "right_eye": [
        tuple(shape[42]),
        tuple(shape[43]),
        tuple(shape[44]),
        tuple(shape[45]),
        tuple(shape[46]),
        tuple(shape[47]),
    ], "top_lip": [
        tuple(shape[48]),
        tuple(shape[49]),
        tuple(shape[50]),
        tuple(shape[51]),
        tuple(shape[52]),
        tuple(shape[53]),
        tuple(shape[54]),
        tuple(shape[60]),
        tuple(shape[61]),
        tuple(shape[62]),
        tuple(shape[63]),
        tuple(shape[64]),
    ], "bottom_lip": [
        tuple(shape[54]),
        tuple(shape[55]),
        tuple(shape[56]),
        tuple(shape[57]),
        tuple(shape[58]),
        tuple(shape[59]),
        tuple(shape[48]),
        tuple(shape[64]),
        tuple(shape[65]),
        tuple(shape[66]),
        tuple(shape[67]),
        tuple(shape[60]),
    ], "chin": [
        tuple(shape[0]),
        tuple(shape[1]),
        tuple(shape[2]),
        tuple(shape[3]),
        tuple(shape[4]),
        tuple(shape[5]),
        tuple(shape[6]),
        tuple(shape[7]),
        tuple(shape[8]),
        tuple(shape[9]),
        tuple(shape[10]),
        tuple(shape[11]),
        tuple(shape[12]),
        tuple(shape[13]),
        tuple(shape[14]),
        tuple(shape[15]),
        tuple(shape[16]),
    ]}

    return face_landmarks


def is_image(path):
    try:
        extensions = path[-4:]
        image_extensions = ["png", "PNG", "jpg", "JPG"]

        if extensions[1:] in image_extensions:
            return True
        else:
            print("Please input image file. png / jpg")
            return False
    except:
        return False


def get_available_mask_types(config_filename):
    parser = ConfigParser()
    parser.optionxform = str
    parser.read(str(config_filename))
    available_mask_types = parser.sections()
    available_mask_types = [
        string for string in available_mask_types if "left" not in string
    ]
    available_mask_types = [
        string for string in available_mask_types if "right" not in string
    ]

    return available_mask_types


def print_orderly(str, n):
    hyphens = "-" * int((n - len(str)) / 2)
    str_p = hyphens + " " + str + " " + hyphens
    hyphens_bar = "-" * len(str_p)
    print(hyphens_bar)
    print(str_p)
    print(hyphens_bar)


def display_MaskTheFace():
    module_path = Path(__file__).parent.parent
    with open(module_path / "utils/display.txt", "r") as file:
        for line in file:
            cc = 1
            print(line, end="")


def mask_by_img_and_bboxes(image, bboxes, pattern_weight, color_weight):
    import dlib

    module_path = Path(__file__).parent.parent
    _colors = ["#fc1c1a", "#177ABC", "#94B6D2", "#A5AB81", "#DD8047", "#6b425e", "#e26d5a", "#c92c48",
               "#6a506d", "#ffc900", "#ffffff", "#000000", "#49ff00"]
    text_path = module_path / 'masks/textures'
    _patterns = [
        text_path / 'check/check_1.png', text_path / 'check/check_2.jpg', text_path / 'check/check_3.png',
        text_path / 'check/check_4.jpg', text_path / 'check/check_5.jpg', text_path / 'check/check_6.jpg',
        text_path / 'check/check_7.jpg', text_path / 'floral/floral_1.png', text_path / 'floral/floral_2.jpg',
        text_path / 'floral/floral_3.jpg', text_path / 'floral/floral_4.jpg', text_path / 'floral/floral_5.jpg',
        text_path / 'floral/floral_6.jpg', text_path / 'floral/floral_7.png', text_path / 'floral/floral_8.png',
        text_path / 'floral/floral_9.jpg', text_path / 'floral/floral_10.png', text_path / 'floral/floral_11.jpg',
        text_path / 'floral/grey_petals.png', text_path / 'fruits/bananas.png', text_path / 'fruits/cherry.png',
        text_path / 'fruits/lemon.png', text_path / 'fruits/pineapple.png', text_path / 'fruits/strawberry.png',
        text_path / 'others/heart_1.png', text_path / 'others/polka.jpg'
    ]

    Args = namedtuple('Args', ['pattern', 'pattern_weight', 'color', 'color_weight'])
    path_to_dlib_model = module_path / "dlib_models/shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(path_to_dlib_model):
        download_dlib_model(module_path / path_to_dlib_model)

    kps_predictor = dlib.shape_predictor(path_to_dlib_model)

    original_image = image
    image = image.copy()

    # Process each face in the image
    for (i, bbox) in enumerate(bboxes):
        mask_type = random.choice(get_available_mask_types(module_path / 'masks/masks.cfg'))
        color = random.choice(_colors)
        pattern = str(random.choice(_patterns))

        bbox = [int(c) for c in bbox]
        shape = kps_predictor(original_image, dlib.rectangle(*bbox))
        shape = face_utils.shape_to_np(shape)
        face_landmarks = shape_to_landmarks(shape)
        # draw_landmarks(face_landmarks, image)
        six_points_on_face, angle = get_six_points(face_landmarks, original_image)
        image, _ = mask_face(
            image, six_points_on_face, angle, Args(pattern, pattern_weight, color, color_weight),
            type=mask_type
        )

    return image
