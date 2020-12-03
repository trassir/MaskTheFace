import random
from collections import namedtuple
from logging import warning
from pathlib import Path
from typing import List

import numpy as np
from imutils import face_utils

try:
    import dlib
except:
    dlib = None

from albumentations import DualTransform

from mask_the_face.utils.aux_functions import (download_dlib_model, get_available_mask_types,
                                               mask_face, shape_to_landmarks, get_six_points)

__all__ = ['FaceMaskTransformation']


class FaceMaskTransformation(DualTransform):
    def __init__(self, pattern_weight: float = 0.5, color_weight: float = 0.5,
                 bboxes_key: str = 'bboxes', keypoints_key: str = '300w_keypoints',
                 suppress_warnings=False, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self._pattern_weight = pattern_weight
        self._color_weight = color_weight
        self._bboxes_key = bboxes_key
        self._keypoints_key = keypoints_key
        self._suppress_warnings = suppress_warnings
        self._module_path = Path(__file__).parent.parent
        self._dlib_kps_model = None

    def apply(self, image, bboxes, kps_by_bbox, mask_face_params_by_bbox, **params):
        for i, (bbox, keypoints, params) in enumerate(zip(bboxes, kps_by_bbox, mask_face_params_by_bbox)):
            try:
                six_points_on_face, angle = get_six_points(keypoints, image.shape)
                image, _ = mask_face(image, six_points_on_face, angle, params, type=params.mask_type)
            except:
                if not self._suppress_warnings:
                    warning('Some of face cannot be processed in the FaceMaskTransformation')
        return image

    def get_params_dependent_on_targets(self, params):
        h, w = params['image'].shape[:2]
        assert len(params[self._bboxes_key]) > 0
        with_keypoints = self._keypoints_key in params
        if not with_keypoints and self._dlib_kps_model is None:
            self._dlib_kps_model = self._create_kp_dlib_model()

        bboxes, kps_by_bbox = [], []
        if not with_keypoints:
            for (x1, y1, x2, y2) in self._make_list_of_list(params[self._bboxes_key]):
                x1, y1, x2, y2 = map(int, (x1 * w, y1 * h, x2 * w, y2 * h))
                shape = self._dlib_kps_model(params["image"], dlib.rectangle(x1, y1, x2, y2))
                keypoints = shape_to_landmarks(face_utils.shape_to_np(shape))
                bboxes.append([x1, y1, x2, y2])
                kps_by_bbox.append(keypoints)
        else:
            for (x1, y1, x2, y2), keypoints in zip(self._make_list_of_list(params[self._bboxes_key]),
                                                   self._make_list_of_list(params[self._keypoints_key])):
                x1, y1, x2, y2 = map(int, (x1 * w, y1 * h, x2 * w, y2 * h))
                keypoints *= [[w, h]]
                bboxes.append([x1, y1, x2, y2])
                kps_by_bbox.append(shape_to_landmarks(keypoints.astype(np.long)))

        mask_face_params_by_bbox = [self._generate_metadata_types() for _ in bboxes]

        return dict(bboxes=bboxes, kps_by_bbox=kps_by_bbox, mask_face_params_by_bbox=mask_face_params_by_bbox)

    @property
    def targets_as_params(self):
        return ['image', self._bboxes_key, self._keypoints_key]

    def get_transform_init_args_names(self):
        return 'pattern_weight', 'color_weight', 'bboxes_key', 'keypoints_key'

    def _create_kp_dlib_model(self):
        assert dlib is not None, 'You have to install dlib to make keypoints on the faces for mask creation'

        path_to_dlib_model = self._module_path / "dlib_models/shape_predictor_68_face_landmarks.dat"
        if not path_to_dlib_model.exists():
            download_dlib_model(path_to_dlib_model)
        return dlib.shape_predictor(str(path_to_dlib_model))

    def _generate_metadata_types(self):
        _colors = ["#fc1c1a", "#177ABC", "#94B6D2", "#A5AB81", "#DD8047", "#6b425e", "#e26d5a", "#c92c48",
                   "#6a506d", "#ffc900", "#ffffff", "#000000", "#49ff00"]
        text_path = self._module_path / 'masks/textures'
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
        Args = namedtuple('Args', ['pattern', 'pattern_weight', 'color', 'color_weight', 'mask_type'])

        mask_type = random.choice(get_available_mask_types(self._module_path / 'masks/masks.cfg'))
        color = random.choice(_colors)
        pattern = str(random.choice(_patterns))

        return Args(pattern, self._pattern_weight, color, self._color_weight, mask_type)

    @staticmethod
    def _make_list_of_list(some: List):
        return some if some[0] is list else [some]
