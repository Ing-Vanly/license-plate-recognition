import cv2
import numpy as np
import imutils
from skimage import measure


def sort_contours(contours):
    boxes = [cv2.boundingRect(c) for c in contours]

    sorted_data = sorted(
        zip(contours, boxes),
        key=lambda item: item[1][0]
    )

    return [item[0] for item in sorted_data]


def segment_characters(plate_img, fixed_width=400):

    hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
    value_channel = cv2.split(hsv)[2]

    threshold = cv2.adaptiveThreshold(
        value_channel,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    threshold = cv2.bitwise_not(threshold)

    plate_img = imutils.resize(
        plate_img,
        width=fixed_width
    )

    threshold = imutils.resize(
        threshold,
        width=fixed_width
    )

    threshold_bgr = cv2.cvtColor(
        threshold,
        cv2.COLOR_GRAY2BGR
    )

    labels = measure.label(
        threshold,
        background=0
    )

    character_mask = np.zeros(
        threshold.shape,
        dtype="uint8"
    )

    for label in np.unique(labels):

        if label == 0:
            continue

        label_mask = np.zeros(
            threshold.shape,
            dtype="uint8"
        )

        label_mask[labels == label] = 255

        contours, _ = cv2.findContours(
            label_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        contour = max(
            contours,
            key=cv2.contourArea
        )

        x, y, w, h = cv2.boundingRect(contour)

        if h == 0:
            continue

        aspect_ratio = w / float(h)

        solidity = (
            cv2.contourArea(contour)
            / float(w * h)
        )

        height_ratio = (
            h / float(plate_img.shape[0])
        )

        valid_aspect = aspect_ratio < 1.0
        valid_solidity = solidity > 0.15
        valid_height = 0.5 < height_ratio < 0.95

        if (
            valid_aspect
            and valid_solidity
            and valid_height
            and w > 14
        ):

            hull = cv2.convexHull(contour)

            cv2.drawContours(
                character_mask,
                [hull],
                -1,
                255,
                -1
            )

    contours, _ = cv2.findContours(
        character_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    contours = sort_contours(contours)

    characters = []

    padding = 4

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)

        x = max(x - padding, 0)
        y = max(y - padding, 0)

        character = threshold_bgr[
            y:y + h + padding * 2,
            x:x + w + padding * 2
        ]

        characters.append(character)

    return characters


class PlateFinder:

    def __init__(
        self,
        min_plate_area=4100,
        max_plate_area=30000
    ):

        self.min_area = min_plate_area
        self.max_area = max_plate_area

        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (22, 3)
        )

        self.characters = []
        self.coordinates = []

    def preprocess(self, image):

        blurred = cv2.GaussianBlur(
            image,
            (7, 7),
            0
        )

        gray = cv2.cvtColor(
            blurred,
            cv2.COLOR_BGR2GRAY
        )

        sobel_x = cv2.Sobel(
            gray,
            cv2.CV_8U,
            1,
            0,
            ksize=3
        )

        _, threshold = cv2.threshold(
            sobel_x,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        closed = cv2.morphologyEx(
            threshold,
            cv2.MORPH_CLOSE,
            self.kernel
        )

        return closed

    def get_contours(self, image):

        contours, _ = cv2.findContours(
            image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        return contours

    def ratio_check(
        self,
        area,
        width,
        height,
        min_ratio=2.5,
        max_ratio=7
    ):

        if width == 0 or height == 0:
            return False

        # Reject vertical/tall objects.
        # A normal plate should be wider than it is tall.
        if width <= height:
            return False

        ratio = width / float(height)

        if area < self.min_area:
            return False

        if area > self.max_area:
            return False

        if ratio < min_ratio:
            return False

        if ratio > max_ratio:
            return False

        return True

    def find_possible_plates(self, image):

        processed = self.preprocess(image)

        contours = self.get_contours(processed)

        plates = []

        self.characters = []
        self.coordinates = []

        for contour in contours:

            rect = cv2.minAreaRect(contour)

            (_, _), (width, height), _ = rect

            # minAreaRect can return width/height rotated,
            # so normalize them before checking.
            if height > width:
                width, height = height, width

            area = width * height

            if not self.ratio_check(
                area,
                width,
                height
            ):
                continue

            x, y, w, h = cv2.boundingRect(
                contour
            )

            # Extra check using the actual bounding box.
            # Reject very tall/narrow regions.
            if w <= h:
                continue

            plate = image[
                y:y + h,
                x:x + w
            ]

            if plate.size == 0:
                continue

            chars = segment_characters(
                plate
            )

            if chars is None:
                continue

            plates.append(plate)

            self.characters.append(chars)

            self.coordinates.append(
                (x, y, w, h)
            )

        return plates