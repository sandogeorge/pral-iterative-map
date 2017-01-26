"""Image processing functions."""


import cv2
import numpy as np
from scipy.spatial import KDTree


from app.utils.draw import drawHandPalmarBounds
from app.utils.image import (
    getImage, setImage,
)
from app.utils.models import HandPalmar
from config import app_config


__all__ = ['model_observation', 'do_skin_detection', 'do_threshold',
           'do_edges', 'do_contours', 'import_images']


def model_observation(cnt):
    if cnt is not None:
        img = getImage('contours')

        (cx, cy), cr = cv2.minEnclosingCircle(cnt)
        center = (int(round(cx)), int(round(cy)))
        radius = int(round(cr))
        cv2.circle(img, center, 4, app_config.COLORS['blue'], 2)
        cv2.circle(img, center, radius, app_config.COLORS['blue'], 2)

        candidates = list(map(lambda x: x[0], cv2.convexHull(cnt)))
        candidate_tree = KDTree(candidates)

        for p in candidates:
            cv2.circle(img, (p[0], p[1]), 4, app_config.COLORS['red'], 2)

        setImage('estimate', getImage('og').copy())
        setImage('hypothesis', getImage('og').copy())

        aligned = list(
            map(lambda x: x[0], filter(lambda p: p[0][0] == center[0], cnt)))
        base_offset = max(aligned, key=lambda x: x[0])

        base = (base_offset[0] - 8, base_offset[1] - 10)
        palm_height = base_offset[1] - center[1]
        palm_center = (base[0], base[1] - int(np.floor(palm_height / 2)))
        cv2.circle(img, palm_center, 4, app_config.COLORS['red'], 2)

        model = HandPalmar(palm_center, palm_height)

        pinky_angle = -app_config.PINKY_DEFAULT_ANGLE * np.pi / 180
        pinky = model.pinky(pinky_angle)

        ring_angle = -app_config.RING_DEFAULT_ANGLE * np.pi / 180
        ring = model.ring(ring_angle)

        middle_angle = -app_config.MIDDLE_DEFAULT_ANGLE * np.pi / 180
        middle = model.middle(middle_angle)

        index_angle = -app_config.INDEX_DEFAULT_ANGLE * np.pi / 180
        index = model.index(index_angle)

        thumb_angle = -app_config.THUMB_DEFAULT_ANGLE * np.pi / 180
        thumb = model.thumb(thumb_angle)

        classes = {
            'pinky': pinky,
            'ring': ring,
            'middle': middle,
            'index': index,
            'thumb': thumb
        }

        # Probabiliity of any given class hypothesis (Prior).
        py = 1 / len(classes)

        # Probability of any given candidate point.
        pp = 1 / len(candidates)

        maps = []
        for candidate in candidates:
            posteriors = []
            for y, loc in classes.items():
                # Probability of point given class.
                # 1. Find closest points (s < 15) to class hypothesis.
                leaves = candidate_tree.query(
                    loc, len(candidates), distance_upper_bound=15)
                closest = list(filter(
                    lambda x: np.isfinite(x[0]), np.dstack((leaves[0], leaves[1]))[0]))
                # 2. Calculate similarity scores for each of the closest points
                #    based on Euclidean distance from class hypothesis.
                closest = list(
                    map(lambda x: [1 / (1 + x[0]), x[1]], closest))
                # 3. Calculate the total similarity and use it to calculate
                #    the probability that each point belongs to the class
                #    based on their similarities.
                total_sim = sum(map(lambda x: x[0], closest))
                dist = list(
                    map(lambda x: [x[0] / total_sim, candidate_tree.data[int(x[1])]], closest))
                # 4. If candidate in closest at i then P(candidate|y) = P(closest[i]) else
                #    P(candidate|y) = 0
                closest = list(
                    filter(lambda x: x[1][0] == candidate[0] and x[1][1] == candidate[1], dist))
                pcgy = 0
                if len(closest) == 1:
                    pcgy = closest[0][0]
                # 5. Calculate class posterior
                cp = (pcgy * py) / pp
                posteriors.append((y, cp))
            argmax = max(posteriors, key=lambda x: x[1])
            maps.append((candidate, argmax))

        estimates = {}
        for m in maps:
            if m[1][0] in estimates.keys():
                if m[1][1] > estimates[m[1][0]][1]:
                    estimates[m[1][0]] = (m[0], m[1][1])
            else:
                estimates[m[1][0]] = (m[0], m[1][1])

        if len(estimates.keys()) == 5:
            amax = [
                estimates['pinky'][0],
                estimates['ring'][0],
                estimates['middle'][0],
                estimates['index'][0],
                estimates['thumb'][0]
            ]
            model_projection('hypothesis', palm_center, palm_height)
            model_projection('estimate', palm_center, palm_height, amax)


def model_projection(img_key, palm_center, palm_height, amax=None):
    setImage(img_key, getImage('og').copy())
    drawHandPalmarBounds(getImage(img_key), palm_center, palm_height, amax)


def do_skin_detection():
    # define range of HSV intensities that are indicative of skin.
    lh = cv2.getTrackbarPos('LH', 'Skin Detection')
    ls = cv2.getTrackbarPos('LS', 'Skin Detection')
    lv = cv2.getTrackbarPos('LV', 'Skin Detection')

    uh = cv2.getTrackbarPos('UH', 'Skin Detection')
    us = cv2.getTrackbarPos('US', 'Skin Detection')
    uv = cv2.getTrackbarPos('UV', 'Skin Detection')

    lower = np.array([lh, ls, lv], np.uint8)
    upper = np.array([uh, us, uv], np.uint8)

    # Load image and resize it.
    img = getImage('og').copy()

    # Covert colorspace to HSV.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define skin mask.
    # Apply erosions and dilations.
    # Blur to remove noise.
    skin_mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=1)
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)

    # Apply mask to frame to get skin region.
    skin = cv2.bitwise_and(img, img, mask=skin_mask)
    setImage('skin', skin)


def do_threshold():
    img = getImage('skin').copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tv = cv2.getTrackbarPos('Threshold', 'Thresholding')
    _, thresh = cv2.threshold(gray, tv, 255, cv2.THRESH_BINARY)
    setImage('thresh', thresh)


def do_edges():
    img = getImage('thresh').copy()
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    edges = cv2.Canny(blur, 100, 200)
    setImage('edges', edges)


def do_contours():
    img = getImage('edges').copy()
    border = img.copy()
    _, contours, _ = cv2.findContours(
        border, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        cnt_max = max(contours, key=lambda cnt: cv2.contourArea(cnt))
        og = getImage('og').copy()
        cv2.drawContours(og, [cnt_max], -1, app_config.COLORS['green'], 2)

        setImage('contours', og)
        return cnt_max
    else:
        w = app_config.IMG_WIDTH
        h = app_config.IMG_HEIHGT
        setImage('contours', np.zeros((h, w, 1), np.uint8))
        return None


def import_images():
    for name, img in app_config.IMAGES.items():
        setImage(name, cv2.imread(img, cv2.IMREAD_COLOR))
