"""Implements functions used to draw on images."""

import cv2
import math
import numpy as np

from app.utils.image import getImage, setImage
from app.utils.models import HandPalmar, HandSide


def drawHandPalmarBounds(center, height):
    handModel = HandPalmar(center, height)

    # Rectangle for palm.
    cv2.rectangle(
        getImage(),
        handModel.top_left(),
        handModel.bottom_right(),
        (0, 255, 0),
        1
    )

    # Base point of fingers
    cv2.circle(getImage(), handModel.base(), 1, (0, 255, 0), 1)

    # Pinky.
    pinky_angle = -108 * math.pi / 180
    cv2.line(
        getImage(),
        handModel.base(),
        handModel.pinky(pinky_angle),
        (0, 255, 0),
        1
    )
    cv2.circle(
        getImage(),
        handModel.pinky(pinky_angle),
        4,
        (0, 255, 0),
        1
    )
    cv2.circle(
        getImage(),
        handModel.pinkybase(pinky_angle),
        4,
        (255, 0, 0),
        1
    )
    pb = handModel.pinkybox(pinky_angle)
    ppts = np.array([[pb[0][0], pb[0][1]], [pb[1][0], pb[1][1]],
                     [pb[2][0], pb[2][1]], [pb[3][0], pb[3][1]]], np.int32)
    ppts = ppts.reshape((-1, 1, 2))
    cv2.polylines(getImage(), [ppts], True, (0, 0, 255), 1)

    # Ring.
    ring_angle = -95 * math.pi / 180
    cv2.line(
        getImage(),
        handModel.base(),
        handModel.ring(ring_angle),
        (0, 255, 0),
        1
    )
    cv2.circle(
        getImage(),
        handModel.ring(ring_angle),
        4,
        (0, 255, 0),
        1
    )
    cv2.circle(
        getImage(),
        handModel.ringbase(ring_angle),
        4,
        (255, 0, 0),
        1
    )
    rb = handModel.ringbox(ring_angle)
    rpts = np.array([[rb[0][0], rb[0][1]], [rb[1][0], rb[1][1]],
                     [rb[2][0], rb[2][1]], [rb[3][0], rb[3][1]]], np.int32)
    rpts = rpts.reshape((-1, 1, 2))
    cv2.polylines(getImage(), [rpts], True, (0, 0, 255), 1)

    # Middle.
    middle_angle = -85 * math.pi / 180
    cv2.line(
        getImage(),
        handModel.base(),
        handModel.middle(middle_angle),
        (0, 255, 0),
        1
    )
    cv2.circle(
        getImage(),
        handModel.middle(middle_angle),
        4,
        (0, 255, 0),
        1
    )
    cv2.circle(
        getImage(),
        handModel.middlebase(middle_angle),
        4,
        (255, 0, 0),
        1
    )
    mb = handModel.middlebox(middle_angle)
    mpts = np.array([[mb[0][0], mb[0][1]], [mb[1][0], mb[1][1]],
                     [mb[2][0], mb[2][1]], [mb[3][0], mb[3][1]]], np.int32)
    mpts = mpts.reshape((-1, 1, 2))
    cv2.polylines(getImage(), [mpts], True, (0, 0, 255), 1)

    # Index.
    index_angle = -75 * math.pi / 180
    cv2.line(
        getImage(),
        handModel.base(),
        handModel.index(index_angle),
        (0, 255, 0),
        1
    )
    cv2.circle(
        getImage(),
        handModel.index(index_angle),
        4,
        (0, 255, 0),
        1
    )
    cv2.circle(
        getImage(),
        handModel.indexbase(index_angle),
        4,
        (255, 0, 0),
        1
    )
    ib = handModel.indexbox(index_angle)
    ipts = np.array([[ib[0][0], ib[0][1]], [ib[1][0], ib[1][1]],
                     [ib[2][0], ib[2][1]], [ib[3][0], ib[3][1]]], np.int32)
    ipts = ipts.reshape((-1, 1, 2))
    cv2.polylines(getImage(), [ipts], True, (0, 0, 255), 1)

    # Thumb.
    thumb_angle = -45 * math.pi / 180
    cv2.line(
        getImage(),
        handModel.base(),
        handModel.thumb(thumb_angle),
        (0, 255, 0),
        1
    )
    cv2.circle(
        getImage(),
        handModel.thumb(thumb_angle),
        4,
        (0, 255, 0),
        1
    )
    cv2.circle(
        getImage(),
        handModel.thumbbase(thumb_angle),
        4,
        (255, 0, 0),
        1
    )
    tb = handModel.thumbbox(thumb_angle)
    tpts = np.array([[tb[0][0], tb[0][1]], [tb[1][0], tb[1][1]],
                     [tb[2][0], tb[2][1]], [tb[3][0], tb[3][1]]], np.int32)
    tpts = tpts.reshape((-1, 1, 2))
    cv2.polylines(getImage(), [tpts], True, (0, 0, 255), 1)


def drawHandSideBounds(center, height):
    handModel = HandSide(center, height)

    # Rectangle for palm.
    cv2.rectangle(
        getImage(),
        handModel.top_left(),
        handModel.bottom_right(),
        (0, 255, 0),
        1
    )

    # Base point of fingers
    cv2.circle(getImage(), handModel.base(), 1, (0, 255, 0), 1)

    # Finger(s).
    finger_angle = -87 * math.pi / 180
    cv2.line(
        getImage(),
        handModel.base(),
        handModel.finger(finger_angle),
        (0, 255, 0),
        1
    )
    cv2.circle(
        getImage(),
        handModel.finger(finger_angle),
        4,
        (0, 255, 0),
        1
    )
    cv2.circle(
        getImage(),
        handModel.fingerbase(finger_angle),
        4,
        (255, 0, 0),
        1
    )
    fb = handModel.fingerbox(finger_angle)
    fpts = np.array([[fb[0][0], fb[0][1]], [fb[1][0], fb[1][1]],
                     [fb[2][0], fb[2][1]], [fb[3][0], fb[3][1]]], np.int32)
    fpts = fpts.reshape((-1, 1, 2))
    cv2.polylines(getImage(), [fpts], True, (0, 0, 255), 1)

    # Thumb.
    thumb_angle = -45 * math.pi / 180
    cv2.line(
        getImage(),
        handModel.base(),
        handModel.thumb(thumb_angle),
        (0, 255, 0),
        1
    )
    cv2.circle(
        getImage(),
        handModel.thumb(thumb_angle),
        4,
        (0, 255, 0),
        1
    )
    cv2.circle(
        getImage(),
        handModel.thumbbase(thumb_angle),
        4,
        (255, 0, 0),
        1
    )
    tb = handModel.thumbbox(thumb_angle)
    tpts = np.array([[tb[0][0], tb[0][1]], [tb[1][0], tb[1][1]],
                     [tb[2][0], tb[2][1]], [tb[3][0], tb[3][1]]], np.int32)
    tpts = tpts.reshape((-1, 1, 2))
    cv2.polylines(getImage(), [tpts], True, (0, 0, 255), 1)