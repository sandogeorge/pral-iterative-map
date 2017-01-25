"""Mathematical models used to construct 2D wireframes."""

import cv2
import numpy as np

from app.utils.helper import checklineIntersection, lineIntersection, rotatePoint
from config import app_config


class Hand(object):
    """Base class for hand models."""

    # Center point of palm.
    center = [0, 0]

    # Height of palm.
    height = 0

    # Width of palm.
    width = 0

    # Thickness of finger
    finger_width = app_config.FINGER_WIDTH

    # Result of iterative MAP estimation.
    amax = None

    def __init__(self, center_point, palm_height, amax=None):
        self.center[0] = center_point[0]
        self.center[1] = center_point[1]
        self.height = palm_height
        self.amax = amax


class HandPalmar(Hand):

    def __init__(self, center_point, palm_height, amax=None):
        super().__init__(center_point, palm_height, amax)
        self.width = round(0.90 * palm_height)

    def peak(self):
        return (int(self.center[0]), int(round(self.center[1] - (self.height / 2))))

    def base(self):
        return (int(self.center[0]), int(round(self.center[1] + (self.height / 2))))

    def top_left(self):
        return (int(round(self.peak()[0] - (self.width / 2))), int(self.peak()[1]))

    def top_right(self):
        return (int(round(self.peak()[0] + (self.width / 2))), int(self.peak()[1]))

    def bottom_right(self):
        return (int(round(self.base()[0] + (self.width / 2))), int(self.base()[1]))

    def bottom_left(self):
        return (int(round(self.base()[0] - (self.width / 2))), int(self.base()[1]))

    def pinky(self, theta):
        if self.amax is not None:
            return tuple(self.amax[2][0])

        return (
            int(round(
                self.base()[0] + (app_config.PINKY_DEFAULT_RATIO * self.height * np.cos(theta)))),
            int(round(self.base()[
                1] + (app_config.PINKY_DEFAULT_RATIO * self.height * np.sin(theta))))
        )

    def pinkybase(self, theta):
        i = None
        if checklineIntersection(self.top_left(), self.top_right(),
                                 self.base(), self.pinky(theta)):
            i = lineIntersection(
                [self.top_left(), self.top_right()],
                [self.base(), self.pinky(theta)])
        else:
            i = lineIntersection(
                [self.top_left(), self.bottom_left()],
                [self.base(), self.pinky(theta)])

        return (int(round(i[0])), int(round(i[1])))

    def pinkylength(self, theta):
        return cv2.norm(np.array(self.pinkybase(theta)), np.array(self.pinky(theta)))

    def pinkybox(self, theta):
        pinky = self.pinky(theta)
        pinkylen = self.pinkylength(theta)
        pinkybase = self.pinkybase(theta)
        pinkytop = (pinkybase[0], pinkybase[1] - pinkylen + 2)

        pinkytl = (pinkytop[0] - self.finger_width, pinkytop[1])
        pinkytr = (pinkytop[0] + self.finger_width, pinkytop[1])
        pinkybr = (pinkybase[0] + self.finger_width, pinkybase[1])
        pinkybl = (pinkybase[0] - self.finger_width, pinkybase[1])

        if self.amax is not None:
            dy = pinky[1] - pinkybase[1]
            dx = pinky[0] - pinkybase[0]
            theta = np.arctan2(dy, dx)

        thetar = theta + (90 * np.pi / 180)
        tlr = rotatePoint(pinkytl, pinkybase, thetar)
        trr = rotatePoint(pinkytr, pinkybase, thetar)
        brr = rotatePoint(pinkybr, pinkybase, thetar)
        blr = rotatePoint(pinkybl, pinkybase, thetar)

        return [
            (round(tlr[0]), round(tlr[1])),
            (round(trr[0]), round(trr[1])),
            (round(brr[0]), round(brr[1])),
            (round(blr[0]), round(blr[1])),
        ]

    def ring(self, theta):
        if self.amax is not None:
            return tuple(self.amax[2][1])

        return (
            int(round(
                self.base()[0] + (app_config.RING_DEFAULT_RATIO * self.height * np.cos(theta)))),
            int(round(self.base()[
                1] + (app_config.RING_DEFAULT_RATIO * self.height * np.sin(theta))))
        )

    def ringbase(self, theta):
        i = lineIntersection(
            [self.top_left(), self.top_right()],
            [self.base(), self.ring(theta)])

        return (int(round(i[0])), int(round(i[1])))

    def ringlength(self, theta):
        return cv2.norm(np.array(self.ringbase(theta)), np.array(self.ring(theta)))

    def ringbox(self, theta):
        ring = self.ring(theta)
        ringlen = self.ringlength(theta)
        ringbase = self.ringbase(theta)
        ringtop = (ringbase[0], ringbase[1] - ringlen + 2)

        ringtl = (ringtop[0] - self.finger_width, ringtop[1])
        ringtr = (ringtop[0] + self.finger_width, ringtop[1])
        ringbr = (ringbase[0] + self.finger_width, ringbase[1])
        ringbl = (ringbase[0] - self.finger_width, ringbase[1])

        if self.amax is not None:
            dy = ring[1] - ringbase[1]
            dx = ring[0] - ringbase[0]
            theta = np.arctan2(dy, dx)

        thetar = theta + (90 * np.pi / 180)
        tlr = rotatePoint(ringtl, ringbase, thetar)
        trr = rotatePoint(ringtr, ringbase, thetar)
        brr = rotatePoint(ringbr, ringbase, thetar)
        blr = rotatePoint(ringbl, ringbase, thetar)

        return [
            (round(tlr[0]), round(tlr[1])),
            (round(trr[0]), round(trr[1])),
            (round(brr[0]), round(brr[1])),
            (round(blr[0]), round(blr[1])),
        ]

    def middle(self, theta):
        if self.amax is not None:
            return tuple(self.amax[2][2])

        return (
            int(round(
                self.base()[0] + (app_config.MIDDLE_DEFAULT_RATIO * self.height * np.cos(theta)))),
            int(round(self.base()[
                1] + (app_config.MIDDLE_DEFAULT_RATIO * self.height * np.sin(theta))))
        )

    def middlebase(self, theta):
        i = lineIntersection(
            [self.top_left(), self.top_right()],
            [self.base(), self.middle(theta)])

        return (int(round(i[0])), int(round(i[1])))

    def middlelength(self, theta):
        return cv2.norm(np.array(self.middlebase(theta)), np.array(self.middle(theta)))

    def middlebox(self, theta):
        middle = self.middle(theta)
        middlelen = self.middlelength(theta)
        middlebase = self.middlebase(theta)
        middletop = (middlebase[0], middlebase[1] - middlelen + 2)

        middletl = (middletop[0] - self.finger_width, middletop[1])
        middletr = (middletop[0] + self.finger_width, middletop[1])
        middlebr = (middlebase[0] + self.finger_width, middlebase[1])
        middlebl = (middlebase[0] - self.finger_width, middlebase[1])

        if self.amax is not None:
            dy = middle[1] - middlebase[1]
            dx = middle[0] - middlebase[0]
            theta = np.arctan2(dy, dx)

        thetar = theta + (90 * np.pi / 180)
        tlr = rotatePoint(middletl, middlebase, thetar)
        trr = rotatePoint(middletr, middlebase, thetar)
        brr = rotatePoint(middlebr, middlebase, thetar)
        blr = rotatePoint(middlebl, middlebase, thetar)

        return [
            (round(tlr[0]), round(tlr[1])),
            (round(trr[0]), round(trr[1])),
            (round(brr[0]), round(brr[1])),
            (round(blr[0]), round(blr[1])),
        ]

    def index(self, theta):
        if self.amax is not None:
            return tuple(self.amax[2][3])

        return (
            int(round(
                self.base()[0] + (app_config.INDEX_DEFAULT_RATIO * self.height * np.cos(theta)))),
            int(round(self.base()[
                1] + (app_config.INDEX_DEFAULT_RATIO * self.height * np.sin(theta))))
        )

    def indexbase(self, theta):
        i = None
        if checklineIntersection(self.top_left(), self.top_right(),
                                 self.base(), self.index(theta)):
            i = lineIntersection(
                [self.top_left(), self.top_right()],
                [self.base(), self.index(theta)])
        else:
            i = lineIntersection(
                [self.top_right(), self.bottom_right()],
                [self.base(), self.index(theta)])

        return (int(round(i[0])), int(round(i[1])))

    def indexlength(self, theta):
        return cv2.norm(np.array(self.indexbase(theta)), np.array(self.index(theta)))

    def indexbox(self, theta):
        index = self.index(theta)
        indexlen = self.indexlength(theta)
        indexbase = self.indexbase(theta)
        indextop = (indexbase[0], indexbase[1] - indexlen + 2)

        indextl = (indextop[0] - self.finger_width, indextop[1])
        indextr = (indextop[0] + self.finger_width, indextop[1])
        indexbr = (indexbase[0] + self.finger_width, indexbase[1])
        indexbl = (indexbase[0] - self.finger_width, indexbase[1])

        if self.amax is not None:
            dy = index[1] - indexbase[1]
            dx = index[0] - indexbase[0]
            theta = np.arctan2(dy, dx)

        thetar = theta + (90 * np.pi / 180)
        tlr = rotatePoint(indextl, indexbase, thetar)
        trr = rotatePoint(indextr, indexbase, thetar)
        brr = rotatePoint(indexbr, indexbase, thetar)
        blr = rotatePoint(indexbl, indexbase, thetar)

        return [
            (round(tlr[0]), round(tlr[1])),
            (round(trr[0]), round(trr[1])),
            (round(brr[0]), round(brr[1])),
            (round(blr[0]), round(blr[1])),
        ]

    def thumb(self, theta):
        if self.amax is not None:
            return tuple(self.amax[2][4])

        return (
            int(round(
                self.base()[0] + (app_config.THUMB_DEFAULT_RATIO * self.height * np.cos(theta)))),
            int(round(self.base()[
                1] + (app_config.THUMB_DEFAULT_RATIO * self.height * np.sin(theta))))
        )

    def thumbbase(self, theta):
        i = lineIntersection(
            [self.top_right(), self.bottom_right()],
            [self.base(), self.thumb(theta)])

        return (int(round(i[0])), int(round(i[1])))

    def thumblength(self, theta):
        return cv2.norm(np.array(self.thumbbase(theta)), np.array(self.thumb(theta)))

    def thumbbox(self, theta):
        thumb = self.thumb(theta)
        thumblen = self.thumblength(theta)
        thumbbase = self.thumbbase(theta)
        thumbtop = (thumbbase[0], thumbbase[1] - thumblen + 2)

        thumbtl = (thumbtop[0] - self.finger_width, thumbtop[1])
        thumbtr = (thumbtop[0] + self.finger_width, thumbtop[1])
        thumbbr = (thumbbase[0] + self.finger_width, thumbbase[1])
        thumbbl = (thumbbase[0] - self.finger_width, thumbbase[1])

        if self.amax is not None:
            dy = thumb[1] - thumbbase[1]
            dx = thumb[0] - thumbbase[0]
            theta = np.arctan2(dy, dx)

        thetar = theta + (90 * np.pi / 180)
        tlr = rotatePoint(thumbtl, thumbbase, thetar)
        trr = rotatePoint(thumbtr, thumbbase, thetar)
        brr = rotatePoint(thumbbr, thumbbase, thetar)
        blr = rotatePoint(thumbbl, thumbbase, thetar)

        return [
            (round(tlr[0]), round(tlr[1])),
            (round(trr[0]), round(trr[1])),
            (round(brr[0]), round(brr[1])),
            (round(blr[0]), round(blr[1])),
        ]


class HandSide(Hand):

    def __init__(self, center_point, palm_height):
        super().__init__(center_point, palm_height)
        self.width = round(0.31 * palm_height)

    def peak(self):
        return (self.center[0], round(self.center[1] - (self.height / 2)))

    def base(self):
        return (self.center[0], round(self.center[1] + (self.height / 2)))

    def top_left(self):
        return (round(self.peak()[0] - (self.width / 2)), self.peak()[1])

    def top_right(self):
        return (round(self.peak()[0] + (self.width / 2)), self.peak()[1])

    def bottom_right(self):
        return (round(self.base()[0] + (self.width / 2)), self.base()[1])

    def bottom_left(self):
        return (round(self.base()[0] - (self.width / 2)), self.base()[1])

    def finger(self, theta):
        return (
            round(
                self.base()[0] + (1.81 * self.height * np.cos(theta))),
            round(self.base()[1] + (1.81 * self.height * np.sin(theta)))
        )

    def fingerbase(self, theta):
        i = None
        if checklineIntersection(self.top_left(), self.top_right(),
                                 self.base(), self.finger(theta)):
            i = lineIntersection(
                [self.top_left(), self.top_right()],
                [self.base(), self.finger(theta)])
        else:
            i = lineIntersection(
                [self.top_right(), self.bottom_right()],
                [self.base(), self.finger(theta)])

        return (round(i[0]), round(i[1]))

    def fingerlength(self, theta):
        return cv2.norm(self.fingerbase(theta), self.finger(theta))

    def fingerbox(self, theta):
        fingerlen = self.fingerlength(theta)
        fingerbase = self.fingerbase(theta)
        fingertop = (fingerbase[0], fingerbase[1] - fingerlen + 2)

        fingertl = (fingertop[0] - 5, fingertop[1])
        fingertr = (fingertop[0] + 5, fingertop[1])
        fingerbr = (fingerbase[0] + 5, fingerbase[1])
        fingerbl = (fingerbase[0] - 5, fingerbase[1])

        thetar = theta + (90 * np.pi / 180)
        tlr = rotatePoint(fingertl, fingerbase, thetar)
        trr = rotatePoint(fingertr, fingerbase, thetar)
        brr = rotatePoint(fingerbr, fingerbase, thetar)
        blr = rotatePoint(fingerbl, fingerbase, thetar)

        return [
            (round(tlr[0]), round(tlr[1])),
            (round(trr[0]), round(trr[1])),
            (round(brr[0]), round(brr[1])),
            (round(blr[0]), round(blr[1])),
        ]

    def thumb(self, theta):
        return (
            round(
                self.base()[0] + (0.63 * self.height * np.cos(theta))),
            round(self.base()[1] + (0.63 * self.height * np.sin(theta)))
        )

    def thumbbase(self, theta):
        i = lineIntersection(
            [self.top_right(), self.bottom_right()],
            [self.base(), self.thumb(theta)])

        return (round(i[0]), round(i[1]))

    def thumblength(self, theta):
        return cv2.norm(self.thumbbase(theta), self.thumb(theta))

    def thumbbox(self, theta):
        thumblen = self.thumblength(theta)
        thumbbase = self.thumbbase(theta)
        thumbtop = (thumbbase[0], thumbbase[1] - thumblen + 2)

        thumbtl = (thumbtop[0] - 5, thumbtop[1])
        thumbtr = (thumbtop[0] + 5, thumbtop[1])
        thumbbr = (thumbbase[0] + 5, thumbbase[1])
        thumbbl = (thumbbase[0] - 5, thumbbase[1])

        thetar = theta + (90 * np.pi / 180)
        tlr = rotatePoint(thumbtl, thumbbase, thetar)
        trr = rotatePoint(thumbtr, thumbbase, thetar)
        brr = rotatePoint(thumbbr, thumbbase, thetar)
        blr = rotatePoint(thumbbl, thumbbase, thetar)

        return [
            (round(tlr[0]), round(tlr[1])),
            (round(trr[0]), round(trr[1])),
            (round(brr[0]), round(brr[1])),
            (round(blr[0]), round(blr[1])),
        ]
