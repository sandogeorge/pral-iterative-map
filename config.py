"""Application configuration classes."""

import os

BASEDIR = os.path.abspath(os.path.dirname(__file__))


class Config(object):

    WINDOWS = [
        'Input Image',
        'Skin Detection',
        'Thresholding',
        'Edge Detection',
        'Contours',
        'Hypothesis',
        'MAP Estimate'
    ]

    IMAGES = {
        'training': os.path.join(BASEDIR, 'app', 'files', 'hand-train.jpg'),
        'apt-test': os.path.join(BASEDIR, 'app', 'files', 'hand-apt.jpg'),
        'false-test': os.path.join(BASEDIR, 'app', 'files', 'img-false.jpg'),
        'anon': os.path.join(BASEDIR, 'app', 'files', 'anon.png'),
        'no-bg-test': os.path.join(BASEDIR, 'app', 'files', 'no-bg-test.jpg'),
        'hand-new': os.path.join(BASEDIR, 'app', 'files', 'hand-new.jpg'),
        'apt-test-2': os.path.join(BASEDIR, 'app', 'files', 'hand-apt-2.jpg')
    }

    IMG_WIDTH = 256
    IMG_HEIHGT = 256

    # Default angles (degrees) and palm to finger ratios.
    PINKY_DEFAULT_ANGLE = 119
    PINKY_DEFAULT_RATIO = 1.47

    RING_DEFAULT_ANGLE = 103
    RING_DEFAULT_RATIO = 1.77

    MIDDLE_DEFAULT_ANGLE = 90
    MIDDLE_DEFAULT_RATIO = 1.88

    INDEX_DEFAULT_ANGLE = 75
    INDEX_DEFAULT_RATIO = 1.75

    THUMB_DEFAULT_ANGLE = 28
    THUMB_DEFAULT_RATIO = 1.13

    FINGER_WIDTH = 6

    COLORS = {
        'black': (0, 0, 0),
        'white': (255, 255, 255),
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0)
    }


app_config = Config()
