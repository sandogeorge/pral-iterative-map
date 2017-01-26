"""Application entry point."""


import cv2
import numpy as np

from app.utils.image import (
    getImage, setImage,
)
from app.utils.processing import *
from config import app_config


def main():
    """"""

    for i, window in enumerate(app_config.WINDOWS):
        j = i % 4
        k = int(np.floor(i / 4))
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        if j in [1, 2] and k == 1:
            cv2.moveWindow(window, (j + 1) * 330 + 30, k * 300 + 80)
            continue

        cv2.moveWindow(window, j * 330 + 30, k * 300 + 35)

    import_images()
    w = app_config.IMG_WIDTH
    h = app_config.IMG_HEIHGT

    img = getImage('apt-test-2')
    res = np.zeros((h, w, 3), np.uint8)
    cv2.resize(img, (w, h), res)

    setImage('og', res)
    setImage('hypothesis', res.copy())
    setImage('estimate', res.copy())

    cv2.createTrackbar('Threshold', 'Thresholding', 175, 255, lambda x: x)

    cv2.createTrackbar('LH', 'Skin Detection', 0, 180, lambda x: x)
    cv2.createTrackbar('LS', 'Skin Detection', 23, 255, lambda x: x)
    cv2.createTrackbar('LV', 'Skin Detection', 160, 255, lambda x: x)

    cv2.createTrackbar('UH', 'Skin Detection', 17, 180, lambda x: x)
    cv2.createTrackbar('US', 'Skin Detection', 80, 255, lambda x: x)
    cv2.createTrackbar('UV', 'Skin Detection', 255, 255, lambda x: x)

    while(1):
        cv2.imshow('Input Image', getImage('og'))

        do_skin_detection()
        cv2.imshow('Skin Detection', getImage('skin'))

        do_threshold()
        cv2.imshow('Thresholding', getImage('thresh'))

        do_edges()
        cv2.imshow('Edge Detection', getImage('edges'))

        cnt_max = do_contours()
        model_observation(cnt_max)
        cv2.imshow('Contours', getImage('contours'))
        cv2.imshow('Hypothesis', getImage('hypothesis'))
        cv2.imshow('MAP Estimate', getImage('estimate'))

        if (cv2.waitKey(0) & 0xFF) in [27, 255]:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
