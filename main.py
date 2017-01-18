"""Application entry point."""


import os

import numpy as np
import cv2

from config import app_config

from app.utils.draw import (
    drawHandPalmarBounds,
    drawHandPalmarFilled,
    drawHandSideBounds,
    drawMouseInfo
)
from app.utils.image import getImage, setImage

np.set_printoptions(threshold=np.inf)


def main():
    """Create window and display image with bounds drawn."""

    for i, window in enumerate(app_config.WINDOWS):
        j = i % 3
        k = int(np.floor(i / 3))
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window, j * 350 + 150, k * 350 + 35)

    import_images()
    train_expectation()
    # train_maximization()

    while True:
        cv2.imshow('Training Image', getImage('em_input'))
        cv2.imshow('Training Output', getImage('em_output'))
        cv2.imshow('Test Input', getImage('test_input'))
        cv2.imshow('Test Output', getImage('test_output'))

        if (cv2.waitKey(0) & 0xFF) in [27, 255]:
            cv2.destroyAllWindows()
            break


def train_maximization():
    w = app_config.IMG_WIDTH
    h = app_config.IMG_HEIHGT
    img = getImage('apt-test')
    res = np.zeros((h, w, 3), np.uint8)

    # Resize training image.
    cv2.resize(img, (w, h), res)
    img = res
    setImage('test_input', img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    probs0 = np.load(app_config.FILES['probs'])

    em = cv2.ml.EM_create()
    em.setClustersNumber(2)
    em.setTermCriteria(
        (cv2.TERM_CRITERIA_EPS, 0, 0.03))
    em.trainM(np.reshape(gray, (h * w, 1)), probs0)

    probs2 = em.predict(np.reshape(gray.astype(float), (h * w, 1)))

    test_output = np.zeros((h, w, 3), np.uint8)
    for i, pixel in enumerate(probs2[1]):
        j = int(np.floor(i / w))
        k = i % w

        if pixel[0] < 0.5:
            test_output[j][k] = np.array([255, 255, 255])
        else:
            test_output[j][k] = np.array([0, 0, 0])

    setImage('test_output', test_output)


def train_expectation():
    labels = None
    logs = None
    probs = None

    w = app_config.IMG_WIDTH
    h = app_config.IMG_HEIHGT
    tr_img = getImage('training')
    tr_res = np.zeros((h, w, 3), np.uint8)

    # Resize training image.
    cv2.resize(tr_img, (w, h), tr_res)
    tr_img = tr_res

    # Smooth training image.
    blur = cv2.blur(tr_img, (5, 5))
    tr_img = blur
    setImage('em_input', tr_img)
    gray = cv2.cvtColor(tr_img, cv2.COLOR_BGR2GRAY)
#     if os.path.isfile(app_config.FILES['labels']):
#         labels = np.load(app_config.FILES['labels'])
#         logs = np.load(app_config.FILES['logs'])
#         probs = np.load(app_config.FILES['probs'])
#     else:
    em = cv2.ml.EM_create()
    em.setClustersNumber(5)
    success, logs, labels, probs = em.trainEM(
        np.reshape(tr_img, (h * w, 3)))

    if success:
        np.save(app_config.FILES['labels'], labels)
        np.save(app_config.FILES['logs'], logs)
        np.save(app_config.FILES['probs'], probs)

    em_output = np.zeros((h, w, 3), np.uint8)
    for i, pixel in enumerate(labels):
        j = int(np.floor(i / w))
        k = i % w

        if pixel == 1:
            em_output[j][k] = np.array([255, 255, 255])
        else:
            em_output[j][k] = np.array([0, 0, 0])

    setImage('em_output', em_output)

    test_img = getImage('apt-test')
    test_res = np.zeros((h, w, 3), np.uint8)
    cv2.resize(test_img, (w, h), test_res)
    test_img = test_res
    setImage('test_input', test_res)

    gray2 = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(test_img, 100, 200)

    probs2 = em.predict(np.reshape(test_img.astype(float), (h * w, 3)))

    test_output = np.zeros((h, w, 3), np.uint8)
    for i, pixel in enumerate(probs2[1]):
        j = int(np.floor(i / w))
        k = i % w

        if pixel[0] > 0.5:
            test_output[j][k] = np.array([255, 255, 255])
        else:
            test_output[j][k] = np.array([0, 0, 0])

    setImage('test_output', test_output)


def import_images():
    for name, img in app_config.IMAGES.items():
        setImage(name, cv2.imread(img, cv2.IMREAD_COLOR))


if __name__ == '__main__':
    main()
