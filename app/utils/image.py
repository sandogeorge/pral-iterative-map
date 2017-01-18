"""Implements methods to keep track of changes to the image."""


image = {}


def setImage(key, img):
    image[key] = img


def getImage(key):
    return image[key]
