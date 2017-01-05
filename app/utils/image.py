"""Implements methods to keep track of changes to the image."""


image = None


def setImage(img):
    global image
    image = img


def getImage():
    global image
    return image