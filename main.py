"""Application entry point."""

import numpy as np
import cv2

from config import app_config

from app.utils.draw import drawHandPalmarBounds, drawHandSideBounds
from app.utils.image import setImage, getImage


def main():
    """Create window and display image with bounds drawn."""
    process_input_image()
    cv2.namedWindow(app_config.WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        cv2.imshow(app_config.WINDOW_NAME, getImage())
        if (cv2.waitKey(0) & 0xFF) in [27, 255]:
            cv2.destroyAllWindows()
            break


def process_input_image():
    if app_config.IMAGE is None:
        setImage(np.zeros((512, 512, 3), np.uint8))
    else:
        setImage(cv2.imread(app_config.IMAGE, cv2.IMREAD_COLOR))

#     drawHandPalmarBounds((150, 256), 125)
#     drawHandSideBounds((350, 256), 125)

if __name__ == '__main__':
    main()
