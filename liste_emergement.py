"""liste_emergement.py

Helper script for scanning the attendance list image and selecting ROIs.

This module demonstrates how to load the sample attendance image and call
`select_roi` from `feuille_examen.py` to manually choose a region of interest
(ROI). The repo contains additional helper functions such as
`extract_table` and `detect_table_contours` which may be used by other
processing scripts.

Notes:
 - `roi_dict` stores relative ROI coordinates (left, top, width, height)
   as fractions of the image dimensions. Adjust as needed for other images.
 - `select_roi` is expected to open an OpenCV GUI window for manual selection.
   Close the window or press the appropriate key to finish selection.
"""

import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar

# Functions expected to exist in `feuille_examen.py`. They are imported here
# for convenience — this script acts as a small runnable example.
from feuille_examen import select_roi, extract_table, detect_table_contours

# Example ROI dictionary: keys name a logical region and the values are
# relative coordinates: [x_frac, y_frac, w_frac, h_frac]. These are useful
# to store positions that are robust to different image resolutions.
roi_dict = {
    "qr_code_examen": [0.0453, 0.0769, 0.1255, 0.0676],
}


def load_and_resize(path: str, fx: float = 0.55, fy: float = 0.55):
    """Load an image from `path` and resize it by factors `fx`, `fy`.

    Raises a FileNotFoundError if the image cannot be loaded.
    Returns the resized image (numpy.ndarray).
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at: {path}")
    return cv2.resize(img, None, fx=fx, fy=fy)


if __name__ == "__main__":
    # Path relative to the repository root. Change if your image is elsewhere.
    src_path = r"images\liste_emargement.jpg"

    # Load and resize the image for interactive ROI selection.
    image = load_and_resize(src_path)

    # Example usage: ask the user to select an ROI. `select_roi` should return
    # integers (x, y, w, h) in pixel coordinates relative to the provided image.
    x, y, w, h = select_roi(image)
    print(f"Manually selected ROI: x={x}, y={y}, w={w}, h={h}")

    # Always close OpenCV windows opened by `select_roi` or other GUI calls.
    cv2.destroyAllWindows()
