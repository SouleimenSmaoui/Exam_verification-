import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar
from feuille_examen import select_roi, extract_table, detect_table_contours
#define a dictionary to store ROI coordinates for different sections
roi_dict = {"qr_code_examen": [0.0453,0.0769,0.1255,0.0676],}

if __name__ == "__main__":
    image_path = r"images\liste_emargement.jpg"
    image_path = cv2.resize(cv2.imread(image_path), None, fx=0.55, fy=0.55)
    # Example usage of select_roi
    x, y, w, h = select_roi(image_path)
    print(f"Manually selected ROI: x={x}, y={y}, w={w}, h={h}")
    cv2.destroyAllWindows()
