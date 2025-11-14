import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar
import pytesseract 
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

#--------------------------------------
#Configuration
#--------------------------------------
# roi coordinates are now relative (x_rel, y_rel, w_rel, h_rel) in [0,1]
roi_dict = {
    "table_1": [0.0015,0.2363,0.7903,0.0839],
    "table_2": [0.7771,0.2456,0.2038,0.0725],
    "qr_code_etudiant": [0.0044,0.1358,0.7874,0.1212],
    "qr_code_exament": [0.0073,0.0187,0.1935,0.1233],
    "note_exament": [0.7930,0.1531,0.1219,0.0662]
}

#--------------------------------------
#fonction to select manually ROI on the image and return coordinates x y w h (relative)
#--------------------------------------
def select_roi(image):
    # Select ROI (returns absolute x,y,w,h)
    r = cv2.selectROI("Select ROI", image, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    x_abs, y_abs, w_abs, h_abs = map(int, r)
    h_img, w_img = image.shape[:2]

    # Convert to relative coordinates
    x_rel = x_abs / w_img
    y_rel = y_abs / h_img
    w_rel = w_abs / w_img
    h_rel = h_abs / h_img

    # write the coordinates on the terminal (relative)
    print(f"Selected ROI (relative) - {x_rel:.4f},{y_rel:.4f},{w_rel:.4f},{h_rel:.4f}")
    print(f"Selected ROI (absolute) - {x_abs},{y_abs},{w_abs},{h_abs}")
    return x_rel, y_rel, w_rel, h_rel

#--------------------------------------
# helper: convert relative roi to absolute pixel roi
#--------------------------------------
def _roi_to_absolute(roi, image_shape):
    """
    Accepts roi as either relative (values in [0,1]) or absolute (values > 1).
    Returns integer absolute (x, y, w, h) clipped to image bounds.
    """
    h_img, w_img = image_shape[:2]
    if not (hasattr(roi, "__len__") and len(roi) == 4):
        raise ValueError("roi must be a list or tuple with 4 elements: [x, y, w, h]")

    x, y, w, h = roi

    # detect relative (floats in [0,1]) vs absolute (ints or floats > 1)
    is_relative = all(isinstance(v, (float, int)) and 0.0 <= v <= 1.0 for v in (x, y, w, h))
    if is_relative:
        x_abs = int(round(x * w_img))
        y_abs = int(round(y * h_img))
        w_abs = int(round(w * w_img))
        h_abs = int(round(h * h_img))
    else:
        # treat as absolute pixels
        x_abs, y_abs, w_abs, h_abs = int(round(x)), int(round(y)), int(round(w)), int(round(h))

    # clip to image bounds
    x_abs = max(0, min(x_abs, w_img - 1))
    y_abs = max(0, min(y_abs, h_img - 1))
    w_abs = max(1, min(w_abs, w_img - x_abs))
    h_abs = max(1, min(h_abs, h_img - y_abs))

    return x_abs, y_abs, w_abs, h_abs

#define ROI for table extraction and show the table area
def extract_table(image, roi=None):
    """
    Define ROI for table extraction and show the table area.

    Parameters:
    - image: numpy array (already read with cv2.imread)
    - roi: None or list/tuple [x_rel, y_rel, w_rel, h_rel] (relative) OR absolute [x,y,w,h]

    Returns:
    - binary: the binary image of the extracted table region
    """
    if image is None:
        raise ValueError("image must be a valid numpy array (loaded with cv2.imread)")

    # default relative roi if None (matches previous default region roughly)
    if roi is None:
        roi = roi_dict["table_1"]

    # Convert to absolute pixel coordinates
    x, y, w, h = _roi_to_absolute(roi, image.shape)

    # Define the region of interest (ROI) for the table
    roi_img = image[y:y+h, x:x+w]

    # Convert the ROI to grayscale
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)


    # Show the extracted table area for debugging
    cv2.imshow("Extracted Table", binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return binary

#--------------------------------------
#table detection using contours that has as input the ROI of the image from extract_table function 
#--------------------------------------
def detect_table_contours(table_image):
    # Ensure binary input (0 or 255)
    if len(table_image.shape) == 3:
        gray = cv2.cvtColor(table_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = table_image.copy()

    # --- Step 1: Extract lines using MORPH_OPEN ---
    scale = max(table_image.shape[:2]) // 300  # Auto scaling of kernel size
    scale = max(scale, 1)  # Ensure scale is at least 1
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20 * scale, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5 * scale))

    horizontal_lines = cv2.morphologyEx(table_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=3)
    vertical_lines = cv2.morphologyEx(table_image, cv2.MORPH_OPEN, vertical_kernel, iterations=3)

    # --- Step 2: Combine lines ---
    table_mask = cv2.add(horizontal_lines, vertical_lines)

    # --- Step 3: Clean & close small gaps ---
    table_mask = cv2.dilate(table_mask, np.ones((3, 3), np.uint8), iterations=1)
    cv2.imshow("Horizontal Lines", horizontal_lines)
    cv2.waitKey(0)
    cv2.imshow("Vertical Lines", vertical_lines)
    cv2.waitKey(0)
    cv2.imshow("Table Mask", table_mask)
    cv2.waitKey(0)
    # --- Step 4: Find Contours ---
    contours, _ = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # --- Step 5: Filter and sort contours ---
    cell_contours = []
    img_h, img_w = table_image.shape[:2]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cell_contours.append((x, y, w, h))

    # Sort contours by position (top to bottom, left to right)
    cell_contours.sort(key=lambda rect: (rect[1], rect[0]))

    # --- Step 6: Show each detected cell ---
    """ the table have 2 lines. devide number of cells by 2.
    for every column in the second half show the cell, then apply the mathematical morphology : dilation to know the marked cell.
    print the biggest marked cell number. """

    max_non_zero_count = 0
    max_cell_index = -1
    max_cell_data = None

    # Iterate over the detected cell contours
    for i, (x, y, w, h) in enumerate(cell_contours):
        if i >= (len(cell_contours)//2)+1:  # only second half (2 rows)
            cell = table_image[y:y+h, x:x+w]
            # Apply dilation to detect marked cells
            dilated_cell = cv2.dilate(cell, np.ones((3, 3), np.uint8), iterations=3)
            #show each cell after dilation for debugging
            cv2.imshow(f"Cell {i}", dilated_cell)
            cv2.waitKey(0)
            # Keep track of non-zero pixel count
            non_zero_count = cv2.countNonZero(dilated_cell)
            #print the non-zero count for each cell for debugging
            print(f"Cell {i} non-zero pixel count after dilation: {non_zero_count}")
            # Keep track of the cell with the maximum non-zero count
            if non_zero_count > max_non_zero_count:
                max_non_zero_count = non_zero_count
                max_cell_index = i
                max_cell_data = (x, y, w, h, cell)

    if max_cell_index != -1:
        x, y, w, h, cell = max_cell_data
        cv2.imshow(f"Cell {max_cell_index} (Biggest)", cell)
        print(f"Showing cell {max_cell_index} (biggest) at (x={x}, y={y}, w={w}, h={h}) with {max_non_zero_count} non-zero pixels")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return cell_contours,max_cell_index


def detect_grade_marks(cell_contours,max_cell_index):
    marked_exam_grade = None
    decimal_grade_value = None
    # --- extract the Whole Number of the exam grade from 0 to 20 ---
    if len(cell_contours) > 20:
        # number of columns = half the contours
        num_cols = (len(cell_contours) // 2)+1

        # convert the cell index to column index (0 → first column)
        col_index = max_cell_index - num_cols

        # clamp the result from 0 to 20
        marked_exam_grade = max(0, min(20, col_index))
        print(f"Detected whole exam grade: {marked_exam_grade}")
        final_grade = marked_exam_grade

    # --- extract the decimal Number of the exam grade : 00, 25, 50, 75 ---
    else:
        num_cols = (len(cell_contours) // 2)+1 
        col_index = max_cell_index - num_cols  # 0,1,2,3 → columns

        # map 0→0, 1→25, 2→50, 3→75
        decimal_values = [0, 25, 50, 75]
        if 0 <= col_index < 4:
            decimal_grade_value = decimal_values[col_index]
            print(f"Detected decimal exam grade: {decimal_grade_value}")
        else:
            decimal_grade_value = 0  # fallback safety
        final_grade = decimal_grade_value / 100
    return final_grade


#extract qr code area from select_roi function
def extract_qr_code(image, roi):
    """
    Extract QR code area from image using roi list/tuple [x_rel, y_rel, w_rel, h_rel]
    or absolute [x,y,w,h].

    Parameters:
    - image: numpy array (already read with cv2.imread)
    - roi: list/tuple [x_rel, y_rel, w_rel, h_rel] or absolute

    Returns:
    - qr_roi: the cropped image region containing the QR code
    """
    if image is None:
        raise ValueError("image must be a valid numpy array (loaded with cv2.imread)")

    x, y, w, h = _roi_to_absolute(roi, image.shape)
    qr_roi = image[y:y+h, x:x+w]
    return qr_roi


# read and decode QR code from image
def decode_qr_code(image):
    decoded_objects = pyzbar.decode(image)
    for obj in decoded_objects:
        print("Type:", obj.type)
        print("Data:", obj.data.decode("utf-8"))
    if not decoded_objects:
        print("No QR code detected.")
    return decoded_objects


#check if the image is flipped and correct it
def check_image_flipped(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    # Show binary image for debugging
    #cv2.imshow("Binary Image", binary)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    height, width = binary.shape

    # Check top-left corner
    top_left = binary[0:height//10, 0:width//10]
    top_left_non_zero = cv2.countNonZero(top_left)

    # Check bottom-right corner
    bottom_right = binary[height - height//10:height, width - width//10:width]
    bottom_right_non_zero = cv2.countNonZero(bottom_right)

    print(f"Top-left non-zero pixels: {top_left_non_zero}")
    print(f"Bottom-right non-zero pixels: {bottom_right_non_zero}")

    if bottom_right_non_zero > top_left_non_zero:
        print("Image appears to be flipped.")
        # Rotate the image 180 degrees
        image = cv2.rotate(image, cv2.ROTATE_180)
        return image
    else:
        print("Image orientation is correct.")
        return image


def extract_text_with_ocr(image, roi):
    """
    Extract text from a specified ROI in the image using OCR.

    Parameters:
    - image: numpy array (already read with cv2.imread)
    - roi: list/tuple [x_rel, y_rel, w_rel, h_rel] or absolute [x,y,w,h]

    Returns:
    - extracted_text: the text extracted from the specified ROI
    """
    if image is None:
        raise ValueError("image must be a valid numpy array (loaded with cv2.imread)")

    x, y, w, h = _roi_to_absolute(roi, image.shape)
    text_roi = image[y:y+h, x:x+w]

    # Convert ROI to grayscale
    gray_roi = cv2.cvtColor(text_roi, cv2.COLOR_BGR2GRAY)
    #show the ROI for debugging
    cv2.imshow("Text ROI", gray_roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Preprocess the image for better OCR results
    gray_roi = cv2.threshold(gray_roi,225, 255, cv2.THRESH_BINARY )[1]
    #show the preprocessed ROI for debugging
    cv2.imshow("Preprocessed Text ROI", gray_roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Apply OCR to extract text
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.'  # OEM and PSM configuration for better accuracy
    extracted_text = pytesseract.image_to_string(gray_roi, config=custom_config)

    print("Extracted Text:", extracted_text.strip())
    return extracted_text.strip()


# Example usage in main function
if __name__ == "__main__":
    image_path = r"examens\anis121125_page-0006.jpg"  # Replace with your image path
    image = cv2.imread(image_path)
    image = cv2.resize(image, None, fx=0.55, fy=0.55)
    image = check_image_flipped(image)

    #x_rel, y_rel, w_rel, h_rel = select_roi(image)  # if you want to select ROI interactively (returns relative)
    
    #test OCR on note_exament area
    extracted_text = extract_text_with_ocr(image, roi_dict["note_exament"])
    

    
    ##extract marks from table 1 
    #table_image = extract_table(image, roi_dict["table_1"])
    #cell_contours,max_cell_index=detect_table_contours(table_image)
    #marked_grade=detect_grade_marks(cell_contours,max_cell_index)
#
    ##extract marks from table 2
    #table_image2 = extract_table(image, roi_dict["table_2"])
    #cell_contours2,max_cell_index2=detect_table_contours(table_image2)
    #decimal_grade=detect_grade_marks(cell_contours2,max_cell_index2)
#
    ##calculate final grade
    #final_grade=marked_grade+decimal_grade
    #print(f"Final detected grade: {final_grade:.2f}")
#
    ##extract and decode qr code of exam
    #qr_image = extract_qr_code(image, roi_dict["qr_code_etudiant"])
    #decode_qr_code(qr_image)
    #qr_image_etudiant = extract_qr_code(image, roi_dict["qr_code_etudiant"])
    #decode_qr_code(qr_image_etudiant)
    #note_image = extract_qr_code(image, roi_dict["note_exament"])
    #cv2.imshow("Note Exament", note_image)
    #cv2.waitKey(0)


