# Exam_verification-

This repository contains simple scripts to help verify and extract information
from exam attendance lists and similar scanned images. The core scripts use
OpenCV and pyzbar for image processing and QR/barcode reading.

**Setup & Quick Start**

- **Create a virtual environment (PowerShell)**:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

- **Install dependencies**:

```powershell
pip install -r requirements.txt
```

If you prefer pinned versions for reproducibility, edit `requirements.txt` and
specify exact versions (for example `opencv-python==4.7.0.72`).

**Note (pyzbar on Windows)**: If `pyzbar` raises errors about the zbar
backend, you may need to install a platform zbar binary. Usually `pip install
pyzbar` on Windows will work, but if it fails, try finding a compatible zbar
installer or use a Python wheel that bundles zbar.

**Files**
- `feuille_examen.py`: helper functions such as `select_roi`, `extract_table`,
	and `detect_table_contours` used by example scripts.
- `liste_emargement.py`: example script demonstrating loading an attendance
	list image and interactively selecting a region of interest (ROI).
- `requirements.txt`: project dependencies.

**Run the example**

```powershell
# From repository root
python liste_emargement.py
```

The script will open an image window to allow manual ROI selection (via
`select_roi` in `feuille_examen.py`). After selection the program prints the
pixel coordinates of the chosen ROI.

**Troubleshooting & Tips**
- If OpenCV GUI windows do not appear or are unresponsive, ensure you run the
	script in a desktop session (not headless) and install a compatible
	`opencv-python` wheel for your Python version.
- If images are not found, check the relative path `images\liste_emargement.jpg`
	or adjust the path in `liste_emargement.py`.

If you'd like, I can also:
- add a small CLI to pick one of multiple images
- add automated ROI extraction using `roi_dict`
- pin dependency versions for reproducible installs

Tell me which of these you'd like next.
