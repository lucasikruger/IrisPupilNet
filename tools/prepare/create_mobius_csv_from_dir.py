"""
--------------------------
OVERVIEW
--------------------------
Create a MOBIUS CSV from a directory of images.
The script scans a specified directory for mask images and their corresponding
original images, extracts metadata from the filenames, and compiles this
information into a structured CSV file.

--------------------------
COMMAND LINE ARGUMENTS
--------------------------
--input_dir   : Path to the MOBIUS dataset directory. Default is "/media/agot-lkruger/X9 Pro/facu/facu/tesis/MOBIUS"
--output_csv  : Path to the output CSV file. Default is "mobius_output/mobius_dataset.csv"

--------------------------
HOW TO USE
--------------------------
1. Ensure you have the required directory structure and files as per the MOBIUS dataset.
2. Run the script from the command line:
    python create_mobius_csv_from_dir.py --input_dir /path/to/mobius --output_csv /path/to/output.csv
3. The output CSV will be created at the specified location.

--------------------------
FILENAME PATTERN
--------------------------

The MOBIUS dataset filenames follow a specific pattern:
    <ID>_<phone_code><light_code>_<eye_code><gaze_code>_<number>[.ext]
Where:
- <ID>       : natural number (e.g., 123)
- <phone_code>  : natural number mapping to a phone model (1, 2, 3, …)
- <light_code>    : one of {'i','n','p'} → {'indoor','natural','poor'}
- <eye_code>      : one of {'L','R'} → {'left_eye','right_eye'}
- <gaze_code>     : one of {'l','r','s','u'} → {'left','right','straight','up'}
- <number>   : natural number (e.g., 42)

--------------------------
MOBIUS DIRECTORY STRUCTURE
--------------------------

The MOBIUS dir structure is assumed to be:
<MOBIUS_DIR>/
    data.csv                # Original data CSV
    images/                 # Original images
        {ID}/
            <ID>_<phone_code><light_code>_<eye_code><gaze_code>_<number>.jpg
    Masks/                  # Mask images
        {ID}/
            <ID>_<phone_code><light_code>_<eye_code><gaze_code>_<number>.png

--------------------------
OUTPUT CSV STRUCTURE
--------------------------

The output CSV will contain columns for:
    ID
    gender
    age
    colour
    glasses/lenses
    dioptres (l)
    dioptres (r)
    cylinders (l)
    cylinders (r)
    smoker
    eye conditions
    drops
    allergies
    rel_mask_path
    rel_image_path
    phone_code
    phone_model
    light_code
    light
    eye_code
    eye
    gaze_code
    gaze
    number
"""

from pathlib import Path
import pandas as pd
import argparse
import re
from typing import Dict, Any, Union

# ============================
# CONSTANTS AND DEFAULTS
# ============================

DEFAULT_MOBIUS_DIR = str("/media/agot-lkruger/X9 Pro/facu/facu/tesis/MOBIUS")
DEFAULT_OUTPUT_DIR = str("/home/agot-lkruger/tesis/IrisPupilNet/dataset/mobius_output")
DEFAULT_MASKS_SUBDIR_NAME = "Masks"
DEFAULT_IMAGES_SUBDIR_NAME = "images"
DEFAULT_DATA_CSV_NAME = "data.csv"

# ============================
# MAIN FUNCTIONALITY
# ============================

def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)

    # Check if input directory exists
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist or is not a directory.")

    # Check for the csv 
    data_csv_path = input_dir / DEFAULT_DATA_CSV_NAME
    if not data_csv_path.exists() or not data_csv_path.is_file():
        raise FileNotFoundError(f"{DEFAULT_DATA_CSV_NAME} not found in {input_dir}.")

    # Read image paths from the csv
    data_df = pd.read_csv(data_csv_path)
    
    # Check for mask images directory
    masks_dir = input_dir / DEFAULT_MASKS_SUBDIR_NAME
    if not masks_dir.exists() or not masks_dir.is_dir():
        raise FileNotFoundError(f"Masks subdirectory not found in {input_dir}.")

    mask_image_paths = list(masks_dir.rglob("*.png"))  
    if not mask_image_paths:
        raise FileNotFoundError(f"No mask images found in directory {masks_dir}.")
    

    # Create a DataFrame for mask image paths
    # The mask stem is in the format: <ID>_<phone><light>_<Eye><gaze>_<number>
    mask_data = []
    dataset_folder = input_dir.name or input_dir.as_posix()

    for mask_path in mask_image_paths:

        # Skip "bad" images
        if "bad" in mask_path.name.lower():
            continue

        # Check if the image of the mask exists
        image_path = input_dir / DEFAULT_IMAGES_SUBDIR_NAME / mask_path.relative_to(masks_dir).with_suffix(".jpg")

        if not image_path.exists() or not image_path.is_file():
            raise FileNotFoundError(f"Corresponding image file not:\n{image_path}\nfor mask:\n{mask_path}")
        
        filename_data = parse_mobius_filename(mask_path)  # Validate filename
        
        # Append relative path and parsed data
        rel_mask = (Path(dataset_folder) / mask_path.relative_to(input_dir)).as_posix()
        rel_image = (Path(dataset_folder) / image_path.relative_to(input_dir)).as_posix()

        mask_data.append({
            "rel_mask_path": rel_mask,
            "rel_image_path": rel_image,
            **filename_data
        })
    mask_df = pd.DataFrame(mask_data)

    # Merge with the original data DataFrame using ID, mantain all columns
    merged_df = pd.merge(data_df, mask_df, on="ID", how="inner")
    if merged_df.empty:
        raise ValueError("No matching entries found between data CSV and mask images based on 'id'.")
    
    # Save to output CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_csv, index=False)
    print(f"Successfully created Mobius CSV at {output_csv}")

# =============================
# FUNCTIONS
# =============================


def parse_mobius_filename(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Parse a filename of the MOBIUS form:
        <ID>_<phone_code><light_code>_<eye_code><gaze_code>_<number>[.ext]

    The function extracts all components and returns a dictionary with
    normalized, human-readable fields (e.g., full phone model name, expanded
    gaze/light labels), as well as the original codes.

    Pattern details
    ---------------
    • <ID>       : natural number (e.g., 123)
    • <phone_code>  : natural number mapping to a phone model (1, 2, 3, …)
    • <light_code>    : one of {'i','n','p'} → {'indoor','natural','poor'}
    • <eye_code>      : one of {'L','R'} → {'left_eye','right_eye'}
    • <gaze_code>     : one of {'l','r','s','u'} → {'left','right','straight','up'}
    • <number>   : natural number (e.g., 42)
    • The file extension (if any) is ignored.

    Assumed phone code mapping
    ------------------------
    1 → "Sony Xperia Z5 Compact"
    2 → "Apple iPhone 6s"
    3 → "Xiaomi Pocophone F1"

    Args
    ----
    path : str | pathlib.Path
        Path to the file (or just the name). Only the stem (filename without
        extension) is parsed.

    Returns
    -------
    Dict[str, Any]
        A dictionary with the following keys:
        - 'id'          : int         # parsed <ID>
        - 'phone_code'    : int         # parsed <phone_code>
        - 'phone_model' : str         # mapped model name or "UNKNOWN"
        - 'light_code'  : str         # one of {'i','n','p'}
        - 'light'       : str         # 'indoor' | 'natural' | 'poor' | 'unknown'
        - 'eye_code'    : str         # 'L' | 'R'
        - 'eye'         : str         # 'left_eye' | 'right_eye' | 'unknown'
        - 'gaze_code'   : str         # 'l' | 'r' | 's' | 'u'
        - 'gaze'        : str         # 'left' | 'right' | 'straight' | 'up' | 'unknown'
        - 'number'      : int         # parsed <number>
        - 'stem'        : str         # original stem parsed

    Raises
    ------
    ValueError
        If the filename stem does not match the expected pattern.

    Examples
    --------
    >>> parse_stem_to_row("123_1i_Ls_42.png")
    {
      'id': 123, 'phone_code': 1, 'phone_model': 'Sony Xperia Z5 Compact',
      'light_code': 'i', 'light': 'indoor',
      'eye_code': 'L', 'eye': 'left_eye',
      'gaze_code': 's', 'gaze': 'straight',
      'number': 42, 'stem': '123_1i_Ls_42'
    }
    """
    # --- Local mappings ---
    phone_models: Dict[int, str] = {
        1: "Sony Xperia Z5 Compact",
        2: "Apple iPhone 6s",
        3: "Xiaomi Pocophone F1",
    }
    gaze_map: Dict[str, str] = {"l": "left", "r": "right", "s": "straight", "u": "up"}
    light_map: Dict[str, str] = {"i": "indoor", "n": "natural", "p": "poor"}
    eye_map: Dict[str, str] = {"L": "left_eye", "R": "right_eye"}

    # Compile the strict pattern
    pattern = re.compile(
        r"""
        ^(?P<id>\d+)_                 # <ID>
        (?P<phone_code>\d+)(?P<light_code>[inp])_  # <phone_code><light_code>
        (?P<eye_code>[LR])(?P<gaze_code>[lrsu])_      # <eye_code><gaze_code>
        (?P<number>\d+)$               # <number>
        """,
        re.VERBOSE,
    )

    stem = Path(path).stem
    m = pattern.match(stem)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: '{path}'")

    # Extract & normalize
    id_ = int(m.group("id"))
    phone_code = int(m.group("phone_code"))
    light_code = m.group("light_code")
    eye_code = m.group("eye_code")
    gaze_code = m.group("gaze_code")
    number = int(m.group("number"))

    result: Dict[str, Any] = {
        "ID": id_,
        "phone_code": phone_code,
        "phone_model": phone_models.get(phone_code, "UNKNOWN"),
        "light_code": light_code,
        "light": light_map.get(light_code, "unknown"),
        "eye_code": eye_code,
        "eye": eye_map.get(eye_code, "unknown"),
        "gaze_code": gaze_code,
        "gaze": gaze_map.get(gaze_code, "unknown"),
        "number": number,
        "stem": stem,
    }
    return result

# ============================
# ARGUMENT PARSING
# ============================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create Mobius CSV from directory of images."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=DEFAULT_MOBIUS_DIR,
        help="Path to the input directory containing images.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=str(Path(DEFAULT_OUTPUT_DIR) / "mobius_dataset.csv"),
        help="Path to the output CSV file.",
    )
    return parser.parse_args()

# ============================
# ENTRY POINT
# ============================
if __name__ == "__main__":
    main()
