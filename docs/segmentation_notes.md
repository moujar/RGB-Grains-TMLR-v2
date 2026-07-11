# Project structure


This sub-project is meant to convert large .hyspex files into numerous samll .npy files (1 per grain) containing the RGB int16 input.

```
ROOT/
│
├── data/                       # Raw and processed datasets
├── gala/                       # External or experimental modules
│
├── grain_hs/                   # Core hyperspectral preprocessing package
│   ├── __init__.py             # Package initialization
│   ├── _path.py                # Path management utilities
│   ├── brightest_csv.py        # Extract brightest pixel data to CSV
│   ├── constants.py            # Global constants and configuration
│   ├── full_band_pipeline.py   # Full-band processing pipeline
│   ├── hdr_grains.py           # HDR grain processing
│   ├── reflectance.py          # Reflectance computation
│   ├── rgb_pipeline.py         # RGB processing pipeline
│   ├── segmentation.py         # Grain segmentation logic
│   ├── spectralon.py           # Spectralon calibration utilities
│
├── full_band_rgb.py            # Convert full-band images to RGB
├── hyspex_to_full_band.py      # Process raw .hyspex → full-band images
├── hyspex_to_rgb.py            # Process raw .hyspex → RGB images
│
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
```


# Setup (macOS, isolated venv)

```bash
cd "/path/to/[root_dir]"
cd $HOME
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```


### Smoke test

```bash
cd "/path/to/Phuoc's-preprocessing"
python -c "import grain_hs; from gala import morpho; import cv2, spectral; print('OK')"
python hyspex_to_rgb.py --help
python hyspex_to_full_band.py --help
python full_band_to_rgb.py --help
```

---

## Usage

**End-to-end**

```bash
python hyspex_to_grain_rgb.py path/to/image.hdr -o path/to/out/
```

**End-to-end but saving intermediate full-bands images**

```bash
python hyspex_to_full_band.py path/to/image.hdr -o path/to/full_npz/
python full_band_to_rgb.py --src path/to/full_npz/ --out path/to/rgb_npz/
```
## Brightest-Band CSV

Used for pipelines like `hyspex_to_rgb` or `hyspex_to_full_band`.

Generate the CSV:

```bash
python -m grain_hs.brightest_csv \
    --img-dir <path_to_images> \
    --out-dir <output_directory>
```

Then pass it to your pipeline:

```bash
--brightest-csv <output_directory>/brightest_bands.csv
```

---

## Tuning Parameters

You can adjust the following parameters for better preprocessing and segmentation:

- `--crop-idx-dim1` → Crop range along dimension 1  
- `--reflectance-trim` → Trim reflectance values  
- `--watershed-trim` → Control watershed segmentation sensitivity  
- `--area-min` / `--area-max` → Filter objects by size  
- `--solidity` → Shape filtering threshold  
- `--binary-thresh` → Threshold for binarization  
- `--segmentation-band` → Band used for segmentation  
- `--variety` → Grain type or dataset variant  
- `--start-index` → Starting index for processing  

---
