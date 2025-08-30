import gdown
import os
import requests
import gdown

# Pls follow this format for adding datasets -> 'filename': 'download_url'
# This is meant for downloading raw data sets so we can process them using preprocessing scripts
# Make sure the google drive csv file is set to "anyone w the link can edit the file"
datasets = {
    "review-south_dakota.json.gz": "https://drive.google.com/uc?export=download&id=1l7fz0nc7U4N-gmLaKmZxuJS_wnaUIEQS",
    "meta-south_dakota.json.gz": "https://drive.google.com/uc?export=download&id=1xKfdWKLM_Lbr28PKIAH-75bvGIa1VmLv",
    # Add other datasets here
    }

RAW_DIR = "../data/raw"

def download_raw_data():
    os.makedirs("../data/raw", exist_ok=True)

    for filename, url in datasets.items():
        output_path = os.path.join("../data/raw", filename)
        if os.path.exists(output_path):
            print(f"{filename} already exists, skipping download.")
            continue

        print(f"downloading {filename}")
        try:
            gdown.download(url, output_path, quiet=False)
            print(f"{filename} downloaded")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
