import os
import requests
import gdown

# Pls follow this format for adding datasets -> 'filename': 'download_url'
# This is meant for downloading raw data sets so we can process them using preprocessing scripts
# Make sure the google drive csv file is set to "anyone w the link can edit the file"
datasets = {
    "south_dakota.csv": "https://drive.google.com/uc?export=download&id=175FN8ZY-mnJySrWvN11vLFsRAn1d4rEv",
    # Add other datasets here
}

os.makedirs("../data/raw", exist_ok=True)

for filename, url in datasets.items():
    output_path = os.path.join("../data/raw", filename)
    if os.path.exists(output_path):
        print(f"{filename} already exists, skipping download.")
        continue

    print(f"downloading {filename}")
    try:
        gdown.download(url, output_path, quiet=False)
        # response = requests.get(url, stream=True)
        # response.raise_for_status()
        # with open(output_path, "wb") as f:
        #     for chunk in response.iter_content(chunk_size=8192):
        #         f.write(chunk)
        print(f"{filename} downloaded")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
