import gdown
import os
import requests

# Pls follow this format for adding datasets -> 'filename': 'download_url'
# This is meant for downloading raw data sets so we can process them using preprocessing scripts
# Make sure the google drive csv file is set to "anyone w the link can edit the file"
datasets = {
    # i.e. "review-hawaii.json.gz": "https://drive.google.com/uc?export=download&id=1JZmr0auFe-GB87Mj61vUUAlr5a4HMC56,
    # Add other datasets here
    "review-hawaii.json.gz": "https://drive.google.com/uc?export=download&id=1JZmr0auFe-GB87Mj61vUUAlr5a4HMC56",
    "meta-hawaii.json.gz": "https://drive.google.com/uc?export=download&id=1gyQhntzVWAG7bcIzkGekFNqvCdV3URKH",
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
