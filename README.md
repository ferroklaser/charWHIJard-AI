# charWHIJard-AI

## Project Overview
Online reviews shape public perception of businesses and locations, but many are irrelevant, misleading, or low-quality.  
This project addresses the challenge of **assessing the quality and relevancy of Google location reviews** by leveraging **LLMs**.

Our approach:
 - Use **ChatGPT (GPT-4o)** to assign relevancy scores, treating its outputs as *pseudo ground truth*.  
 - Use **Qwen3-8B** for prompt engineering and evaluation.  
 - Compare Qwen’s performance against ChatGPT to evaluate the model.  
 - Refine prompts or improve labelling strategies based on the results.  

 ---

## Getting started
1. Clone this repository:
```
git clone https://github.com/ferroklaser/charWHIJard-AI.git
cd charWHIJard-AI
```

2. Create and activate a virtual environment. (Optional but recommended):
```
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate     # On Windows
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Add your API keys for OpenAI to a .env file:
```
OPEN_API_KEY=your_key
```

## How to Reproduce Results
1. Ensure your dataset is in a .json.gz file and upload it to google drive.

2. Set the access to "Anyone on the internet with the link can view". Retrieve the url link, for example: "https://drive.google.com/file/d/fileId/view?usp=drive_link". Then, format it in the form "https://drive.google.com/uc?export=download&id=fileId".

3. Place the link inside the `datasets` dictionary in scripts/download_raw_data.py

4. Run the downloading script to download the datasets:
```
cd scripts/
python download_raw_data.py
```

5. Run the preprocessing script to clean and filter relevant columns:
```
python process_data.py
```

6. Run the labeling script for ChatGPT outputs:
``` 
python baseline_auto_labeling.py
```

7. Run the labeling script for Qwen outputs:
```
python full_pipeline.py
```

8. Evaluate Qwen outputs against ChatGPT's baseline:
```
python evaluate_tests.py
```

The evaluation results will be printed in the terminal.

