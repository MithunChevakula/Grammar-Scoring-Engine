# Grammar Scoring Engine

An audio-based regression baseline for the SHL spoken grammar scoring competition.

## Contents

- `grammar_scoring_engine.ipynb` - documented, executed notebook with preprocessing, evaluation, visualizations, and submission generation.
- `audio_model.py` - reusable feature extraction and model training script.
- `requirements.txt` - Python dependencies.
- `Dataset_Final/submission.csv` - generated predictions for the 216 test clips.

## Results

Five-fold out-of-fold RMSE is approximately 0.750 and Pearson correlation is approximately 0.797. The final submission contains continuous predictions clipped to the valid 0-5 range.

The original WAV dataset is not included in this repository because it is approximately 1.7 GB. Place the competition's `Dataset_Final` directory next to the scripts before running them.

## Run

```powershell
python -m pip install -r requirements.txt
python audio_model.py --cv
```

Open the notebook in VS Code or Jupyter and run all cells to reproduce the report and submission.
