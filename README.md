# AI-Assisted Writing Detection

An end-to-end machine-learning notebook for predicting whether a submission was AI-assisted from writing-behavior and stylometric metadata.

## What is included

- `AI_Writing_Detection_EndToEnd.ipynb` — exploratory analysis, leakage-safe preprocessing, SMOTE-based class balancing, model training, hyperparameter tuning, evaluation, feature importance, model export, and inference.
- `ai_writing_detection_dataset.csv` — the dataset used by the notebook.
- `requirements.txt` — Python dependencies.

The target column is `Is_AI_Assisted` (`1` for AI-assisted and `0` for human-written). The notebook evaluates Logistic Regression, SVM, and XGBoost models using accuracy, precision, recall, F1, ROC-AUC, and PR-AUC.

## Run locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook AI_Writing_Detection_EndToEnd.ipynb
```

Run the cells from top to bottom. The notebook expects `ai_writing_detection_dataset.csv` in the project root and writes the selected fitted pipeline as a `.joblib` file.

## Notes

This project is intended for research and educational use. Predictions should not be treated as definitive proof of authorship or misconduct; model performance depends on the dataset and the quality of the available behavioral features.
