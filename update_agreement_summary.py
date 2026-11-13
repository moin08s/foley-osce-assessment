import json
import os
from train_cohort_model import load_cohort_dataset, validate_cohort_agreement
from src.model import FoleyStepClassifier

model = FoleyStepClassifier.load('models/foley_osce_mlp.npz')
X, y, stem_slices = load_cohort_dataset('dataset/features')
val_summary = validate_cohort_agreement(model, X, y, stem_slices, 'dataset/cohort_ground_truth.json')

summary_file = r"C:\Medical\reports\teacher_model_agreement_summary.json"
os.makedirs(os.path.dirname(summary_file), exist_ok=True)
with open(summary_file, "w", encoding="utf-8") as f:
    json.dump(val_summary, f, indent=2)

print("\nSaved updated cohort validation report to:", summary_file)
print(f"Agreement: {val_summary['matched_candidates']}/{val_summary['total_candidates']} ({val_summary['agreement_pct']:.1f}%)")
