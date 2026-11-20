
from pathlib import Path
import joblib

project_root = Path(__file__).resolve().parent

model_path = project_root / "models" / "olist_low_satisfaction_model.joblib"

model = joblib.load(model_path)

print("MODEL TYPE:")
print(type(model))

print("\nMODEL PIPELINE STEPS:")
if hasattr(model, "steps"):
    for name, step in model.steps:
        print(name, ":", type(step))
else:
    print("No pipeline steps found.")

print("\nMODEL CLASSES:")
print(model.classes_ if hasattr(model, "classes_") else "Not directly available")

print("\nFEATURE NAMES:")
if hasattr(model, "feature_names_in_"):
    print(model.feature_names_in_)
else:
    print("Not directly available")
