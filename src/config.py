from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "water_potability.csv"
MODELS = ROOT / "models"; REPORTS = ROOT / "reports"
MODELS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)
SEED = 42
TARGET = "Potability"
FEATURES = ["ph","Hardness","Solids","Chloramines","Sulfate",
            "Conductivity","Organic_carbon","Trihalomethanes","Turbidity"]
ALL_FEATURES = FEATURES + ["Hardness_pH_Ratio", "Organic_Chloramine_Hazard"]