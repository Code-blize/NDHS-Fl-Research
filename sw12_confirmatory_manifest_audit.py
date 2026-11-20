
from pathlib import Path
import json
import hashlib

ROOT = Path(
    r"C:\Users\usher\Documents\NDHS-Fl-Research\NDHS_FL_Research"
)

PACKAGE = ROOT / "09_flower_implementation_v1_2_1"
OUTPUTS = PACKAGE / "outputs" / "confirmatory_v2_seeds_42"

EVALUATION_FILE = PACKAGE / "ndhs_flower" / "evaluation.py"
SERVER_FILE = PACKAGE / "ndhs_flower" / "server_app.py"

print("=" * 72)
print("SW-01: CONFIRMATORY MANIFEST AND EVALUATION SOURCE AUDIT")
print("=" * 72)

# --------------------------------------------------
# 1. Inspect local evaluation source
# --------------------------------------------------

for path in [EVALUATION_FILE, SERVER_FILE]:
    print(f"\nSOURCE: {path}")
    print("Exists:", path.exists())

    if path.exists():
        content = path.read_text(encoding="utf-8")
        digest = hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

        print("SHA256:", digest)

        if path.name == "evaluation.py":
            checks = {
                "sigmoid_probability_conversion":
                    "torch.sigmoid" in content,
                "threshold_0_5":
                    "prob >= 0.5" in content,
                "roc_auc_from_probabilities":
                    "roc_auc_score(y, prob)" in content,
                "average_precision_from_probabilities":
                    "average_precision_score(y, prob)" in content,
            }

            for name, passed in checks.items():
                print(f"{name}: {'PASS' if passed else 'NOT FOUND'}")

        if path.name == "server_app.py":
            print(
                "implementation_version_1_2_1:",
                '"implementation_version": "1.2.1"' in content
            )
            print(
                "test_archive_not_configured:",
                '"test-data"' not in content
            )

# --------------------------------------------------
# 2. Discover all run manifests
# --------------------------------------------------

manifests = sorted(OUTPUTS.rglob("run_manifest.json"))

print("\n" + "=" * 72)
print("MANIFEST INVENTORY")
print("=" * 72)

print("Output directory exists:", OUTPUTS.exists())
print("Manifest count:", len(manifests))

if len(manifests) != 40:
    print("WARNING: Expected exactly 40 manifests.")

records = []

for path in manifests:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        cfg = data.get("configuration", {})
        software = data.get("software", {})

        records.append({
            "folder": path.parent.name,
            "implementation": data.get("implementation_version"),
            "status": data.get("status"),
            "algorithm": cfg.get("algorithm"),
            "feature": cfg.get("feature_config"),
            "seed": cfg.get("seed"),
            "rounds": cfg.get("num_server_rounds"),
            "epochs": cfg.get("local_epochs"),
            "batch_size": cfg.get("batch_size"),
            "learning_rate": cfg.get("learning_rate"),
            "proximal_mu": cfg.get("proximal_mu"),
            "input_dim": cfg.get("input_dim"),
            "sklearn": software.get("scikit_learn"),
            "torch": software.get("torch"),
            "final_hash_present": bool(
                data.get("final_parameters_sha256")
            ),
            "validation_rows": data.get("validation_rows"),
        })

    except Exception as exc:
        print("MANIFEST READ ERROR:", path, type(exc).__name__, exc)

# --------------------------------------------------
# 3. Summarise run configuration
# --------------------------------------------------

print("\n" + "=" * 72)
print("RUN SUMMARY")
print("=" * 72)

for record in records:
    print(
        record["folder"],
        "|", record["implementation"],
        "|", record["status"],
        "|", record["algorithm"],
        "|", record["feature"],
        "| seed:", record["seed"],
        "| rounds:", record["rounds"],
        "| epochs:", record["epochs"],
        "| input:", record["input_dim"],
        "| final hash:", record["final_hash_present"]
    )

# --------------------------------------------------
# 4. Check expected configuration
# --------------------------------------------------

print("\n" + "=" * 72)
print("EXPECTED CONFIGURATION CHECKS")
print("=" * 72)

checks = {
    "40_manifests": len(records) == 40,
    "all_completed": all(r["status"] == "completed" for r in records),
    "all_v1_2_1": all(
        r["implementation"] == "1.2.1" for r in records
    ),
    "all_10_seeds": all(
        r["seed"] in range(42, 52) for r in records
    ),
    "all_10_rounds": all(r["rounds"] == 10 for r in records),
    "all_5_epochs": all(r["epochs"] == 5 for r in records),
    "all_batch_32": all(r["batch_size"] == 32 for r in records),
    "all_learning_rate_001": all(
        r["learning_rate"] == 0.001 for r in records
    ),
    "all_final_hashes_present": all(
        r["final_hash_present"] for r in records
    ),
    "baseline_dimension_15": all(
        r["input_dim"] == 15
        for r in records if r["feature"] == "baseline"
    ),
    "expanded_dimension_51": all(
        r["input_dim"] == 51
        for r in records if r["feature"] == "expanded"
    ),
}

for name, passed in checks.items():
    print(f"{name}: {'PASS' if passed else 'CHECK'}")

# --------------------------------------------------
# 5. Configuration combinations
# --------------------------------------------------

print("\n" + "=" * 72)
print("COMBINATION COUNTS")
print("=" * 72)

from collections import Counter

combinations = Counter(
    (r["feature"], r["algorithm"])
    for r in records
)

for combination, count in sorted(combinations.items()):
    print(combination, ":", count)

print("\nAUDIT COMPLETE")
print("No checkpoints loaded.")
print("No files modified.")
