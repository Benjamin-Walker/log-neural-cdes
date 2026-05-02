import json
from copy import deepcopy
from pathlib import Path

from run_experiment import run_experiments


DROP_PERCENTAGES = (0.3, 0.7, 0.95)
DATASET_NAMES = (
    "EigenWorms",
    "EthanolConcentration",
    "Heartbeat",
    "MotorImagery",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
)
MODEL_NAMES = ("ncde",)
BASE_CONFIG_ROOT = Path("experiment_configs/repeats")
GENERATED_CONFIG_ROOT = Path("experiment_configs/uea_drop_ncde_grid")


def _drop_folder_name(drop_percentage):
    return f"drop_{int(round(drop_percentage * 100)):02d}"


def write_configs():
    generated_roots = []
    for drop_percentage in DROP_PERCENTAGES:
        target_root = GENERATED_CONFIG_ROOT / _drop_folder_name(drop_percentage)
        generated_roots.append(target_root)
        for model_name in MODEL_NAMES:
            for dataset_name in DATASET_NAMES:
                source = BASE_CONFIG_ROOT / model_name / f"{dataset_name}.json"
                target = target_root / model_name / f"{dataset_name}.json"
                with open(source, "r", encoding="ascii") as f:
                    data = deepcopy(json.load(f))
                data["drop_percentage"] = drop_percentage
                target.parent.mkdir(parents=True, exist_ok=True)
                with open(target, "w", encoding="ascii") as f:
                    json.dump(data, f, indent=4)
                    f.write("\n")
    return generated_roots


def main():
    generated_roots = write_configs()
    for drop_percentage, config_root in zip(DROP_PERCENTAGES, generated_roots):
        print(f"Starting NCDE UEA drop sweep for drop={drop_percentage:.2f}")
        run_experiments(
            model_names=list(MODEL_NAMES),
            dataset_names=list(DATASET_NAMES),
            experiment_folder=str(config_root),
            pytorch_experiments=False,
        )


if __name__ == "__main__":
    main()
