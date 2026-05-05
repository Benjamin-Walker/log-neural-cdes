import json
import os
import pathlib
import shutil

import diffrax
import submitit

from train import create_dataset_model_and_train


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

JOB_NAME = "NCDEDropUEA"
HOME_DIRECTORY = "/home/shug6778/Log-Neural-CDEs"
WORKING_DIRECTORY = "/data/math-datasig/shug6778/Log-Neural-CDEs"
PARTITION = "short"
TIME = "12:00:00"
GPUS = 1
ARRAY_PARALLELISM = 90
DROP_PERCENTAGES = (0.3, 0.7, 0.95)
DATASET_NAMES = (
    "EigenWorms",
    "EthanolConcentration",
    "Heartbeat",
    "MotorImagery",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
)


def run_with_config(
    run_fn,
    run_config,
    directory,
    parallel=False,
    **cluster_config,
):
    working_directory = pathlib.Path(directory) / cluster_config["job_name"]
    if not working_directory.is_dir():
        os.chdir(HOME_DIRECTORY)
        ignore_list = os.listdir(WORKING_DIRECTORY + "/data/processed/UEA/")
        ignore_list += [
            "results",
            "logs_slurm",
            ".git",
            "__pycache__",
            ".pytest_cache",
            "wandb",
            "outputs",
            "outputs_*",
            "data",
        ]
        if os.path.exists(working_directory):
            shutil.rmtree(working_directory)
        shutil.copytree(
            ".",
            working_directory,
            ignore=shutil.ignore_patterns(*ignore_list),
        )
    os.chdir(working_directory)
    print(f"Running at {working_directory}")

    executor = submitit.SlurmExecutor(folder=HOME_DIRECTORY + "/logs_slurm")
    executor.update_parameters(**cluster_config)
    if parallel:
        jobs = executor.map_array(run_fn, run_config)
        print(f"job_ids: {jobs}")
    else:
        for cfg in run_config:
            job = executor.submit(run_fn, cfg)
            print(f"job_id: {job}")


def build_model_args(config):
    return {
        "num_blocks": None,
        "block_size": None,
        "hidden_dim": int(config["hidden_dim"]),
        "vf_depth": int(config["vf_depth"]),
        "vf_width": int(config["vf_width"]),
        "ssm_dim": None,
        "ssm_blocks": None,
        "dt0": float(config["dt0"]),
        "solver": diffrax.Heun(),
        "stepsize_controller": diffrax.ConstantStepSize(),
        "scale": config.get("scale", 1.0),
        "lambd": None,
        "parallel_steps": None,
        "walsh_hadamard": None,
        "diagonal_dense": None,
        "sparsity": None,
        "piecewise_abelian": config.get("piecewise_abelian", True),
        "rank": None,
    }


def build_run_configs():
    cfg_list = []
    data_dir = WORKING_DIRECTORY + "/data"
    config_root = pathlib.Path("experiment_configs/repeats/ncde")

    for drop_percentage in DROP_PERCENTAGES:
        for dataset_name in DATASET_NAMES:
            with open(config_root / f"{dataset_name}.json", "r", encoding="ascii") as f:
                data = json.load(f)

            lr_scheduler = eval(data["lr_scheduler"])
            model_args = build_model_args(data)

            for seed in data["seeds"]:
                # Repeat configs often leave output_parent_dir empty, but the
                # current train entrypoint prepends it twice. Keep it explicitly
                # relative here so Slurm jobs write inside the copied worktree
                # rather than trying to create /outputs on the cluster root.
                output_parent_dir = data["output_parent_dir"] or "."
                cfg_list.append(
                    [
                        seed,
                        data_dir,
                        data["use_presplit"],
                        dataset_name,
                        1,
                        data["metric"],
                        data["time"].lower() == "true",
                        data["T"],
                        drop_percentage,
                        data.get("drop_mode", "same"),
                        data.get("path_drop_window_mode", "fixed"),
                        "ncde",
                        1,
                        1,
                        model_args,
                        data["num_steps"],
                        data["print_steps"],
                        data["early_stopping_steps"],
                        float(data["lr"]),
                        lr_scheduler,
                        data["batch_size"],
                        output_parent_dir,
                    ]
                )
    return cfg_list


def run(cfg):
    create_dataset_model_and_train(*cfg)


def main():
    exclude_nodes = ",".join(
        f"htc-g{n:03d}"
        for n in [
            53,
            54,
            55,
            56,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            83,
            84,
        ]
    )

    run_with_config(
        run,
        build_run_configs(),
        WORKING_DIRECTORY,
        parallel=True,
        array_parallelism=ARRAY_PARALLELISM,
        job_name=JOB_NAME,
        time=TIME,
        partition=PARTITION,
        gres=f"gpu:{GPUS}",
        exclude=exclude_nodes,
        qos="priority",
        account="math-datasig",
    )


if __name__ == "__main__":
    main()
