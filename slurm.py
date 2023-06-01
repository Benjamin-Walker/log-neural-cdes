import os
import pathlib
import shutil

import submitit

from train import run_training


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

JOB_NAME = "Experiments"
HOME_DIRECTORY = "/home/shug6778/Log-Neural-CDEs"
WORKING_DIRECTORY = "/data/math-datasig/shug6778/Log-Neural-CDEs"
PARALLEL = True

PARTITION = "short"
TIME = "12:00:00"
GPUS = 1


def run_with_config(
    run_fn,
    run_config,
    directory,
    results_dir,
    parallel=False,
    **cluster_config,
):

    copy_and_change_dir = True
    if copy_and_change_dir:
        print("Let's use slurm!")
        working_directory = pathlib.Path(directory) / cluster_config["job_name"]
        if not working_directory.is_dir():
            os.chdir(HOME_DIRECTORY)
            ignore_list = os.listdir(WORKING_DIRECTORY + "/data/processed/UEA/")
            ignore_list.remove(run_config[0][1])
            ignore_list.append("results")
            shutil.copytree(
                ".", working_directory, ignore=shutil.ignore_patterns(*ignore_list)
            )
        os.chdir(working_directory)
        if os.path.isdir(results_dir):
            raise ValueError(f"Warning: Output directory {results_dir} already exists")
        # os.makedirs(results_dir, exist_ok=True)
        print(f"Running at {working_directory}")

    executor = submitit.SlurmExecutor(
        folder=os.path.join(WORKING_DIRECTORY, results_dir)
    )
    executor.update_parameters(**cluster_config)
    if parallel:
        print(run_config)
        jobs = executor.map_array(run_fn, run_config)
        print(f"job_ids: {jobs}")
    else:
        job = executor.submit(run_fn, run_config)
        print(f"job_id: {job}")


def run_experiments(
    model_name,
    dataset_name,
    num_steps,
    print_steps,
    lr,
    stepsize,
    batch_size,
    logsig_depth,
    model_args,
    seed,
):

    SEEDS = [1234]

    output_parent_dir = HOME_DIRECTORY + "/outputs/" + model_name + "/" + dataset_name
    output_dir = f"nsteps_{num_steps}_lr_{lr}"
    if model_name == "log_ncde" or model_name == "nrde":
        output_dir += f"_stepsize_{stepsize}_logsigdepth_{logsig_depth}"
    for k, v in model_args.items():
        output_dir += f"_{k}_{v}"
    output_dir += f"_seed_{seed}"

    cfg_list = []

    for seed in SEEDS:
        cfg_list.append(
            [
                seed,
                dataset_name,
                model_name,
                output_parent_dir,
                output_dir,
                stepsize,
                logsig_depth,
                model_args,
                num_steps,
                print_steps,
                lr,
                batch_size,
                True,
            ]
        )

    run_with_config(
        run,
        cfg_list,
        WORKING_DIRECTORY,
        output_parent_dir + "/" + output_dir,
        parallel=PARALLEL,
        array_parallelism=500,
        job_name=JOB_NAME + f"_{dataset_name}",
        time=TIME,
        partition=PARTITION,
        gres=f"gpu:{GPUS}",
        # constraint="gpu_mem:20GB",
        qos="priority",
        account="math-datasig",
    )


def run(cfg):
    run_training(*cfg)


if __name__ == "__main__":
    seed = 1234
    num_steps = 100000
    print_steps = 1000
    batch_size = 32
    lr = 1e-3
    # Spoken Arabic Digits has nan values in training data
    dataset_names = [
        # "EigenWorms",
        "EthanolConcentration",
        "FaceDetection",
        "FingerMovements",
        "HandMovementDirection",
        "Handwriting",
        "Heartbeat",
        "Libras",
        "LSST",
        "InsectWingbeat",
        "MotorImagery",
        "NATOPS",
        "PhonemeSpectra",
        "RacketSports",
        "SelfRegulationSCP1",
        "SelfRegulationSCP2",
        "UWaveGestureLibrary",
    ]
    dataset_name = "Libras"
    stepsize = 4
    logsig_depth = 2
    model_name = "log_ncde"

    model_args = {"hidden_dim": 20, "vf_depth": 3, "vf_width": 8}

    run_experiments(
        model_name,
        dataset_name,
        num_steps,
        print_steps,
        lr,
        stepsize,
        batch_size,
        logsig_depth,
        model_args,
        seed,
    )
