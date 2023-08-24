import os
import pathlib
import shutil

import diffrax
import submitit

from train import create_dataset_model_and_train


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
            ignore_list.remove(run_config[0][3])
            ignore_list.append("results")
            ignore_list.append("logs_slurm")
            if os.path.exists(working_directory):
                shutil.rmtree(working_directory)
            shutil.copytree(
                ".", working_directory, ignore=shutil.ignore_patterns(*ignore_list)
            )
            os.chdir(working_directory)
        print(f"Running at {working_directory}")

    executor = submitit.SlurmExecutor(
        folder=HOME_DIRECTORY + "/logs_slurm",
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
    data_dir,
    use_presplit,
    T,
    num_steps,
    print_steps,
    lr,
    lr_scheduler,
    stepsize,
    batch_size,
    logsig_depth,
    model_args,
):
    SEEDS = [1234]

    cfg_list = []

    for seed in SEEDS:
        cfg_list.append(
            [
                seed,
                data_dir,
                use_presplit,
                dataset_name,
                T,
                model_name,
                stepsize,
                logsig_depth,
                model_args,
                num_steps,
                print_steps,
                lr,
                lr_scheduler,
                batch_size,
                WORKING_DIRECTORY + "/",
            ]
        )

    run_with_config(
        run,
        cfg_list,
        WORKING_DIRECTORY,
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
    create_dataset_model_and_train(*cfg)


if __name__ == "__main__":
    # Spoken Arabic Digits has nan values in training data
    data_dir = WORKING_DIRECTORY + "/data"
    use_presplit = True
    dataset_names = [
        "EigenWorms",
        "EthanolConcentration",
        "FaceDetection",
        "FingerMovements",
        "HandMovementDirection",
        "Handwriting",
        "Heartbeat",
        "InsectWingbeat",
        "JapaneseVowels",
        "Libras",
        "LSST",
        "MotorImagery",
        "NATOPS",
        "PEMS-SF",
        "PhonemeSpectra",
        "SelfRegulationSCP1",
        "SelfRegulationSCP2",
    ]

    lengths = {"Heartbeat": 405, "SelfRegulationSCP2": 1152, "MotorImagery": 3000}

    model_names = ["log_ncde"]

    num_steps = 10000
    print_steps = 100
    batch_size = 32
    lr = 1e-4
    lr_scheduler = lambda x: x
    T = 10
    dt0 = 0.1
    include_time = False
    solver = None
    stepsize_controller = None
    stepsize = 4
    logsig_depth = 2
    num_blocks = 1
    ssm_dim = 128
    ssm_blocks = 1
    hidden_dim = 64

    for dataset_name in dataset_names:
        for model_name in model_names:
            for stepsize in [8]:
                for include_time in [True, False]:
                    for T in [1]:
                        for dt0 in [T / 1200]:
                            solver = diffrax.Heun()
                            stepsize_controller = diffrax.ConstantStepSize()
                            model_args = {
                                "num_blocks": num_blocks,
                                "hidden_dim": hidden_dim,
                                "vf_depth": 2,
                                "vf_width": 32,
                                "ssm_dim": ssm_dim,
                                "ssm_blocks": ssm_blocks,
                                "dt0": dt0,
                                "include_time": include_time,
                                "solver": solver,
                                "stepsize_controller": stepsize_controller,
                            }
                            run_experiments(
                                model_name,
                                dataset_name,
                                data_dir,
                                use_presplit,
                                T,
                                num_steps,
                                print_steps,
                                lr,
                                lr_scheduler,
                                stepsize,
                                batch_size,
                                logsig_depth,
                                model_args,
                            )
