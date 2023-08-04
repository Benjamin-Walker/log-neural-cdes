import os
import pathlib
import shutil

import optax
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
            ignore_list.remove(run_config[0][2])
            ignore_list.append("results")
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
    dataset_names = [
        "EigenWorms",
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

    model_names = ["rnn_lstm", "lru", "ssm"]

    num_steps = 100000
    print_steps = 1000
    batch_size = 32
    lr = 1e-3
    lr_scheduler = lambda x: x
    T = 10
    dt0 = 0.1
    include_time = False
    solver = None
    stepsize_controller = None
    stepsize = 4
    logsig_depth = 2

    for dataset_name in dataset_names:
        for model_name in model_names:
            for lr in [1e-3, 3e-4, 1e-4]:
                for lr_scheduler in [
                    lambda x: x,
                    lambda x: optax.warmup_cosine_decay_schedule(
                        init_value=1e-6,
                        decay_steps=num_steps,
                        peak_value=x,
                        warmup_steps=int(num_steps / 100),
                        end_value=1e-6,
                    ),
                ]:
                    for hidden_dim in [8, 32, 128]:
                        if model_name == "ssm" or model_name == "lru":
                            for num_blocks in [2, 4, 6]:
                                if model_name == "ssm":
                                    for ssm_dim in [16, 64, 256]:
                                        for ssm_blocks in [
                                            ssm_dim // 2,
                                            ssm_dim // 4,
                                            ssm_dim // 8,
                                        ]:
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
                                elif model_name == "lru":
                                    model_args = {
                                        "num_blocks": num_blocks,
                                        "hidden_dim": hidden_dim,
                                        "vf_depth": 2,
                                        "vf_width": 32,
                                        "ssm_dim": 32,
                                        "ssm_blocks": 2,
                                        "dt0": dt0,
                                        "include_time": include_time,
                                        "solver": solver,
                                        "stepsize_controller": stepsize_controller,
                                    }
                                    run_experiments(
                                        model_name,
                                        dataset_name,
                                        data_dir,
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

                        elif model_name == "rnn_lstm":
                            model_args = {
                                "num_blocks": 6,
                                "hidden_dim": hidden_dim,
                                "vf_depth": 2,
                                "vf_width": 32,
                                "ssm_dim": 32,
                                "ssm_blocks": 2,
                                "dt0": dt0,
                                "include_time": include_time,
                                "solver": solver,
                                "stepsize_controller": stepsize_controller,
                            }
                            run_experiments(
                                model_name,
                                dataset_name,
                                data_dir,
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
