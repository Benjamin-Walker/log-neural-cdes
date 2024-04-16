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
            # ignore_list.remove(run_config[0][3])
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
    include_time,
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
    # SEEDS = [2345, 3456, 4567, 5678, 6789]
    SEEDS = [1234]

    cfg_list = []

    for seed in SEEDS:
        cfg_list.append(
            [
                seed,
                data_dir,
                use_presplit,
                dataset_name,
                include_time,
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
        job_name=JOB_NAME + f"_{dataset_name}_no_duplicates",
        time=TIME,
        partition=PARTITION,
        gres=f"gpu:{GPUS}",
        constraint="gpu_mem:32GB",
        # mem="3000G",
        qos="priority",
        account="math-datasig",
    )


def run(cfg):
    create_dataset_model_and_train(*cfg)


if __name__ == "__main__":
    data_dir = WORKING_DIRECTORY + "/data"
    use_presplit = True

    model_names = ["log_ncde", "ncde", "nrde", "ssm", "lru"]
    num_steps = 100000
    batch_size = 32

    repeat_experiments = False

    if repeat_experiments:

        args = open("best_hyperparameters_final.txt", "r")
        experiments = args.read().split("\n")

        for experiment in experiments:
            experiment = experiment.split(" ")
            model_name = experiment[0]
            if model_name in model_names:
                if (
                    model_name == "log_ncde"
                    or model_name == "ncde"
                    or model_name == "nrde"
                ):
                    print_steps = 100
                else:
                    print_steps = 1000
                dataset_name = experiment[1]
                T = float(experiment[3])
                include_time = True if experiment[5] == "True" else False
                lr = float(experiment[9])
                if model_name == "log_ncde" or model_name == "nrde":
                    stepsize = int(float(experiment[11]))
                    logsig_depth = int(experiment[13])
                    idx = 4
                else:
                    stepsize = 1
                    logsig_depth = 1
                    idx = 0
                num_blocks = int(experiment[12 + idx])
                hidden_dim = int(experiment[15 + idx])
                vf_depth = int(experiment[18 + idx])
                vf_width = int(experiment[21 + idx])
                ssm_dim = int(experiment[24 + idx])
                ssm_blocks = int(experiment[27 + idx])
                dt0 = float(experiment[29 + idx])
                solver = diffrax.Heun()
                stepsize_controller = diffrax.ConstantStepSize()
                scale = float(experiment[36 + idx])
                lambd = float(experiment[38 + idx])
                model_args = {
                    "num_blocks": num_blocks,
                    "hidden_dim": hidden_dim,
                    "vf_depth": vf_depth,
                    "vf_width": vf_width,
                    "ssm_dim": ssm_dim,
                    "ssm_blocks": ssm_blocks,
                    "dt0": dt0,
                    "solver": solver,
                    "stepsize_controller": stepsize_controller,
                    "scale": scale,
                    "lambd": lambd,
                }
                run_experiments(
                    model_name,
                    dataset_name,
                    data_dir,
                    use_presplit,
                    include_time,
                    T,
                    num_steps,
                    print_steps,
                    lr,
                    lambda x: x,
                    stepsize,
                    batch_size,
                    logsig_depth,
                    model_args,
                )
    else:
        use_presplit = True

        dataset_names = [
            "EigenWorms",
            # "EthanolConcentration",
            # "Heartbeat",
            # "MotorImagery",
            # "SelfRegulationSCP1",
            # "SelfRegulationSCP2",
        ]

        lengths = {
            "EigenWorms": 17984,
            "EthanolConcentration": 1751,
            "Heartbeat": 405,
            "MotorImagery": 3000,
            "SelfRegulationSCP1": 896,
            "SelfRegulationSCP2": 1152,
        }

        model_names = [
            "log_ncde",
            "ncde",
            "nrde",
            "ssm",
            "lru",
        ]
        lr_scheduler = lambda x: x
        stepsize = 1
        logsig_depth = 1
        num_blocks = 1
        ssm_dim = 128
        ssm_blocks = 1
        lambd = 0.0
        T = 1

        for dataset_name in dataset_names:
            for model_name in model_names:
                for lr in [1e-3, 1e-4, 1e-5]:
                    for include_time in [True, False]:
                        for hidden_dim in [16, 64, 128]:
                            if (
                                model_name == "log_ncde"
                                or model_name == "nrde"
                                or model_name == "ncde"
                            ):
                                num_steps = 10000
                                print_steps = 100
                                for solvercontroller in [
                                    (diffrax.Heun(), diffrax.ConstantStepSize()),
                                ]:
                                    solver = solvercontroller[0]
                                    stepsize_controller = solvercontroller[1]
                                    for vf_dims in [
                                        (2, 32),
                                        (3, 64),
                                        (3, 128),
                                        (4, 128),
                                    ]:
                                        vf_depth = vf_dims[0]
                                        vf_width = vf_dims[1]
                                        length = lengths[dataset_name]
                                        if (
                                            model_name == "log_ncde"
                                            or model_name == "nrde"
                                        ):
                                            for depthstep in [
                                                (1, 1),
                                                (2, 2),
                                                (2, 4),
                                                (2, 8),
                                                (2, 12),
                                                (2, 16),
                                            ]:
                                                logsig_depth = depthstep[0]
                                                stepsize = depthstep[1]
                                                n_steps = max(
                                                    500, 1 + int(length / stepsize)
                                                )
                                                print(n_steps)
                                                dt0 = T / n_steps
                                                if model_name == "log_ncde":
                                                    scale = T * 1000
                                                    for lambd in [1e-3, 1e-6, 0]:
                                                        model_args = {
                                                            "num_blocks": num_blocks,
                                                            "hidden_dim": hidden_dim,
                                                            "vf_depth": vf_depth,
                                                            "vf_width": vf_width,
                                                            "ssm_dim": ssm_dim,
                                                            "ssm_blocks": ssm_blocks,
                                                            "dt0": dt0,
                                                            "solver": solver,
                                                            "stepsize_controller": stepsize_controller,
                                                            "scale": scale,
                                                            "lambd": lambd,
                                                        }
                                                        run_experiments(
                                                            model_name,
                                                            dataset_name,
                                                            data_dir,
                                                            use_presplit,
                                                            include_time,
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
                                                elif model_name == "nrde":
                                                    scale = T
                                                    model_args = {
                                                        "num_blocks": num_blocks,
                                                        "hidden_dim": hidden_dim,
                                                        "vf_depth": vf_depth,
                                                        "vf_width": vf_width,
                                                        "ssm_dim": ssm_dim,
                                                        "ssm_blocks": ssm_blocks,
                                                        "dt0": dt0,
                                                        "solver": solver,
                                                        "stepsize_controller": stepsize_controller,
                                                        "scale": scale,
                                                        "lambd": lambd,
                                                    }
                                                    run_experiments(
                                                        model_name,
                                                        dataset_name,
                                                        data_dir,
                                                        use_presplit,
                                                        include_time,
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
                                        elif model_name == "ncde":
                                            scale = T
                                            n_steps = max(500, 1 + length)
                                            print(n_steps)
                                            dt0 = T / n_steps
                                            model_args = {
                                                "num_blocks": num_blocks,
                                                "hidden_dim": hidden_dim,
                                                "vf_depth": vf_depth,
                                                "vf_width": vf_width,
                                                "ssm_dim": ssm_dim,
                                                "ssm_blocks": ssm_blocks,
                                                "dt0": dt0,
                                                "solver": solver,
                                                "stepsize_controller": stepsize_controller,
                                                "scale": scale,
                                                "lambd": lambd,
                                            }
                                            run_experiments(
                                                model_name,
                                                dataset_name,
                                                data_dir,
                                                use_presplit,
                                                include_time,
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
                            else:
                                num_steps = 100000
                                print_steps = 1000
                                for num_blocks in [2, 4, 6]:
                                    for ssm_dim in [16, 64, 256]:
                                        if model_name == "ssm":
                                            for ssm_blocks in [2, 4, 8]:
                                                model_args = {
                                                    "num_blocks": num_blocks,
                                                    "hidden_dim": hidden_dim,
                                                    "vf_depth": 1,
                                                    "vf_width": 1,
                                                    "ssm_dim": ssm_dim,
                                                    "ssm_blocks": ssm_blocks,
                                                    "dt0": 1,
                                                    "solver": diffrax.Heun(),
                                                    "stepsize_controller": diffrax.ConstantStepSize(),
                                                    "scale": 1,
                                                    "lambd": 0,
                                                }
                                                run_experiments(
                                                    model_name,
                                                    dataset_name,
                                                    data_dir,
                                                    use_presplit,
                                                    include_time,
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
                                                "vf_depth": 1,
                                                "vf_width": 1,
                                                "ssm_dim": ssm_dim,
                                                "ssm_blocks": ssm_blocks,
                                                "dt0": 1,
                                                "solver": diffrax.Heun(),
                                                "stepsize_controller": diffrax.ConstantStepSize(),
                                                "scale": 1,
                                                "lambd": 0,
                                            }
                                            run_experiments(
                                                model_name,
                                                dataset_name,
                                                data_dir,
                                                use_presplit,
                                                include_time,
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
