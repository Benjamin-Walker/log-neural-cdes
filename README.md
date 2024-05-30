# Log-Neural-CDEs
Applying the Log-ODE method to improve the training and performance of Neural CDEs.

## Data

The data_dir folder contains the scripts for downloading data, preprocessing the data, 
generating path objects and creating dataloaders and datasets. Raw data should be 
downloaded into the data/raw folder. Processed data should be saved into the data/processed
in the following format: 
```
processed/{dataset_name}/data.pkl, 
processed/{dataset_name}/labels.pkl,
processed/{dataset_name}/original_idxs.pkl (if the dataset has original data splits)
```
where data.pkl and labels.pkl are jnp.arrays with shape (n_samples, n_timesteps, n_features) 
and (n_samples, n_classes) respectively. If the dataset had original_idxs then those should
be saved as a list of jnp.arrays with shape [(n_train,), (n_val,), (n_test,)].

The UEA dataset can be downloaded using the `download_data.py` script. The UEA data can be preprocessed by 
running the `process_uea.py` script.

## Models

The scrips in the models folder are used to define the various deep learning
models used in the experiments. In order to be integrated into the training, 
the `__call__` function of the model should only take one argument as input. In 
order to handle this, the dataloaders return the model's inputs as a list, 
which is unpacked within the model `__call__`. 

Currently, the models folder contains the following models:
- `RNN`: A simple recurrent neural network which can use any cell. Currently,
the available cells are `Linear`, `GRU`, `LSTM`, and `MLP`.
- `NeuralCDE`: A Neural CDE model, which uses a Hermite cubic spline 
with backward differences and Tsit5 solver.
- `NeuralRDE`: A Neural RDE model, which takes the log-signature intervals as
an argument.

## Training

`train.py` is the main script for training the models. It defines a run_fn 
which takes the model, dataloaders, num_steps, print_steps, learning rate, 
batch size, random key, and output dir as arguments. 

The best model, alongside statistics about the training process, are saved in
output dir.

## Requirements

```
conda create -n Log-NCDE python=3.10
conda activate Log-NCDE
conda install --file conda_requirements.txt -c conda-forge
/path/to/conda/envs/Log-NCDE/bin/pip install -r pip_requirements.txt
```

- python 3.10
- pre-commit 3.3.1
- sktime 0.17.2
- jaxlib 0.4.7
- jax 0.4.9
- tqdm 4.65.0
- equinox 0.10.3
- optax 0.1.5
- diffrax 0.3.1
- roughpy 0.0.1

If process_uea throws this error: No module named 'packaging'
Then run: pip install packaging

After installing the requirements, run `pre-commit install` to install the pre-commit hooks.
