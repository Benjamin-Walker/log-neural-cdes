# Log-Neural-CDEs
Applying the Log-ODE method to improve the training and performance of Neural CDEs.

## Data

The data folder contains the scripts for downloading data, preprocessing the data, 
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

## Requirements

- python 3.10
- pre-commit 3.3.1
- sktime 0.17.2
- jaxlib 0.4.7
- jax 0.4.9
- tqdm 4.65.0
- equinox 0.10.3
- optax 0.1.5
- diffrax 0.3.1

If process_uea throws this error: No module named 'packaging'
Then run: pip install packaging