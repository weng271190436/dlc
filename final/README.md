# Benchmarking Deep Learning Architectures for predicting Readmission to the ICU and Describing patients-at-Risk

These codes were largely borrowed from https://github.com/sebbarb/time_aware_attention

I added 2 simple models: birnn and birnn_attention for my final project

How to run:
- change `mimic_dir` in `hyperparameters.py` to your MIMIC-III data path
- create `data_dir` and `log_dir` which are pointed to by `hyperparameters.py`
- select the model that you want to test in `hyperparameters.py` by uncommenting the appropriate `net_variant =` line
- `python train.py` to train. After training, the model will be stored in `./logdir/`
- `python test.py` to test.
