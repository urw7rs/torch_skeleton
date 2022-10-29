# Training & Testing models using ``torch_skeleton`` 

Examples using ``torch_skeleton`` to train models


Implemented models:

* SGN

Using [LightningCLI](https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli.html)
from [pytorch_lightning](https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction.html) ,
training configurations are set using yaml files.

## Install dependencies

```python
pip install torch_skeleton[exmaples]
```

### SGN

Training

```python
python3 exampels/sgn/main.py fit --config examples/sgn/configs/sgn_ntu60_xsub.yaml
```

Testing

```python
python3 exampels/sgn/main.py test --config examples/sgn/configs/sgn_ntu60_xsub.yaml --ckpt_path <path_to_ckpt>
```

change config file to train & test on different datasets
