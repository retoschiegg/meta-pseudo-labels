# Meta Pseudo Labels
Unofficial TF2 implementation of "Meta Pseudo Labels" (official [Paper](https://arxiv.org/abs/2003.10580) and [Code](https://github.com/google-research/google-research/tree/master/meta_pseudo_labels)).


## Results

<table>
    <thead>
        <tr>
            <th></th>
            <th colspan=2>CIFAR-10-4K</th>
        </tr>
        <tr>
            <th></th>
            <th>w/o finetune</th>
            <th>w/ finetune</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><b>Paper</b></td>
            <td align=center><a href="https://tensorboard.dev/experiment/i6sjrvdVQcanGLca8FV3LQ/#scalars" target="_blank">96.08%</a></td>
            <td align=center></td>
        </tr>
        <tr>
            <td><b>Implementation</b></td>
            <td align=center><a href="https://tensorboard.dev/experiment/5c8iVvpbSGS3MNBT21YlFA/#scalars" target="_blank">94.70%</a></td>
            <td align=center><a href="https://tensorboard.dev/experiment/19KuM2iORpSBLDoOqIbvvA/#scalars" target="_blank">94.71%</a></td>
        </tr>
    </tbody>
</table>


## Usage

### Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pre-commit install
```

### Run Scripts
```bash
export PYTHONPATH=src

# download dataset
dvc pull -r origin

# train & evaluate
python src/main.py --data-dir data/cifar10 --config-name cifar10 --model-dir workdir/training --mpl-epochs 10000 --mpl-batch-size 64 --finetune-epochs 1000 --finetune-batch-size 512

# only evaluate
python src/evaluate.py --data-dir data/cifar10 --config-name cifar10 --saved-model-dir [workdir/training/finetune/model, saved_models/mpl, saved_models/finetune]
```
