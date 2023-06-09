
from clize import run
from subprocess import call
from copy import deepcopy

base= {
    "gpus": 16,
    "save-frequency": 1,
    "zeroshot-frequency": 1,
    "dataset-type": "webdataset",
    "train-data": "datacomp_filelist",
    "dataset-resampled": True,
    "report-to": "tensorboard",
    "train-num-samples": 100000000,
    "warmup": 50,
    "batch-size": 64,
    "epochs": 1,
    "workers": 8,
    "model": "coca_encoder-mt5-xxl_decoder-scratch_vis-ViT-BigG-14",
    "logs": "logs/mt5",
    "seed": 0,
    "ddp-static-graph": True,
    "local-loss": True,
    "gather-with-grad": True,
    "lr": 0.0001,
    "log-every-n-steps": 1,
    "save-most-recent": True,
    "resume": "latest",
    "grad-checkpointing": True,
    "fsdp": True,
    "fsdp-limit-allgathers": True,
    "fsdp-init-on-cpu": True,
    "fsdp-layers-to-wrap": "ResidualAttentionBlock MT5Block LayerNorm HFTextEncoder VisionTransformer Transformer Embedding",
    "lock-text": True,
    "lock-text-unlocked-layers": 7,
    "lock-image": True,
    "lock-image-unlocked-groups": 13,
    "force-patch-dropout": 0.5,
    "precision": "amp_bfloat16"
}


def exp1():
    return base

def exp2():
    exp = deepcopy(base)
    exp['model'] = "coca_encoder-mt5-xxl_decoder-mt5-xl_vis-ViT-BigG-14"
    exp['lock-text-decoder'] = True
    exp['batch-size'] = 128
    return exp

exps = [exp1, exp2]

def main(name, *, per_node=4):
    params  = None
    for exp in exps:
        if exp.__name__ == name:
            params = exp()
            params['name'] = name
    
    if params is not None:
        nodes = params['gpus'] // per_node
        params = [f"--{k} {v}" if type(v) != bool else f"--{k}" for k, v in params.items() if k != "gpus"]
        params = " ".join(params)
        print(params)
        call(f"sbatch  -N {nodes} template.sbatch {params}", shell=True)


if __name__ == "__main__":
    run(main)   