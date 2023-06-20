
from clize import run
from subprocess import call
from copy import deepcopy

base= {
    "gpus": 32,
    "save-frequency": 1,
    "zeroshot-frequency": 1,
    "dataset-type": "webdataset",
    "train-data": "datacomp_filelist",
    "dataset-resampled": True,
    "report-to": "tensorboard",
    "train-num-samples": 10_000_000,
    "warmup": 2000,
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
    "log-every-n-steps": 10,
    #"save-most-recent": True,
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
    "precision": "amp_bfloat16",
    "resume": "latest",
    # "fsdp-sharding-strategy": "hybrid",
}

def exp1():
    # base training
    exp = deepcopy(base)
    exp['model'] = "mt5-xxl-ViT-BigG-14"
    exp['pretrained'] = "pretrained/mt5-xxl-ViT-BigG-14.pt"
    exp['batch-size'] = 256
    exp["gpus"] = 32
    exp["lock-text-unlocked-layers"] = 6
    exp["lock-image-unlocked-groups"] = 6
    exp["train-num-samples"] =  10_000_000
    exp["epochs"] =  1
    exp["warmup"] =  100
    exp["lr"] =  0.0005
    return exp

def exp2():
    exp = exp1()
    exp["warmup"] =  500
    return exp

def exp3():
    # mitchell hypers
    exp = exp1()
    exp["warmup"] =  500
    exp["beta2"] = 0.95
    exp["grad-clip-norm"] = 1
    exp["fsdp-sharding-strategy"] = "full"
    return exp

def exp4():
    # trying a first coca model
    exp = exp2()
    exp['model'] = "coca_encoder-mt5-xxl_decoder-mt5-base_vis-ViT-BigG-14"
    exp['batch-size'] = 64
    exp['pretrained'] = "pretrained/coca_encoder-mt5-xxl_decoder-mt5-base_vis-ViT-BigG-14.pt"
    return exp

def exp5():
    # increase BS to maximum possible with OOM
    exp = exp4()
    exp['batch-size'] = 100
    return exp

def exp6():
    # trying to see the impact of locking the text decoder, objective is to make it faster
    # -> that does not really make it any faster anyway
    # also, locking decoder seems make contrastive loss increase abrubtly after few initial steps (but never the caption loss)
    # as if the two objectives are not compatible 
    exp = exp5()
    exp['lock-text-decoder'] = True
    return exp

def exp7():
    # trying warmup more exp6
    exp = exp6()
    exp["warmup"] = 1000
    return exp

def exp8():
    # warmup helps in exp7, but still explodes at later step, trying to lower LR
    # -> decrease than start to increase, less abruptly
    exp = exp6()
    exp["lr"] =  0.0001
    return exp

def exp9():
    # tring to parially lock text decoder layers
    # also no speed difference with full text decoder training
    # -> still increases after a step, but not abruptly
    exp = exp6()
    exp['lock-text-decoder-unlocked-layers'] = 6
    return exp

def exp10():
    # trying more warmup from exp9
    # -> does help but contrastive loss starts to increase again
    exp = exp9()
    exp["warmup"] = 1000
    return exp

def exp11():
    # trying a larger decoder
    # -> I get OOM
    exp = exp5()
    exp['pretrained'] = "pretrained/coca_encoder-mt5-xxl_decoder-mt5-large_vis-ViT-BigG-14.pt"
    exp['model'] = "coca_encoder-mt5-xxl_decoder-mt5-large_vis-ViT-BigG-14"
    return exp

def exp12():
    # decrease  BS from exp11 to not get OOM
    # works but decrease speed significanttly as we have a larger text decoder now
    # also, even if no text decoder locking, contrastive loss decrease then start to increase
    exp = exp11()
    exp['batch-size'] = 96
    return exp

def exp13():
    # trying mitchell hypers on exp12, to see whether they can help, could be a gradient explosion
    exp = exp12()
    exp["beta2"] = 0.95
    exp["grad-clip-norm"] = 1
    return exp

def exp14():
    exp = exp12()
    exp["gpus"] = 64
    exp["epochs"] = 2
    return exp

exps = [v for k, v in vars().items() if k.startswith("exp")]

def main(name, *, per_node=8):
    params  = None
    for exp in exps:
        if exp.__name__ == name:
            params = exp()
            params['name'] = name
            #params['remote-sync'] = "s3://s-laion/mt5clip/" + name
    
    if params is not None:
        nodes = params['gpus'] // per_node
        params = [f"--{k} {v}" if type(v) != bool else f"--{k}" for k, v in params.items() if k != "gpus"]
        params = " ".join(params)
        print(params)
        call(f"sbatch  -N {nodes} template.sbatch {params}", shell=True)


if __name__ == "__main__":
    run(main)
