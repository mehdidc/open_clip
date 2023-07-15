import os
from clize import run
from subprocess import call
from copy import deepcopy

base= {
    "gpus": 1024,
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
    "pretrained": "pretrained/coca_encoder-mt5-xxl_decoder-scratch_vis-ViT-BigG-14.pt",
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
    "precision": "amp_bfloat16",
}

def exp1():
    exp = deepcopy(base)
    exp['model'] = "coca_encoder-mt5-xxl_decoder-scratch_vis-ViT-BigG-14"
    exp['pretrained'] = "pretrained/coca_encoder-mt5-xxl_decoder-scratch_vis-ViT-BigG-14.pt"
    exp['batch-size'] = 64
    exp["gpus"] = 128
    exp["lock-text-unlocked-layers"] = 6
    exp["lock-image-unlocked-groups"] = 6
    exp["train-num-samples"] =  10_000_000
    exp["epochs"] =  1  
    exp["warmup"] =  500
    exp["lr"] =  0.0005
    return exp

def exp2():
    exp = exp1()
    exp["lr"] =  0.001
    return exp

def exp3():
    exp = exp1()
    exp["lr"] =  0.0001
    return exp

def exp4():
    exp = exp1()
    exp['model'] = "coca_encoder-mt5-xxl_decoder-mt5-base_vis-ViT-BigG-14"
    exp['pretrained'] = "pretrained/coca_encoder-mt5-xxl_decoder-mt5-base_vis-ViT-BigG-14.pt"
    return exp

def exp5():
    exp = exp4()
    exp["lr"] =  0.0002
    return exp

def exp6():
    exp = exp1()
    exp["train-num-samples"] =  20_000_000
    return exp

def exp7():
    exp = exp1()
    exp['model'] = "coca_encoder-mt5-xxl_decoder-mt5-base_vis-ViT-BigG-14"
    exp['pretrained'] = "pretrained/coca_encoder-mt5-xxl_decoder-mt5-base_vis-ViT-BigG-14.pt"
    exp["train-num-samples"] =  20_000_000
    return exp

def exp8():
    exps = []
    for li, lt in ( (0, 6), (0, 12), (7, 6), (13, 12) ):
        exp = exp1()
        exp["epochs"] = 100
        exp["lock-text-unlocked-layers"] = lt
        exp["lock-image-unlocked-groups"] = li
        exp["lr"] =  0.0005
        name = f"li-{li}_lt-{lt}"
        exp["name"] = name
        exps.append(exp)
    return exps

def exp9():
    exps = []
    for li, lt in ( (0, 6), (0, 12), (7, 6), (13, 12) ):
        exp = exp1()
        exp["epochs"] = 100
        exp["lock-text-unlocked-layers"] = lt
        exp["lock-image-unlocked-groups"] = li
        exp['model'] = "coca_encoder-mt5-xxl_decoder-mt5-base_vis-ViT-BigG-14"
        exp['pretrained'] = "pretrained/coca_encoder-mt5-xxl_decoder-mt5-base_vis-ViT-BigG-14.pt"
        exp["lr"] =  0.0005
        name = f"li-{li}_lt-{lt}"
        exp["name"] = name
        exps.append(exp)
    return exps

def exp10():
    exp = exp1()
    exp["lock-text-unlocked-layers"] = 6
    exp["lock-image-unlocked-groups"] = 7
    exp["lr"] =  0.0005
    exp["gpus"] = 512 
    exp["warmup"] = 2000 
    exp["train-num-samples"] =  50_000_000
    exp["epochs"] = 20
    return exp

def exp11():
    exp = exp10()
    exp["gpus"] = 96
    exp["train-num-samples"] =  100000
    exp['batch-size'] = 32
    exp['dataset-type'] = 'synthetic'
    exp['workers'] = 0
    exp['fsdp-sharded-state-dict'] = True
    return exp

def exp12():
    exp = exp1()
    exp["lock-text-unlocked-layers"] = 6
    exp["lock-image-unlocked-groups"] = 7
    exp["lr"] =  0.0005
    exp["gpus"] = 512
    exp["warmup"] = 2000 
    exp["train-num-samples"] =  50_000_000
    exp["epochs"] = 20
    exp['fsdp-sharded-state-dict'] = True
    exp['fsdp-sharded-state-dict-type'] = 'sharded'
    exp['wd'] = 0.
    return exp

def exp13():
    exp = exp12()
    exp["warmup"] = 5000 
    return exp

def exp13_eval():
    exp = exp13()
    exp["fsdp-only-save-full-checkpoint"] = True
    exp["time_minutes"] = 15
    exp['name'] = 'exp13'
    return exp

def exp14():
    exp = exp13()
    exp['gpus'] = 512
    exp["time_minutes"] = 144 * 2 # 2 epochs
    exp['model'] = "coca_encoder-mt5-large_decoder-scratch_vis-ViT-BigG-14"
    exp['pretrained'] = "pretrained/coca_encoder-mt5-large_decoder-scratch_vis-ViT-BigG-14.pt"
    exp['lock-text'] = False
    exp['lock-image'] = True
    exp["lock-image-unlocked-groups"] = 0
    exp["lock-text-unlocked-layers"] = 0
    exp['batch-size'] = 96
    exp['warmup'] = 5000
    exp['force-patch-dropout'] = 0
    exp['fsdp-sharded-state-dict'] = True
    exp["train-num-samples"] =  50_000_000
    exp['fsdp-sharding-strategy'] = 'full'
    exp["epochs"] = 20
    return exp

def exp14_eval():
    exp = exp14()
    exp["fsdp-only-save-full-checkpoint"] = True
    exp["time_minutes"] = 15
    exp['name'] = 'exp14'
    return exp


exps = [v for k, v in vars().items() if k.startswith("exp")]

def main(name, *, resume='', per_node=4):
    all_params  = None
    for exp in exps:
        if exp.__name__ == name:
            params = exp()
            if type(params) == list:
                all_params = [p for p in params]
                for p in all_params:
                    p["logs"] = os.path.join(p["logs"], name)
                    os.makedirs(p["logs"], exist_ok=True)
            else:
                if 'name' not in params:
                    params['name'] = name
                all_params = [params]
            break
    
    if all_params is not None:
        for params in all_params:
            nodes = params['gpus'] // per_node
            time_minutes = params.get("time_minutes")
            if resume:
                params['resume'] = resume
            params = [f"--{k} {v}" if type(v) != bool else (f"--{k}" if v else "") for k, v in params.items() if k not in ("gpus", "time_minutes")]
            params = " ".join(params)
            print(params)
            duration = f"-t {time_minutes}" if time_minutes else ""
            call(f"sbatch  -N {nodes} {duration} template.sbatch {params}", shell=True)


if __name__ == "__main__":
    run(main)
