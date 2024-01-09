from glob import glob
from subprocess import call
from clize import run
import os
import re

gpus = os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",")

def main(*, params="", root=".", batch_size=64, dataset="imagenet1k", dataset_root="/p/scratch/ccstdl/cherti1/clip_benchmark_datasets/imagenet1k", regexp="", amp=True, normalize=False, normalizer="microsoft/phi-2", normalize_type="add", output_file_format="{dataset}_{model}_{pretrained}.json", template='{c}', distributed=False):
    # get all .pt files recursively from root
    files = glob(os.path.join(root, "**", "*.pt"), recursive=True) 
    if regexp:
        files = [f for f in files if re.search(regexp, f)]
    cmds = []
    files = sorted(files, key=lambda x: os.path.getmtime(x))
    for i, f in enumerate(files):
        params_file = os.path.join(os.path.dirname(f), "..", "params.txt") if not params else params
        model_name = None
        path = os.path.dirname(params_file)
        if not os.path.exists(params_file):
            continue
        for line in open(params_file).readlines():
           
            try:
                k, v = line.split(":")
            except Exception as e:
                continue
            k = k.strip()   
            v = v.strip()
            if k == "model":
                model_name = v
                break
        if model_name is None:
            continue
        if "latest" in f:
            continue
        #python eval.py --model $model --dataset-root /p/scratch/ccstdl/cherti1/clip_benchmark_datasets/$ds --dataset-name=$ds --pretrained "$f/checkpoints/*.pt" --output-file-format "$f/{dataset}_{model}_{pretrained}.json" --skip-existing

        amp_s = "--no_amp" if (amp==False) else ""
        normalize_s = "--normalize" if (normalize==True) else ""
        cmd = f"python eval.py --pretrained {f} --model {model_name} --batch-size {batch_size} --dataset-name {dataset} --dataset-root {dataset_root} --output-file-format '{path}/{output_file_format}' --skip-existing {normalize_s} --normalizer {normalizer} --normalize-type {normalize_type} {amp_s} --template '{template}' {'--distributed' if distributed else ''}"
        cmds.append(cmd)
    
    if distributed:
        cmd_all = ";".join(cmds)
        call(cmd_all, shell=True)
    else:
        for i in range(0, len(cmds), len(gpus)):
            cmd_batch = cmds[i:i+len(gpus)]
            # execute in parallel cmd_batch each in a different gpu
            for j, cmd in enumerate(cmd_batch):
                cmd_batch[j] = f"CUDA_VISIBLE_DEVICES={gpus[j]} " + cmd
            cmd_batch = " & ".join(cmd_batch)
            print(cmd_batch)
            call(cmd_batch, shell=True)

if __name__ == "__main__":
    run(main)
