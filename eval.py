from clip_benchmark.datasets.builder import build_dataset, get_dataset_collate_fn, get_dataset_default_task, dataset_collection, get_dataset_collection_from_file
from clip_benchmark.metrics import image_caption_selection, zeroshot_classification, zeroshot_retrieval, linear_probe, captioning, image_caption_selection
from clip_benchmark.model_collection import get_model_collection_from_file, model_collection
from clip_benchmark.models import load_clip

import generative_classifier
import webdataset as wds
import torch
import open_clip
from clize import run
import json
from glob import glob
import os

def main(*, model="cap_ViT-B-32", pretrained=None, dataset_name="cifar10", dataset_root="", batch_size=32, device="cuda", output_file_format="{dataset}_{model}_{pretrained}.json", skip_existing=False):
    tokenizer = open_clip.get_tokenizer(model)
    
    if not pretrained:
        pretraineds = [None]
    else:
        pretraineds = glob(pretrained)
    print(pretraineds)
    for  pretrained in pretraineds:
        output_file = output_file_format.format(dataset=dataset_name, model=model, pretrained=os.path.basename(pretrained) if pretrained else '')
        if os.path.exists(output_file) and skip_existing:
            print(f"Skipping {output_file}")
            print(open(output_file).read())
            continue
        model_, _, transform = open_clip.create_model_and_transforms(model, pretrained)
        model_.to(device)
        dataset = build_dataset(
            dataset_name=dataset_name,
            root=dataset_root,
            transform=transform,
            split="test",
            download=True,
        )
        if type(dataset) == wds.WebDataset:
            dataloader = torch.utils.data.DataLoader(
                dataset.batched(batch_size), batch_size=None,
                shuffle=False, num_workers=4,
            )
        else:
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size,
                shuffle=False, num_workers=4,
            )
        classnames = dataset.classes
        #templates = dataset.templates
        templates = ['{c}']

        zeroshot_classification
        if 'cap' in model  or 'dec' in model:
            eval_func = generative_classifier
        else:
            eval_func = zeroshot_classification
        results = eval_func.evaluate(
            model_,
            dataloader,
            tokenizer,
            classnames, 
            templates,
            device,
        )
        print(f"Saving to {output_file}")
        with open(output_file, "w") as f:
            f.write(json.dumps(results))
        print(results)


if __name__ == "__main__":
    run(main)
