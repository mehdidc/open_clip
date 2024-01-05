from clip_benchmark.datasets.builder import build_dataset, get_dataset_collate_fn, get_dataset_default_task, dataset_collection, get_dataset_collection_from_file
from clip_benchmark.metrics import image_caption_selection, zeroshot_classification, zeroshot_retrieval, linear_probe, captioning, image_caption_selection
from clip_benchmark.model_collection import get_model_collection_from_file, model_collection
from clip_benchmark.models import load_clip

import generative_classifier
import generative_image_caption_selection
import webdataset as wds
import torch
import open_clip
from clize import run
import json
from glob import glob
import os

def world_info_from_env():
    # from openclip
    local_rank = 0
    for v in ('LOCAL_RANK', 'MPI_LOCALRANKID', 'SLURM_LOCALID', 'OMPI_COMM_WORLD_LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ('RANK', 'PMI_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ('WORLD_SIZE', 'PMI_SIZE', 'SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break
    return local_rank, global_rank, world_size

def main(*, model="cap_ViT-B-32", pretrained=None, dataset_name="cifar10", dataset_root="", batch_size=32, device="cuda", output_file_format="{dataset}_{model}_{pretrained}.json", skip_existing=False, normalize=False,  normalizer="microsoft/phi-2", normalize_type="add", template='{c}', distributed=False):
    if distributed:
        local_rank, global_rank, world_size = world_info_from_env()
        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['RANK'] = str(global_rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        torch.distributed.init_process_group(backend="nccl", init_method="env://", rank=global_rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        rank_zero = global_rank == 0
    else:
        local_rank, global_rank, world_size = 0, 0, 1
        rank_zero = True
    
    if not pretrained:
        pretraineds = [None]
    else:
        pretraineds = glob(pretrained)
    print(pretraineds)
    for  pretrained in pretraineds:
        tokenizer = open_clip.get_tokenizer(model)
        output_file = output_file_format.format(dataset=dataset_name.replace('/', '_'), model=model, pretrained=os.path.basename(pretrained) if pretrained else '')
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
            task="" if "crepe" in dataset_name else "zeroshot_classification",
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
                collate_fn=get_dataset_collate_fn(dataset_name),
                sampler=torch.utils.data.DistributedSampler(dataset, num_replicas=world_size, rank=global_rank) if distributed else None
            )
        kw = {}
        if 'sugar_crepe' in dataset_name:
            if 'cap' in model  or 'dec' in model:
                results = generative_image_caption_selection.evaluate(
                    model_, dataloader, tokenizer,  device, normalize=normalize, normalizer=normalizer, normalize_type=normalize_type
                )
            else:
                results = image_caption_selection.evaluate(
                    model_, dataloader, tokenizer,  device,
                )
        else:
            classnames = dataset.classes
            #templates = dataset.templates
            #templates = ['{c}']
            templates = [template]
            if 'cap' in model  or 'dec' in model:
                results = generative_classifier.evaluate(
                    model_,
                    dataloader,
                    tokenizer,
                    classnames, 
                    templates,
                    device,
                    normalize=normalize, normalizer=normalizer, normalize_type=normalize_type,
                    distributed=distributed,
                )            
            else:
                results = zeroshot_classification.evaluate(
                    model_,
                    dataloader,
                    tokenizer,
                    classnames, 
                    templates,
                    device,
            )          
        results['normalize'] = normalize
        results['normalizer'] = normalizer
        results['normalize_type'] = normalize_type

        if rank_zero:
            print(f"Saving to {output_file}")
            with open(output_file, "w") as f:
                f.write(json.dumps(results))
            print(results)


if __name__ == "__main__":
    run(main)
