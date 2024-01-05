import torch
import torch.nn.functional as F

import logging
from contextlib import suppress

import torch
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.metrics import classification_report, balanced_accuracy_score

@torch.no_grad()
def zero_shot_classifier(model, tokenizer, classnames, templates, device, amp=True):
    autocast = torch.cuda.amp.autocast if amp else suppress
    classes = []
    classes_raw = []
    for classname in tqdm(classnames):
        if type(templates) == dict:
            # class-specific prompts (e.g., CuPL https://arxiv.org/abs/2209.03320)
            texts = templates[classname]
        elif type(templates) == list:
            # generic prompts tht are specialized for each class by replacing {c} with the class name
            texts = [template.format(c=classname) for template in templates]
        else:
            raise ValueError("templates must be a list or a dict")
        #texts = [f'a {classname}']
        #texts = [f'a picture of a {classname}']
        #texts = [classname]
        #texts = [f'{classname} picture']
        #texts = [classname]
        #print(texts)
        classes_raw.append(texts)
        texts = tokenizer(texts)  # tokenize
        classes.append(texts)
    # nb_classes, nb_templates, nb_tokens
    templates_prefix = [t.replace('{c}', '').replace('.', '') for t in templates]
    return torch.stack(classes).to(device), tokenizer(templates_prefix).to(device), classes_raw

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """
    Compute top-k accuracy

    output: torch.Tensor
        shape (N, C) where N is the number of examples, C the number of classes.
        these are the logits.
    
    target: torch.Tensor
        shape (N,) where N is the number of examples. Groundtruth class id of each example.
    
    topk: tuple
        which topk to compute, e.g., topk=(1,5) will compute top-1 and top-5 accuracies
    
    Returns
    -------
    
    list of top-k accuracies in the same order as `topk`
    """
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    n = len(target)
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) / n for k in topk]


def run_classification(model, classifier, dataloader, device, amp=True, partial_mask=False, normalize=False, normalize_type="add", normalizer=None, distributed=False):
    """
    Run zero-shot classifcation

    model: torch.nn.Module
        CLIP-like model with `encode_image` and `encode_text`
    
    classifier: torch.Tensor
        obtained from the function `zero_shot_classifier`
    
    dataloader: torch.utils.data.Dataloader 
    
    Returns
    -------
    (pred, true)  where
        - pred (N, C) are the logits
        - true (N,) are the actual classes
    """
    autocast = torch.cuda.amp.autocast if amp else suppress
    if normalize:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_id = normalizer
        #lm_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", flash_attn=True, flash_rotary=True, fused_dense=True, device_map=device, trust_remote_code=True)
        lm_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map=device, trust_remote_code=True)

        lm_tokenizer =  AutoTokenizer.from_pretrained(model_id)
        lm_tokenizer.pad_token = lm_tokenizer.eos_token
    pred = []
    true = []
    nb = 0
    class_per_batch = 100
    templates_per_batch = 1

    classifier, templates, classifier_raw = classifier

    nb_classes, nb_templates, _ = classifier.shape
    #print(classifier.shape, templates.shape)
    #templates = templates[0:1]
    #classifier = classifier[:, 0:1]

    max_text_len = (classifier==0).float().argmax(dim=2).max()
    classifier = classifier[:, :, 0:max_text_len]
    templates = templates[:, 0:max_text_len]
    if normalize:
        prompts = [p for ps in classifier_raw for p in ps]
        tokenized_classifier = lm_tokenizer.batch_encode_plus(prompts, padding=True, return_tensors="pt", max_length=77, pad_to_max_length=True).input_ids
        tokenized_classifier = tokenized_classifier.to(device)
        bs = 10
        priors_all = []
        for i in range(0, tokenized_classifier.shape[0], bs):
            output = lm_model(tokenized_classifier[i:i+bs, 0:-1])
            target = tokenized_classifier[i:i+bs, 1:]
            priors = -F.cross_entropy(output.logits.transpose(1, 2), target, reduction="none", ignore_index=lm_tokenizer.pad_token_id).sum(dim=1).data.cpu()
            priors_all.append(priors)
        priors = torch.cat(priors_all, 0)
        priors = priors.view(1, nb_classes, nb_templates).to(device)

    if partial_mask:
        templates[templates==49407] = 0
        for t in range(len(templates)):
            c = classifier[:, t]
            tp = templates[t]
            c[c==tp] = 0

    ##
    lengths = (classifier!=0).sum(dim=2)
    ##
    with torch.no_grad():
        for images, target in tqdm(dataloader):
            images = images.to(device)
            target = target.to(device)
            with autocast():
                if partial_mask:
                    image_embs = model.encode_image(images)
                    logits_batch = []
                    for j in range(0, len(templates), templates_per_batch):

                        #I, LI, DI = image_embs.shape
                        #tpls = templates[j:j+templates_per_batch, :]
                        #T, LT = tpls.shape
                        #ims = image_embs.view(I, 1, LI, DI).repeat(1, T, 1, 1).view(I*T, LI, DI)
                        #tpls = tpls.view(1, T, LT).repeat(I, 1, 1).view(I*T, LT)
                        #raw = model.predict(image_embs=ims, text=tpls)
                        raw = model.predict(
                            image_embs=image_embs, 
                            text=templates[j:j+templates_per_batch, :].repeat(images.shape[0], 1)
                        )
                        logits_mini_batch = []
                        for i in range(0, nb_classes, class_per_batch):
                            texts = classifier[i:i+class_per_batch, j:j+templates_per_batch, :].contiguous()
                            nc, nt = texts.shape[0], texts.shape[1]
                            texts = texts.view(nc*nt, texts.shape[2])
                            scores = model.score(raw, texts)
                            nims, _ = scores.shape
                            scores = scores.view(nims, nc, nt)
                            #print(scores.mean(2), priors.mean(2))
                            if normalize:
                                if normalize_type == "add":
                                    scores = scores + priors[:, i:i+class_per_batch, j:j+templates_per_batch]
                                elif normalize_type == "sub":
                                    scores = scores - priors[:, i:i+class_per_batch, j:j+templates_per_batch]
                            logits_mini_batch.append(scores.float().cpu())
                        logits_batch.append(torch.cat(logits_mini_batch, 1))
                    logits = torch.cat(logits_batch, 2)
                    logits = logits.sum(2)
                else:
                    # full mask

                    if model.causal_mask is False:
                        logits_batch = []
                        image_embs = model.encode_image(images)
                        raw = model.predict(image_embs=image_embs, max_text_len=max_text_len)
                        #weights = model.score(raw, templates)
                        #weights = (weights.view(len(raw), 1, len(templates)))
                        #weights = weights.softmax(dim=2)
                        #print(weights)
                        #weights = torch.exp(weights)
                        #print(weights[0, 0])
                        for i in range(0, nb_classes, class_per_batch):
                            texts = classifier[i:i+class_per_batch, :, :]
                            nc = texts.shape[0]
                            texts = texts.view(nc*nb_templates, texts.shape[2])
                            scores = model.score(raw, texts)
                            nims, _ = scores.shape
                            scores = scores.view(nims, nc, nb_templates)
                            if normalize:
                                if normalize_type == "add":
                                    scores = scores + priors[:, i:i+class_per_batch]
                                elif normalize_type == "sub":
                                    scores = scores - priors[:, i:i+class_per_batch]
                            scores = scores.mean(2)
                            #scores = (scores*weights).sum(2)
                            logits_batch.append(scores.float().cpu())
                        logits = torch.cat(logits_batch, 1)
                    else:
                        logits_batch = []
                        with torch.no_grad():
                            image_embs = model.encode_image(images)
                        for i in range(0, nb_classes, class_per_batch):
                            texts = classifier[i:i+class_per_batch, :, :]
                            nc = texts.shape[0]
                            texts = texts.view(nc*nb_templates, texts.shape[2])
                            nim, lim, dim = image_embs.shape
                            ntext, ltext = texts.shape
                            image_embs_p = image_embs.view(nim, 1, lim, dim).repeat(1, ntext, 1, 1).view(nim*ntext, lim, dim)
                            texts_p = texts.view(1, ntext, ltext).repeat(nim, 1, 1).view(nim*ntext, ltext)
                            input_text = texts_p[:, 0:-1]
                            out_text = texts_p[:, 1:]
                            #print(image_embs.shape, texts.shape)
                            logits = model.predict(image_embs=image_embs_p, text=input_text)
                            scores = model.score_aligned(logits, out_text)
                            scores = scores.view(nim, nc, nb_templates)
                            if normalize:
                                if normalize_type == "add":
                                    scores = scores + priors[:, i:i+class_per_batch]
                                elif normalize_type == "sub":
                                    scores = scores - priors[:, i:i+class_per_batch]
                            scores = scores.mean(2)
                            logits_batch.append(scores.float().cpu())
                        logits = torch.cat(logits_batch, 1)
            true.append(target.cpu())
            pred.append(logits.float().cpu())
            #print(logits.argmax(dim=1))
            #print((target.cpu()==logits.cpu().argmax(dim=1)).float().mean())
    pred = torch.cat(pred)
    true = torch.cat(true)
    if distributed:
        # barrier
        torch.distributed.barrier()
        total = torch.tensor([len(pred)]).to(device)
        torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
        total = total.item()
        print("TOTAL", total)
        pred = all_gather_nd(pred.to(device)).cpu()
        true = all_gather_nd(true.to(device)).cpu()
    return pred, true


def all_gather_nd(tensor):
    """
    Gathers tensor arrays of different lengths in a list.
    The length dimension is 0. This supports any number of extra dimensions in the tensors.
    All the other dimensions should be equal between the tensors.

    Args:
        tensor (Tensor): Tensor to be broadcast from current process.

    Returns:
        (Tensor): output list of tensors that can be of different sizes
    """
    world_size = torch.distributed.get_world_size()
    local_size = torch.tensor(tensor.size(), device=tensor.device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    torch.distributed.all_gather(all_sizes, local_size)

    max_length = max(size[0] for size in all_sizes)

    length_diff = max_length.item() - local_size[0].item()
    if length_diff:
        pad_size = (length_diff, *tensor.size()[1:])
        padding = torch.zeros(pad_size, device=tensor.device, dtype=tensor.dtype)
        tensor = torch.cat((tensor, padding))

    all_tensors_padded = [torch.zeros_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(all_tensors_padded, tensor)
    all_tensors = []
    for tensor_, size in zip(all_tensors_padded, all_sizes):
        all_tensors.append(tensor_[:size[0]])
    return torch.cat(all_tensors)

def average_precision_per_class(scores, targets):
    """
    Compute average precision  for each class
    this metric is used for multi-label classification
    see explanations here https://fangdahan.medium.com/calculate-mean-average-precision-map-for-multi-label-classification-b082679d31be
    Code is adapted from https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py, thanks to the authors of `tnt`.

    Parameters
    ----------

    scores: torch.Tensor
        logits, of shape (N,C) where N is the number of examples, C the number of classes
    
    targets: torch.Tensor
        one-hot vectors of groundtruth targets (N, C), where N is the number of examples, C is the
        number of classes
    
    Returns
    -------

    torch.Tensor of shape (C,) of avereage precision for each class, where C is     
    the number of classes.
    
    """
    ap = torch.zeros(scores.size(1))
    rg = torch.arange(1, scores.size(0) + 1).float()
    # compute average precision for each class
    for k in range(scores.size(1)):
        # sort scores
        scores_k = scores[:, k]
        targets_k = targets[:, k]
        _, sortind = torch.sort(scores_k, 0, True)
        truth = targets_k[sortind]
        tp = truth.float().cumsum(0)
        # compute precision curve
        precision = tp.div(rg)
        # compute average precision
        ap[k] = precision[truth.bool()].sum() / max(float(truth.sum()), 1)
    return ap


def evaluate(model, dataloader, tokenizer, classnames, templates, device, amp=True, verbose=False, save_clf=None, load_clfs=[], partial_mask=False, normalize=False, normalize_type="add", normalizer=None, distributed=False):
    """
    Run zero-shot classification and evaluate the metrics

    Parameters
    ----------

    model: torch.nn.Module
        CLIP-like model with `encode_image` and `encode_text`
    
    dataloader: torch.utils.data.Dataloader

    tokenizer: text tokenizer

    classnames: list of str
        class names
    
    templates: list of str
        templates to use for zero-shot classification
    
    device: cpu/cuda

    amp: whether to use automatic mixed precision

    verbose: whether to use verbose model

    Returns
    -------

    dict of classification metrics
    """
   
    classifier = zero_shot_classifier(model, tokenizer, classnames, templates, device, amp=amp)
    
    if save_clf is not None:
        torch.save(classifier, save_clf)
        # exit() - not sure if we want to exit here or not.

    logits, target = run_classification(
        model, classifier, dataloader, device, amp=amp, partial_mask=partial_mask, normalize=normalize, 
        normalize_type=normalize_type, normalizer=normalizer, distributed=distributed
    )
    is_multilabel = (len(target.shape) == 2)

    if is_multilabel:
        if verbose:
            print("Detected a multi-label classification dataset")
        # Multiple labels per image, multiple classes on the dataset
        ap_per_class = average_precision_per_class(logits, target)
        if verbose:
            for class_name, ap in zip(dataloader.dataset.classes, ap_per_class.tolist()):
                print(f"Class: {class_name}, AveragePrecision: {ap}")
        return {"mean_average_precision": ap_per_class.mean().item()}
    else:
        # Single label per image, multiple classes on the dataset
        # just compute accuracy and mean_per_class_recall

        pred = logits.argmax(axis=1)
        # measure accuracy
        if len(dataloader.dataset.classes) >= 5:
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        else:
            acc1, = accuracy(logits, target, topk=(1,))
            acc5 = float("nan") 
        mean_per_class_recall = balanced_accuracy_score(target, pred)
        if verbose:
            print(classification_report(target, pred, digits=3))
        return {"acc1": acc1, "acc5": acc5, "mean_per_class_recall": mean_per_class_recall}