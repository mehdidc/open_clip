import logging
from contextlib import suppress

import torch
import torch.nn.functional as F
from tqdm import tqdm

def evaluate(model, dataloader, tokenizer,  device, amp=True, recall_k_list=[5], normalize=False, normalize_type="add", normalizer=None, mask_input=False, clip_augment=""):
    """
    Evaluate the model on the given dataset

    Parameters
    ----------
    
    model: torch.nn,Module
        CLIP-like model with `encode_image` and `encode_text`
    
    dataloader: torch.utils.data.Dataloader
        dataloader to use for evaluation

    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers
    
    device: cpu/cuda

    amp: whether to use automatic mixed precision
    
    Returns
    -------
    
    dict of accuracy metric
    """
    if normalize:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if normalizer not in ("self_normalize", "self_guidance"):
            model_id = normalizer        
            lm_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map=device, trust_remote_code=True)
            lm_tokenizer =  AutoTokenizer.from_pretrained(model_id)
    if clip_augment:
        import open_clip
        modeln, pretrained = clip_augment.split(":")
        clip, _, _ = open_clip.create_model_and_transforms(modeln, pretrained=pretrained)
        clip_tokenizer = open_clip.get_tokenizer(modeln)
        clip.to(device)
    autocast = torch.cuda.amp.autocast if amp else suppress
    preds = []
    for batch_images, batch_texts in tqdm(dataloader):
        batch_images = batch_images.to(device)
        # tokenize all texts in the batch
        batch_texts_tok = tokenizer([text for i, texts in enumerate(batch_texts) for text in texts]).to(device)
        batch_texts_raw = ([text for i, texts in enumerate(batch_texts) for text in texts])#.to(device)

        nb_texts_for_each_image = [len(texts) for texts in batch_texts]

        # compute the embedding of images and texts
        with torch.no_grad(), autocast():
            C, H, W =batch_images.shape[1:]# TODO - make this a parameter

            batch_images_emb = model.encode_image(batch_images)
            start = 0
            for i, nb in enumerate(nb_texts_for_each_image):
                end = start + nb
                image_emb = batch_images_emb[i:i+1]
                texts = batch_texts_tok[start:end]
                texts_raw = batch_texts_raw[start:end]
                max_text_len = (texts==0).float().argmax(dim=-1).max().item()
                texts = texts[:, :max_text_len]
                if torch.any(torch.isnan(image_emb)):
                    print("Detected nans in image embs..")
                    return {"acc": 0.0}
                if model.causal_mask is False:
                    raw = model.predict(
                        image_embs=image_emb, 
                        max_text_len=texts.shape[1],
                    )
                    if normalizer == "self_guidance":
                        image_embs = model.encode_image(torch.zeros(len(texts), C, H, W).float().to(device))
                        uncond_logits = model.predict(image_embs=image_embs, max_text_len=texts.shape[1])
                        alpha = 2
                        raw = (1+alpha) * raw - alpha * uncond_logits
                    scores = model.score(raw, texts)
                elif model.causal_mask is True:
                    nim, lim, dim = image_emb.shape
                    ntext, ltext = texts.shape
                    image_embs_p = image_emb.view(nim, 1, lim, dim).repeat(1, ntext, 1, 1).view(nim*ntext, lim, dim)
                    texts_p = texts.view(1, ntext, ltext).repeat(nim, 1, 1).view(nim*ntext, ltext)
                    input_text = texts_p[:, 0:-1]
                    if mask_input:
                        input_text = input_text.clone()
                        input_text.fill_(model.pad_id)
                    out_text = texts_p[:, 1:]
                    logits = model.predict(image_embs=image_embs_p, text=input_text)
                    scores = model.score_aligned(logits, out_text)
                    scores = scores.view(nim, ntext)
                if normalize:
                    if normalizer == "self_normalize":
                        if model.causal_mask is False:
                            target_text = texts
                            image_embs = model.encode_image(torch.zeros(len(texts), C, H, W).float().to(device))
                            logits = model.predict(image_embs=image_embs, max_text_len=target_text.shape[1])
                            priors = model.score_aligned(logits, target_text)
                        elif model.causal_mask is True:
                            input_text = texts[:, 0:-1]
                            target_text = texts[:, 1:]
                            image_embs = model.encode_image(torch.zeros(len(input_text), C, H, W).float().to(device))
                            logits = model.predict(image_embs=image_embs, text=input_text)
                            priors = model.score_aligned(logits, target_text)
                    elif normalizer == "self_guidance":
                        priors = 0
                    else:
                        lls = []
                        for ti in texts_raw:
                            tokenized_text = lm_tokenizer(ti, return_tensors="pt").input_ids.to(device)
                            output = lm_model(tokenized_text, labels=tokenized_text)
                            ll = -output.loss.sum()
                            lls.append(ll.item())
                        priors = torch.Tensor(lls).view(1, -1).to(device)
                    if normalize_type == "add":
                        scores = scores + priors
                    elif normalize_type == "sub":
                        scores = scores - priors
                if clip_augment:
                    texts_clip = clip_tokenizer(texts_raw).to(device)
                    emb_im = clip.encode_image(batch_images[i:i+1], normalize=True)
                    emb_text = clip.encode_text(texts_clip, normalize=True)
                    clip_scores = emb_im @ emb_text.T
                    scores = scores + clip_scores
                scores = scores[0]
                if torch.any(torch.isnan(scores)):
                    print("Detected nans..")
                    return {"acc": 0.0}
                pred = scores.argmax().item()
                start = end 
                preds.append(pred)
    pred = torch.Tensor(preds).long()
    acc = (pred==0).float().mean().item() # 0 is the index of the caption, the rest (>0) are considered negative captions
    metrics = {}
    metrics[f"acc"] = acc
    return metrics
