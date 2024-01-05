import logging
from contextlib import suppress

import torch
import torch.nn.functional as F
from tqdm import tqdm

def evaluate(model, dataloader, tokenizer,  device, amp=True, recall_k_list=[5], normalize=False, normalize_type="add", normalizer=None):
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
        model_id = normalizer
        #lm_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", flash_attn=True, flash_rotary=True, fused_dense=True, device_map=device, trust_remote_code=True)
        
        lm_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map=device, trust_remote_code=True)
        lm_tokenizer =  AutoTokenizer.from_pretrained(model_id)
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
                    scores = model.score(raw, texts)
                else:
                    nim, lim, dim = image_emb.shape
                    ntext, ltext = texts.shape
                    image_embs_p = image_emb.view(nim, 1, lim, dim).repeat(1, ntext, 1, 1).view(nim*ntext, lim, dim)
                    texts_p = texts.view(1, ntext, ltext).repeat(nim, 1, 1).view(nim*ntext, ltext)
                    input_text = texts_p[:, 0:-1]
                    out_text = texts_p[:, 1:]
                    logits = model.predict(image_embs=image_embs_p, text=input_text)
                    scores = model.score_aligned(logits, out_text)
                    scores = scores.view(nim, ntext)
                if normalize:
                    lls = []
                    for ti in texts_raw:
                        tokenized_text = lm_tokenizer(ti, return_tensors="pt").input_ids.to(device)
                        output = lm_model(tokenized_text, labels=tokenized_text)
                        ll = -output.loss.sum()
                        lls.append(ll.item())
                    ll = torch.Tensor(lls).view(1, -1).to(device)
                    if normalize_type == "add":
                        scores = scores + ll
                    elif normalize_type == "sub":
                        scores = scores - ll
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
