import argparse
import json
import os

import pandas as pd
from tqdm import tqdm
import torch

from llava.constants import IMAGE_TOKEN_INDEX
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from llava.mm_utils import tokenizer_image_token_batch
from llava.dataset_som import *
import warnings
warnings.filterwarnings("ignore")


def get_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.6-vicuna-13b")
    parser.add_argument("--dataset_root", type=str, default="../mcdataset")
    parser.add_argument("--checkpoint_path", type=str, default="../train/checkpoints")
    parser.add_argument("--model_base", type=str, default=None) 
    parser.add_argument("--conv_mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--concept_names", nargs='+', default=["Bafan", "Xuezhixia"]) 
    parser.add_argument("--ckpt_name", type=str, default="yollava") 
    parser.add_argument("--mode", type=str, default="choice") 
    parser.add_argument("--more_negative", action="store_true")
    parser.add_argument("--num_tokens", type=int, default=16)
    return parser.parse_args()


def load_trained_weights(model, tokenizer, checkpoint_path, sks_names, ckpt_name, epoch2load=11, prefix_token_count=16):
    all_placeholder_tokens = []
    for i, sks_name in enumerate(sks_names):
        tmp_name = sks_name.split('_')[0]
        placeholder_tokens = [f'<{tmp_name}>'] + [f'<token{j + prefix_token_count * i}>' for j in range(prefix_token_count)]
        num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
        model.resize_token_embeddings(len(tokenizer))
        
        placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
        
        token_path = os.path.join(checkpoint_path, sks_name, ckpt_name, f'{epoch2load}-token.pt')
        lmhead_path = os.path.join(checkpoint_path, sks_name, ckpt_name, f'{epoch2load}-lmhead.pt')
        # token_path = os.path.join(checkpoint_path, sks_name, ckpt_name, f'best-token.pt')
        # lmhead_path = os.path.join(checkpoint_path, sks_name, ckpt_name, f'best-lmhead.pt')
        
        if os.path.exists(token_path) and os.path.exists(lmhead_path):
            token_weights = torch.load(token_path).to(model.device)
            lmhead_weights = torch.load(lmhead_path).to(model.device)
            
            with torch.no_grad():
                model.get_input_embeddings().weight[placeholder_token_ids] = token_weights.to(model.get_input_embeddings().weight.device)
                model.lm_head.weight[placeholder_token_ids] = lmhead_weights.to(model.lm_head.weight.device)
        
            print(f"Loaded weights for {sks_name}")
        all_placeholder_tokens.append(placeholder_tokens)
    return all_placeholder_tokens


def calculate_bleu(reference, candidate):
    reference_tokens = list(reference)
    candidate_tokens = list(candidate)
    
    smoothie = SmoothingFunction().method1
    bleu_score = sentence_bleu([reference_tokens], candidate_tokens, 
                               weights=(0.7, 0.3), 
                               smoothing_function=smoothie)
    
    return bleu_score


def analyze_rec(conv_types, preds, gts):
    
    num_si_sq = conv_types.count("si_sq")
    num_mi_sq = conv_types.count("mi_sq")
    num_mi_mq = conv_types.count("mi_mq")
    num_total = len(conv_types)
    
    true_num_si_sq = sum([1 for i in range(num_total) if conv_types[i] == "si_sq" and gts[i] == "Yes"])
    true_num_mi_sq = sum([1 for i in range(num_total) if conv_types[i] == "mi_sq" and gts[i] == "Yes"])
    true_num_mi_mq = sum([1 for i in range(num_total) if conv_types[i] == "mi_mq" and gts[i] == "Yes"])
    true_num_total = sum([1 for i in range(num_total) if gts[i] == "Yes"])

    false_num_si_sq = sum([1 for i in range(num_total) if conv_types[i] == "si_sq" and gts[i] == "No"])
    false_num_mi_sq = sum([1 for i in range(num_total) if conv_types[i] == "mi_sq" and gts[i] == "No"])
    false_num_mi_mq = sum([1 for i in range(num_total) if conv_types[i] == "mi_mq" and gts[i] == "No"])
    false_num_total = sum([1 for i in range(num_total) if gts[i] == "No"])

    acc_si_sq = sum([1 for i in range(num_total) if conv_types[i] == "si_sq" and preds[i] == gts[i]]) / num_si_sq if num_si_sq else 0
    acc_mi_sq = sum([1 for i in range(num_total) if conv_types[i] == "mi_sq" and preds[i] == gts[i]]) / num_mi_sq if num_mi_sq else 0
    acc_mi_mq = sum([1 for i in range(num_total) if conv_types[i] == "mi_mq" and preds[i] == gts[i]]) / num_mi_mq if num_mi_mq else 0
    acc_total = sum([1 for i in range(num_total) if preds[i] == gts[i]]) / num_total if num_total else 0

    recall_si_sq = sum([1 for i in range(num_total) if conv_types[i] == "si_sq" and preds[i] == gts[i] and gts[i] == "Yes"]) / true_num_si_sq if true_num_si_sq else 0
    recall_mi_sq = sum([1 for i in range(num_total) if conv_types[i] == "mi_sq" and preds[i] == gts[i] and gts[i] == "Yes"]) / true_num_mi_sq if true_num_mi_sq else 0
    recall_mi_mq = sum([1 for i in range(num_total) if conv_types[i] == "mi_mq" and preds[i] == gts[i] and gts[i] == "Yes"]) / true_num_mi_mq if true_num_mi_mq else 0
    recall_total = sum([1 for i in range(num_total) if preds[i] == gts[i] and gts[i] == "Yes"]) / true_num_total if true_num_total else 0

    no_recall_si_sq = sum([1 for i in range(num_total) if conv_types[i] == "si_sq" and preds[i] == gts[i] and gts[i] == "No"]) / false_num_si_sq if false_num_si_sq else 0
    no_recall_mi_sq = sum([1 for i in range(num_total) if conv_types[i] == "mi_sq" and preds[i] == gts[i] and gts[i] == "No"]) / false_num_mi_sq if false_num_mi_sq else 0
    no_recall_mi_mq = sum([1 for i in range(num_total) if conv_types[i] == "mi_mq" and preds[i] == gts[i] and gts[i] == "No"]) / false_num_mi_mq if false_num_mi_mq else 0
    no_recall_total = sum([1 for i in range(num_total) if preds[i] == gts[i] and gts[i] == "No"]) / false_num_total if false_num_total else 0

    print(f"  Accuracy for si_sq: {acc_si_sq}//{num_si_sq}, for mi_sq: {acc_mi_sq}//{num_mi_sq}, for mi_mq: {acc_mi_mq}//{num_mi_mq}. Total: {acc_total}//{num_total}")
    print(f"  Recall for si_sq: {recall_si_sq}//{true_num_si_sq}, for mi_sq: {recall_mi_sq}//{true_num_mi_sq}, for mi_mq: {recall_mi_mq}//{true_num_mi_mq}. Total: {recall_total}//{true_num_total}")
    print(f"  No recall for si_sq: {no_recall_si_sq}//{false_num_si_sq}, for mi_sq: {no_recall_mi_sq}//{false_num_mi_sq}, for mi_mq: {no_recall_mi_mq}//{false_num_mi_mq}. Total: {no_recall_total}//{false_num_total}")

def analyze_yo_rec(preds, gts):
    # calculate recall and no_recall
    true_num = sum([1 for i in range(len(preds)) if gts[i] == "Yes"])
    false_num = sum([1 for i in range(len(preds)) if gts[i] == "No"])
    
    recall = sum([1 for i in range(len(preds)) if preds[i] == gts[i] and gts[i] == "Yes"]) / true_num if true_num else 0
    no_recall = sum([1 for i in range(len(preds)) if preds[i] == gts[i] and gts[i] == "No"]) / false_num if false_num else 0
    
    print(f"  Recall: {recall}//{true_num}, No recall: {no_recall}//{false_num}, Total: {(recall + no_recall) / 2}//{len(preds)}")
    

def analyze_choice(conv_types, preds, gts):
    
    num_si_sq = conv_types.count("si_sq")
    num_mi_sq = conv_types.count("mi_sq")
    num_mi_mq = conv_types.count("mi_mq")
    num_total = len(conv_types)
    
    # Calculate accuracy , and prevent division by zero
    acc_si_sq = sum([1 for i in range(num_total) if conv_types[i] == "si_sq" and preds[i] == gts[i]]) / num_si_sq if num_si_sq else 0
    acc_mi_sq = sum([1 for i in range(num_total) if conv_types[i] == "mi_sq" and preds[i] == gts[i]]) / num_mi_sq if num_mi_sq else 0
    acc_mi_mq = sum([1 for i in range(num_total) if conv_types[i] == "mi_mq" and preds[i] == gts[i]]) / num_mi_mq if num_mi_mq else 0
    acc_total = sum([1 for i in range(num_total) if preds[i] == gts[i]]) / num_total if num_total else 0
    
    print(f"  Accuracy for si_sq: {acc_si_sq}//{num_si_sq}, for mi_sq: {acc_mi_sq}//{num_mi_sq}, for mi_mq: {acc_mi_mq}//{num_mi_mq}. Total: {acc_total}//{num_total}")


def analyze_text_choice(conv_types, preds, gts):
    
    num_si = conv_types.count("s")
    num_mi = conv_types.count("m")
    num_total = len(conv_types)
    
    # calculate accuracy
    acc_si = sum([1 for i in range(num_total) if conv_types[i] == "s" and preds[i] == gts[i]]) / num_si if num_si else 0
    acc_mi = sum([1 for i in range(num_total) if conv_types[i] == "m" and preds[i] == gts[i]]) / num_mi if num_mi else 0
    acc_total = sum([1 for i in range(num_total) if preds[i] == gts[i]]) / num_total if num_total else 0
    
    print(f"  Accuracy for s: {acc_si}//{num_si}, for m: {acc_mi}//{num_mi}. Total: {acc_total}//{num_total}")
    
    
def analyze_vqa(conv_types, preds, gts):
    num_si_sq = conv_types.count("si_sq")
    num_mi_sq = conv_types.count("mi_sq")
    num_mi_mq = conv_types.count("mi_mq")
    num_total = len(conv_types)
    
    # calculate bleu by call calculate_bleu function
    bleu_si_sq = sum([calculate_bleu(gts[i], preds[i]) for i in range(num_total) if conv_types[i] == "si_sq"]) / num_si_sq if num_si_sq else 0
    bleu_mi_sq = sum([calculate_bleu(gts[i], preds[i]) for i in range(num_total) if conv_types[i] == "mi_sq"]) / num_mi_sq if num_mi_sq else 0
    bleu_mi_mq = sum([calculate_bleu(gts[i], preds[i]) for i in range(num_total) if conv_types[i] == "mi_mq"]) / num_mi_mq if num_mi_mq else 0
    bleu_total = sum([calculate_bleu(gts[i], preds[i]) for i in range(num_total)]) / num_total if num_total else 0
    
    print(f"  BLEU for si_sq: {bleu_si_sq}//{num_si_sq}, for mi_sq: {bleu_mi_sq}//{num_mi_sq}, for mi_mq: {bleu_mi_mq}//{num_mi_mq}. Total: {bleu_total}//{num_total}")


def analyze_caption(conv_types, preds, gts):
    
    num_si = conv_types.count("si")
    num_mi = conv_types.count("mi")
    num_total = len(conv_types)
    
    # calculate caption recall
    count_si = 0
    count_mi = 0
    for p, g, t in zip(preds, gts, conv_types):
        if t == "si":
            if g in p:
                count_si += 1
        elif t == "mi":
            # check the number of words from g in p
            # assert type of g is list
            assert type(g) == list
            gt_num = len(g)
            pred_num = 0
            for word in g:
                if word in p:
                    pred_num += 1
            count_mi += pred_num / gt_num if gt_num else 0
    
    recall_si = count_si / num_si if num_si else 0
    recall_mi = count_mi / num_mi if num_mi else 0
    recall_total = (count_si + count_mi) / num_total if num_total else 0
    
    print(f"  Caption Recall for si: {recall_si}//{num_si}, for mi: {recall_mi}//{num_mi}. Total: {recall_total}//{num_total}")

        
def main(args):
        
    tokenizer, model, image_processor, context_len = get_model(args)
    
    # Load concepts knowledge
    all_placeholder_tokens = load_trained_weights(model, tokenizer, args.checkpoint_path, args.concept_names, args.ckpt_name, prefix_token_count=args.num_tokens)
    sks_prompt = ""
    for i, _ in enumerate(args.concept_names):
        sks_prompt_i = f"{all_placeholder_tokens[i][0]} is {''.join(all_placeholder_tokens[i][1:])}."
        sks_prompt += sks_prompt_i
    print(sks_prompt)
    
    if args.mode == "rec":
        test_dataset = TestDatasetForMultiRec(args.dataset_root, 
                                              args.concept_names,
                                              model.device, 
                                              model.config, 
                                              image_processor, 
                                              args.more_negative)
    elif args.mode == "choice" or args.mode == "vqa":
        test_dataset = TestDatasetForMultiVQA(args.dataset_root,
                                              args.concept_names,
                                              model.device,
                                              model.config,
                                              image_processor,
                                              args.mode)
    elif args.mode == "caption":
        test_dataset = TestDatasetForCaptionGeneration(args.dataset_root,
                                                       args.concept_names,
                                                       model.device,
                                                       model.config,
                                                       image_processor)
    elif args.mode == "text":
        test_dataset = TestDatasetForTextOnlyQAChoice(args.dataset_root,
                                                        args.concept_names,)
    elif args.mode == "vg":
        test_dataset = TestDatasetForVG(args.dataset_root,
                                        args.concept_names,
                                        model.device,
                                        model.config,
                                        image_processor)
    
    with torch.no_grad():
        conv_types = []
        preds = []
        gts = []
        quaries = []
        prompts = []
        for i in tqdm(range(len(test_dataset))):
            example = test_dataset[i]
            
            if not example["has_image"]:
                query = example["query"]
                prompt = get_query(args, query, model=model, sks_system_prompt=sks_prompt).conv.get_prompt().replace('<image>\n', '')
                input_ids = [tokenizer.encode(prompt)]
                input_ids = torch.tensor(input_ids, dtype=torch.long).to(model.device)
                model.eval()
                with torch.no_grad():
                    outputs = model.generate(input_ids)
                answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                preds.append(answer)
                gts.append(example["answer"])
                conv_types.append(example["type"])
                quaries.append(query)
            else:
                image = example["images"]
                image_sizes = example["image_sizes"]
                conv_type = example["type"]
                
                if args.mode != "caption":
                    query = example["query"]
                else:
                    names_pool = args.concept_names
                    query = f"Can you see "
                    for name in names_pool:
                        tmp_name = name.split('_')[0]
                        query += f"<{tmp_name}>, "
                    query = query[:-2] + " in the image? Don't answer the question but remember it, and only response a detailed caption for the image. Your caption: "
                
                # if args.mode == "vg":
                prompt  = [get_query(args, query, model=model, sks_system_prompt = sks_prompt + example['som_prompt']).conv.get_prompt()]
                # else:
                    # prompt  = [get_query(args, query, model=model, sks_system_prompt = sks_prompt).conv.get_prompt()]
                input_ids, _ = tokenizer_image_token_batch(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt", return_labels=True)
                outputs = model.generate(input_ids.to(model.device), images=image, image_sizes=image_sizes)
                answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                prompts.append(prompt)
                quaries.append(query)
                preds.append(answer)
                if args.mode != "caption":
                    gts.append(example["answer"])
                else:
                    gts.append(example["query"])
                conv_types.append(conv_type)

    print("    Quaries: ")
    print(f"    {quaries}")
    print("    Prompts: ")
    print(f"    {prompts}")
    print("    Predictions: ")
    print(f"    {preds}")
    print("    Ground truths: ")
    print(f"    {gts}")
    print("    Conversation types: ")
    print(f"    {conv_types}")
    
    assert len(conv_types) == len(preds), "Number of predictions and conversation types do not match."
    assert len(conv_types) == len(gts), "Number of ground truths and conversation types do not match."
    
    if args.mode == "rec" or args.mode == "yorec":
        analyze_rec(conv_types, preds, gts) # acc, recall
    elif args.mode == "choice"  or args.mode == "vg":
        analyze_choice(conv_types, preds, gts) # acc
    elif args.mode == "vqa":
        analyze_vqa(conv_types, preds, gts) # bleu
    elif args.mode == "caption":
        analyze_caption(conv_types, preds, gts) # caption recall
    elif args.mode == "yollava":
        analyze_yo_rec(preds, gts) # recall
    elif args.mode == "text":
        analyze_text_choice(conv_types, preds, gts) # acc
    else:
        raise NotImplementedError(f"Mode {args.mode} is not implemented yet.")
    

if __name__ == "__main__":
    args = get_test_args()
    main(args)
