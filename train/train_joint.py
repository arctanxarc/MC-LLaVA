import argparse
import json
import torch.multiprocessing as mp
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
from llava.dataset import *
from llava.build_qa import generate_joint_qa_pairs
from llava.mm_utils import (expand2square, get_model_name_from_path, tokenizer_image_token,
                            tokenizer_image_token_batch)
import logging
import datetime
from tqdm import tqdm
IMAGE_TOKEN_INDEX = -200

import warnings
warnings.filterwarnings("ignore")

def get_train_args():
    parser = argparse.ArgumentParser()
    #--- Model related
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.6-vicuna-13b")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--conv_mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    #--- Dataset related
    parser.add_argument("--dataset_root", type=str, default="../mcdataset")
    parser.add_argument("--generate_data_root", type=str, default="../generate_training_data")
    parser.add_argument("--neg_template", type=str, default="../recognize_negative_template.json")
    
    #--- K-means related
    parser.add_argument("--use_k_means", default=False, action='store_true')
    parser.add_argument("--when2cls", type=str, default="mm")
    
    #-- Training related
    parser.add_argument("--all_vqa", default=False, action='store_true')
    parser.add_argument("--sks_names", nargs='+', default=['tangmu', 'jierui'])
    parser.add_argument("--prefix_token", type=int, default=16)
    parser.add_argument("--flip_p", type=float, default=0.5)
    parser.add_argument("--pos_image", default=False, action='store_true')
    parser.add_argument("--joint_image", default=False, action='store_true')
    parser.add_argument("--random_image", default=False, action='store_true')
    parser.add_argument("--conversation", default=False, action='store_true')
    parser.add_argument("--extreme_negative", default=False, action='store_true')
    parser.add_argument("--use_mask", default=False, action='store_true')
    
    #--- Log related
    parser.add_argument("--more_negative2test", default=False, action='store_true')
    parser.add_argument("--no_test", default=False, action='store_true')
    parser.add_argument("--checkpoint_path", type=str, default='checkpoints')
    parser.add_argument("--exp_name", type=str, default='train')
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=15)
    train_args = parser.parse_args()
    return train_args


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def get_embeddings(image_tensor, model, when2cls='mm'):
    vision_tower = model.get_vision_tower()
    mm_projector = model.get_model().mm_projector
    image_features_clip = vision_tower(image_tensor)    # [1, 576, 1024]
    if when2cls == 'clip':
        return image_features_clip
    elif when2cls == 'mm':
        image_features_mm = mm_projector(image_features_clip.to(dtype=model.dtype))  # [1, 576, 5120]
        return  image_features_mm
    else:
        raise NotImplementedError(f"Unsupported when2cls: {when2cls}")


def process_masks(masks):
    processed_masks = []
    for mask in masks:
        mask = expand2square(mask, 0)
        mask = mask.resize((336, 336), Image.NEAREST)
        mask_array = np.array(mask)
        mask_tensor = torch.tensor(mask_array, dtype=torch.uint8) // 255
        mask_tensor = mask_tensor.unsqueeze(0)
        processed_masks.append(mask_tensor)
    return torch.cat(processed_masks, dim=0)    # [n, 336, 336]


def main(args):
    
    # Set up logger & save location
    save_locations = []
    for sks in args.sks_names:
        os.makedirs(os.path.join("./train_logs", sks), exist_ok=True)
        logger = get_logger(os.path.join("./train_logs", sks, args.exp_name + f"_{datetime.datetime.now().strftime('%Y%m%d-%H%M.log')}"))
        save_location = os.path.join(args.checkpoint_path, sks, args.exp_name)
        os.makedirs(save_location, exist_ok=True)
        save_locations.append(save_location)
    logger.info(f"Arguments: {args}")
    
    # Set up model
    args.model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = get_model(args)
    model.config.image_aspect_ratio = 'pad'     # For add mask easier
    
    # Set up dataset
    if args.joint_image:
        concept_case = data_w_case[args.sks_names[0]]
        train_root_dir = os.path.join(args.dataset_root, concept_case, "concept", "train")
        generate_joint_qa_pairs(train_root_dir, args.sks_names, args.neg_template, args.generate_data_root)
    
    train_dataset = TrainingDataset(
        generate_data_root=args.generate_data_root,
        sks_names = args.sks_names,
        device=model.device,
        config=model.config,
        image_processor=image_processor,
        flip_p= args.flip_p,
        pos_image=args.pos_image,
        joint_image=args.joint_image,
        random_image=args.random_image,
        conversation=args.conversation,
        extreme_negative=args.extreme_negative,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=2
    )

    test_dataset = TestDatasetForMultiRec(
        dataset_root=args.dataset_root,
        concept_names=args.sks_names,
        device=model.device,
        config=model.config,
        image_processor=image_processor,
        more_negative=args.more_negative2test,
    )
    
    logger.info(f'Image path in tests: {test_dataset.image_paths}')
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=1
    )

    logger.info(f'concepts are: {args.sks_names}')
    logger.info(f'Number of training samples: {len(train_dataset)}')

    # Record the existing norm, for norm the centroids
    existing_embeddings = model.get_input_embeddings().weight
    existing_norm = existing_embeddings.norm(dim=1).mean()
    
    # --- Add <sks>
    joint_sks_prompt = ''
    joint_placeholder_tokens = []
    if args.prefix_token > 0:
        for i, sks in enumerate(args.sks_names):
            prefix_tokens = [f'<token{j + args.prefix_token * i}>' for j in range(args.prefix_token)]
            placeholder_tokens = [f'<{sks}>']
            placeholder_tokens.extend(prefix_tokens)
            sks_prompt = f"{placeholder_tokens[0]} is {''.join(placeholder_tokens[1:])}."
            logger.info(f'system prompt will add: {sks_prompt}')
            joint_sks_prompt += sks_prompt
            joint_placeholder_tokens.extend(placeholder_tokens)
    else:
        for i, sks in enumerate(args.sks_names):
            placeholder_tokens = [f'<{sks}>']
            sks_prompt = f"{placeholder_tokens[0]}"
            logger.info(f'system prompt will add: {sks_prompt}')
            joint_sks_prompt += sks_prompt
            joint_placeholder_tokens.extend(placeholder_tokens)
    num_added_tokens = tokenizer.add_tokens(joint_placeholder_tokens)
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(joint_placeholder_tokens)
        
    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # Record the original embeddings and lm_head
    orig_embeds_params = model.get_input_embeddings().weight.data.clone()   # [32017, 5120]
    orig_lm_params = model.lm_head.weight.data.clone()  # [5120, 32017]
    
    # Set up optimizer
    trainable_params = [model.get_input_embeddings().weight, model.lm_head.weight]
    optimizer = torch.optim.AdamW(
        trainable_params, # for optimize the embeddings and the head
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=1e-6,
        eps=1e-08,
    )
    
    #------------------------------------
    if args.use_k_means:
        
        for i, sks_name in enumerate(args.sks_names):

            concept_case = data_w_case[sks_name]
            train_imgs_paths_file = os.path.join(args.dataset_root, concept_case, "concept", "train")
            
            with open(os.path.join(train_imgs_paths_file, "train.json"), 'r') as f:
                train_imgs_paths = json.load(f)
            
            train_imgs = train_imgs_paths[sks_name]
            train_imgs = [os.path.join(train_imgs_paths_file, sks_name, x) for x in train_imgs]
            
            if args.use_mask:
                train_masks = [re.sub(r"\.(jpg|jpeg|png)$", "_mask.png", x) for x in train_imgs]
            else:
                train_masks = [None] * len(train_imgs)
                
            assert len(train_imgs) == len(train_masks), f"Number of images and masks do not match for {sks_name}"
                
            logger.info(f"Training images for {sks_name}: {train_imgs}")
            logger.info(f"Training masks for {sks_name}: {train_masks}")
            
            if args.use_mask:
                train_masks = [Image.open(x).convert("L") for x in train_masks]
                train_mask_tensors = process_masks(train_masks).to(model.device)    # [n, 336, 336]
            
            train_imgs = [Image.open(x).convert("RGB") for x in train_imgs]
            train_img_tensors = process_images(train_imgs,
                                               image_processor,
                                               model.config).to(model.device)      # [n, 3, 336, 336]

            img_embeddings = []
            for img, mask in zip(train_img_tensors, train_mask_tensors):
                img_tensor = img.unsqueeze(0)   # [1, 3, 336, 336]
                img_embedding = get_embeddings(img_tensor, model, args.when2cls).to(model.device)   # [1, 576, 5120] or [1, 576, 1024]
                img_embedding = img_embedding.squeeze(0)  # [576, 5120] or [576, 1024]
                
                if args.use_mask:
                    mask_tensor = mask.unsqueeze(0).unsqueeze(0)    # [1, 1, 336, 336]
                    mask_tensor = F.interpolate(mask_tensor, (24, 24))  # [1, 1, 24, 24]
                    binary_mask_flat = mask_tensor.view(-1)             # [576]
                    selected_features = img_embedding[binary_mask_flat == 1]
                else:
                    selected_features = img_embedding
                img_embeddings.append(selected_features)
        
            img_embeddings = torch.cat(img_embeddings, dim=0)  # [~576xn, 5120]
            img_embeddings_np = np.array([embedding.detach().cpu().to(torch.float32).numpy() for embedding in img_embeddings])
            kmeans = KMeans(n_clusters=args.prefix_token if img_embeddings_np.shape[0] > args.prefix_token else img_embeddings_np.shape[0], 
                            random_state=0)
            
            kmeans.fit(img_embeddings_np)
            centroids = kmeans.cluster_centers_
            centroids = torch.tensor(centroids).to(device=model.device, dtype=model.get_input_embeddings().weight.dtype)        # [16, 5120] or [16, 1024]
            
            if args.when2cls == 'clip':
                mm_projector = model.get_model().mm_projector
                centroids = mm_projector(centroids.unsqueeze(0).to(dtype=model.dtype)).squeeze(0).to(device=model.device)   # [16, 5120]
            
            centroids_norm = centroids.norm(dim=1, keepdim=True)
            centroids = centroids / centroids_norm * existing_norm
            
            sks = torch.mean(centroids, dim=0, keepdim=True).to(centroids.device)
            cat_tokens = torch.cat([sks, centroids], dim=0) # [17, 5120]]
            len_c = len(cat_tokens)
            
            # load the centroids to the placeholder tokens
            bias = i * (args.prefix_token + 1)
            
            with torch.no_grad():
                model.get_input_embeddings().weight[placeholder_token_ids[bias : len_c + bias]] = cat_tokens
        
    #------------------------------------ 
    
    model.train()
    model.model.requires_grad_(False)
    model.model.embed_tokens.weight.requires_grad_(True)
    best_acc = 0
    model.config.image_aspect_ratio = 'anyres'     # For add mask easier

    for names, p in model.named_parameters():
        if p.requires_grad:
            logger.info(f"{names} requires_grad")
                
    for epoch in tqdm(range(0, args.epoch)):
        for step, batch in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            prompt = [get_query(args, x, model=model, sks_system_prompt = joint_sks_prompt).conv.get_prompt() for x in batch['query']]
            prompt = [x + ' '+ y for x, y in zip(prompt, batch['answer'])]
            
            #--- Train with text only
            if not args.all_vqa and not batch['has_image']:
                prompt = [x.replace('<image>\n', '') for x in prompt]
            
            input_ids, labels = tokenizer_image_token_batch(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt", return_labels=True)
            input_ids = input_ids.to(model.device)
            labels = labels.to(model.device)

            with torch.cuda.amp.autocast(enabled=False, dtype=torch.float16):
                if not args.all_vqa and not batch['has_image']:
                    outputs = model(input_ids, labels=labels)
                else:
                    outputs = model(input_ids, images=batch['images'][0], labels=labels, image_sizes=batch['image_sizes'])
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            
            #---- Do not update the embedding matrix except the place holder
            index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
            index_no_updates[placeholder_token_ids] = False

            with torch.no_grad():
                model.get_input_embeddings().weight[
                    index_no_updates
                ] = orig_embeds_params[index_no_updates]

                model.lm_head.weight[index_no_updates] = orig_lm_params[index_no_updates]
            
            logger.info(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")
            logger.info(f"  Loss/Token-Norm: {model.get_input_embeddings().weight[placeholder_token_ids].norm().item()}")
            logger.info(f"  Loss/index_no_updates-Norm: {model.get_input_embeddings().weight[index_no_updates].norm().item()}")
            logger.info(f"  Loss/lm-head-norm: {model.lm_head.weight[placeholder_token_ids].norm().item()}")
            logger.info(f"  Loss/index_no_updates-lm-head: {model.lm_head.weight[index_no_updates].norm().item()}")
            
        if epoch % args.log_every == 0:
            for i, sks in enumerate(args.sks_names):
                save_location = save_locations[i]
                logger.info(f"Save model for {sks} at: {save_location}")
                save_path_token = os.path.join(save_location, f'{epoch}-token.pt')
                save_path_lmhead = os.path.join(save_location, f'{epoch}-lmhead.pt')
                torch.save(model.get_input_embeddings().weight.data[placeholder_token_ids[(args.prefix_token + 1) * i:(args.prefix_token + 1) * (i + 1)]], save_path_token)
                torch.save(model.lm_head.weight.data[placeholder_token_ids[(args.prefix_token + 1) * i:(args.prefix_token + 1) * (i + 1)]], save_path_lmhead)
        if args.no_test:
            continue
        with torch.no_grad():
            logger.info(f'Epoch {epoch} : Test MultiRec')
            preds = []
            gts = []
            conv_types = []
            for j, batch in enumerate(tqdm(test_dataloader)):
                #--- Ground Truth Answer
                prompt = [get_query(args, x, model=model, sks_system_prompt = joint_sks_prompt).conv.get_prompt() for x in batch['query']]
                input_ids, _ = tokenizer_image_token_batch(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt", return_labels=True)
                outputs = model.generate(input_ids.to(model.device), images=batch['images'][0].to(model.device), image_sizes=batch['image_sizes'])
                answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                preds.append(answer)
                gts.append(batch['answer'][0])
                conv_types.append(batch['type'][0])
            
            logger.info("    Predictions: ")
            logger.info("    ", preds)
            logger.info("    Ground truths: ")
            logger.info("    ", gts)
            logger.info("    Conversation types: ")
            logger.info("    ", conv_types)

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

            logger.info(f"  Accuracy for si_sq: {acc_si_sq}//{num_si_sq}, for mi_sq: {acc_mi_sq}//{num_mi_sq}, for mi_mq: {acc_mi_mq}//{num_mi_mq}. Total: {acc_total}//{num_total}")
            logger.info(f"  Recall for si_sq: {recall_si_sq}//{true_num_si_sq}, for mi_sq: {recall_mi_sq}//{true_num_mi_sq}, for mi_mq: {recall_mi_mq}//{true_num_mi_mq}. Total: {recall_total}//{true_num_total}")
            logger.info(f"  No recall for si_sq: {no_recall_si_sq}//{false_num_si_sq}, for mi_sq: {no_recall_mi_sq}//{false_num_mi_sq}, for mi_mq: {no_recall_mi_mq}//{false_num_mi_mq}. Total: {no_recall_total}//{false_num_total}")

            metric_now = (recall_total + no_recall_total) / 2
            
            if (metric_now >= best_acc) and (epoch > 4):
                for i, sks in enumerate(args.sks_names):
                    save_location = save_locations[i]
                    logger.info(f"Best accuracy for {sks} in Epoch {epoch}:  {metric_now}")
                    save_path_token = os.path.join(save_location, 'best-token.pt')
                    save_path_lmhead = os.path.join(save_location, 'best-lmhead.pt')
                    torch.save(model.get_input_embeddings().weight.data[placeholder_token_ids[(args.prefix_token + 1) * i:(args.prefix_token + 1) * (i + 1)]], save_path_token)
                    torch.save(model.lm_head.weight.data[placeholder_token_ids[(args.prefix_token + 1) * i:(args.prefix_token + 1) * (i + 1)]], save_path_lmhead)
                best_acc = metric_now


if __name__ == "__main__":
    mp.set_start_method('spawn')
    args = get_train_args()
    main(args)
