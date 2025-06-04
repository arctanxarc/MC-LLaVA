from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
import os
import json
import re
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
import numpy as np
from sklearn.cluster import KMeans
import torch
from llava.mm_utils import (expand2square, get_model_name_from_path, process_images)
from PIL import Image
import argparse

DEFAULT_FONT_SIZE = 15
SCALE = 1.0
TAU = 0.318
GAMMA = 100/256/256
when2cls = "clip"

def get_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.6-vicuna-13b")
    parser.add_argument("--model_name", type=str, default="llava-v1.6-vicuna-13b")
    parser.add_argument("--dataset_root", type=str, default="../mc_dataset")
    parser.add_argument("--model_base", type=str, default=None)
    return parser.parse_args()


def get_model(device, model_path=None, model_name=None, model_base=None):
    disable_torch_init()
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name, device_map=device
    )
    return tokenizer, model, image_processor, context_len


def draw_text(ax, text, position, *, font_size=None, color="g", horizontal_alignment="center", rotation=0):
    if not font_size:
        font_size = DEFAULT_FONT_SIZE

    color_rgb = np.maximum(list(mplc.to_rgb(color)), 0.15)
    color_rgb[np.argmax(color_rgb)] = max(0.8, np.max(color_rgb))

    def contrasting_color(rgb):
        R, G, B = rgb
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        return 'black' if Y > 128 else 'white'

    bbox_background = contrasting_color(color_rgb * 255)

    x, y = position
    ax.text(
        x,
        y,
        text,
        size=font_size * SCALE,
        family="sans-serif",
        bbox={"facecolor": bbox_background, "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
        verticalalignment="top",
        horizontalalignment=horizontal_alignment,
        color=color_rgb,
        zorder=10,
        rotation=rotation,
    )

def draw_numbers_on_image(ax, number_coords, font_size=DEFAULT_FONT_SIZE, color="g"):
    for num, coord in number_coords.items():
        draw_text(ax, str(num), coord, font_size=font_size, color=color)
        

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


def get_embeddings(image_tensor, model, when2cls='mm'):
    vision_tower = model.get_vision_tower().to(model.device)
    mm_projector = model.get_model().mm_projector.to(model.device)
    image_features_clip = vision_tower(image_tensor)    # [1, 576, 1024]
    if when2cls == 'clip':
        return image_features_clip
    elif when2cls == 'mm':
        image_features_mm = mm_projector(image_features_clip.to(dtype=model.dtype))  # [1, 576, 1024]
        return  image_features_mm
    else:
        raise NotImplementedError(f"Unsupported when2cls: {when2cls}")


def reshapefromstart2orig_shape(image2vec, image_ori_shapes):
    reshaped_image2vec = {}
    for idx, (img_path, vec) in enumerate(image2vec.items()):
        H, W = image_ori_shapes[idx]
        vec = vec.view(24, 24, -1)    # [24, 24, 1024]
        vec = vec.permute(2, 0, 1)    # [1024, 24, 24] 
        vec = F.interpolate(vec.unsqueeze(0), (max(H, W), max(H, W)), mode="bicubic")  # [1, 1024, max, max] 
        pad_size = (max(H, W) - min(H, W)) // 2
        print(vec.shape)
        if H > W:
            vec = vec[:, :, :, pad_size:-pad_size]
        else:
            vec = vec[:, :, pad_size:-pad_size, :]
            
        print(vec.shape)
        print(H, W)
        
        vec = vec.squeeze(0)  # [1024, H, W] or [1024, H, W]
        reshaped_image2vec[img_path] = vec
    return reshaped_image2vec

def reshapefromstart2336_shape(image2vec):
    reshaped_image2vec = {}
    for idx, (img_path, vec) in enumerate(image2vec.items()):
        vec = vec.view(24, 24, -1)    # [24, 24, 1024]
        vec = vec.permute(2, 0, 1)    # [1024, 24, 24]
        vec = F.interpolate(vec.unsqueeze(0), (336, 336), mode="bicubic")  # [1, 1024, H, W]
        vec = vec.squeeze(0)  # [1024, 336, 336] or [1024, 336, 336]
        reshaped_image2vec[img_path] = vec
    return reshaped_image2vec

def reshapefromstart224_shape(image2vec):
    reshaped_image2vec = {}
    for idx, (img_path, vec) in enumerate(image2vec.items()):
        vec = vec.view(24, 24, -1)    # [24, 24, 1024]
        vec = vec.permute(2, 0, 1)    # [1024, 24, 24]
        reshaped_image2vec[img_path] = vec
    return reshaped_image2vec


def main():
    args = get_test_args()
    model_path = args.model_path
    model_name = args.model_name
    dataset_root = args.dataset_root
    model_base = args.model_base

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model, image_processor, context_len = get_model(device, model_path, model_name, model_base)
    model = model.to(device)
    model.config.image_aspect_ratio = 'pad'  # For easier mask addition

    json_file4concepts = os.path.join(dataset_root, "concept_type.json")
    json_file4scenarios = os.path.join(dataset_root, "scenarios.json")

    with open(json_file4scenarios) as f:
        cases = json.load(f)

    with open(json_file4concepts) as f:
        data_w_case = json.load(f)

    with torch.no_grad():
        for case in cases:
            print(f"Processing case: {case}")
            sks_names = list(case.split("_"))
            concept2vecs = {}

            for i, sks_name in enumerate(sks_names):
                concept_case = data_w_case[sks_name]
                train_imgs_paths_file = os.path.join(dataset_root, concept_case, "concept", "train")

                with open(os.path.join(train_imgs_paths_file, "train.json"), 'r') as f:
                    train_imgs_paths = json.load(f)

                train_imgs = train_imgs_paths[sks_name]
                train_imgs = [os.path.join(train_imgs_paths_file, sks_name, x) for x in train_imgs]
                train_masks = [re.sub(r"\.(jpg|jpeg|png)$", "_mask.png", x) for x in train_imgs]

                assert len(train_imgs) == len(train_masks), f"Number of images and masks do not match for {sks_name}"

                train_masks = [Image.open(x).convert("L") for x in train_masks]
                train_mask_tensors = process_masks(train_masks).to(device)

                train_imgs = [Image.open(x).convert("RGB") for x in train_imgs]
                train_img_tensors = process_images(train_imgs, image_processor, model.config).to(device)

                img_embeddings = []
                for img, mask in zip(train_img_tensors, train_mask_tensors):
                    img_tensor = img.unsqueeze(0)
                    img_embedding = get_embeddings(img_tensor, model, when2cls).to(model.device)
                    img_embedding = img_embedding.squeeze(0)

                    mask_tensor = mask.unsqueeze(0).unsqueeze(0)
                    mask_tensor = F.interpolate(mask_tensor, (24, 24))
                    binary_mask_flat = mask_tensor.view(-1)
                    selected_features = img_embedding[binary_mask_flat == 1]

                    img_embeddings.append(selected_features)

                img_embeddings = torch.cat(img_embeddings, dim=0)
                concept2vecs[sks_name] = img_embeddings

            multi_image_path = os.path.join(dataset_root, concept_case, "multi", case)
            multi_imgs = [
                os.path.join(multi_image_path, image_name)
                for image_name in sorted(os.listdir(multi_image_path))
                if image_name.endswith(("jpg", "jpeg", "png")) and
                "mask" not in image_name and "som" not in image_name and "bbox" not in image_name
            ]

            assert len(multi_imgs) == 5, f"Number of multi images do not match for {case}"
            som_info = {}

            for multi_img_path in multi_imgs:
                multi_img = Image.open(multi_img_path).convert("RGB")
                multi_img_tensor = process_images([multi_img], image_processor, model.config).to(device)
                test_image_feature = get_embeddings(multi_img_tensor, model, when2cls).squeeze(0).to(device)
                test_image_feature = test_image_feature.permute(1, 0)
                
                
                sims_numpy_means = []
                target_croods = {}
                
                for idx, (sks_name, vecs) in enumerate(concept2vecs.items()):
                    sims = torch.matmul(vecs, test_image_feature) / torch.norm(vecs, dim=1).unsqueeze(1) / torch.norm(test_image_feature, dim=0).unsqueeze(0)
                    sims = sims.view(-1, 24, 24)
                    sims = F.interpolate(sims.unsqueeze(0), (336, 336), mode="bilinear")
                    ori_image_size = multi_img.size[::-1]
                    sims = F.interpolate(sims, (max(ori_image_size), max(ori_image_size)), mode="bilinear")

                    pad_size = (max(ori_image_size) - min(ori_image_size)) // 2
                    if ori_image_size[0] > ori_image_size[1]:
                        sims = sims[:, :, :, pad_size:-pad_size]
                    else:
                        sims = sims[:, :, pad_size:-pad_size, :]
                    # sims = F.interpolate(sims, ori_image_size, mode="bilinear", align_corners=False)

                    sims_numpy_mean = sims.squeeze(0).float().detach().cpu().numpy().mean(axis=0)
                    sims_numpy_means.append(sims_numpy_mean)
                    # if (sims_numpy_mean > TAU).sum() / sims_numpy_mean.size > GAMMA:
                        
                    #     max_idx = np.unravel_index(sims_numpy_mean.argmax(), sims_numpy_mean.shape)
                    #     target_croods[idx + 1] = (max_idx[1], max_idx[0])
                    # else:
                    #     print(f"{sks_name} is not in the image {multi_img_path}")
                
                del sims
                global_sims_mean = np.mean(sims_numpy_means, axis=0)  # [H, W]
                if np.isnan(global_sims_mean).any():
                    print("Warning: global_sims_mean contains NaN values, replacing with 0")
                    global_sims_mean = np.nan_to_num(global_sims_mean)
                    
                for idx, sims_numpy_mean in enumerate(sims_numpy_means):
                    adjusted_sims = sims_numpy_mean - global_sims_mean  # [H, W]
                    if adjusted_sims.size == 0 or np.isnan(adjusted_sims).all():
                        print(f"Skipping empty or NaN adjusted_sims for {list(concept2vecs.keys())[idx]}")
                        continue
                    min_adj, max_adj = adjusted_sims.min(), adjusted_sims.max()
                    min_orig, max_orig = sims_numpy_mean.min(), sims_numpy_mean.max()
                    
                    if max_adj - min_adj < 1e-6:  
                        adjusted_sims = sims_numpy_mean  
                    else:
                        adjusted_sims = (adjusted_sims - min_adj) / (max_adj - min_adj)  
                        adjusted_sims = adjusted_sims * (max_orig - min_orig) + min_orig  

                    if (adjusted_sims > TAU).sum() / adjusted_sims.size > GAMMA:
                        max_idx = np.unravel_index(adjusted_sims.argmax(), adjusted_sims.shape)
                        target_croods[idx + 1] = (max_idx[1], max_idx[0])  
                    else:
                        print(f"Concept {list(concept2vecs.keys())[idx]} is not in the image {multi_img_path}")
                        
                som_info[multi_img_path] = target_croods
                del test_image_feature

                multi_img_array = np.array(multi_img)
                height, width = multi_img_array.shape[:2]
                dpi = 100
                fig_width = width / dpi
                fig_height = height / dpi

                fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
                ax.imshow(multi_img_array)
                ax.axis("off")

                draw_numbers_on_image(ax, target_croods, font_size=20, color="white")

                ext = os.path.splitext(multi_img_path)[1]
                save_path = multi_img_path.replace(ext, f"_som{ext}")
                plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
                plt.close(fig)  

                print(f"Saved result to {save_path}")

            for sks_name, vecs in concept2vecs.items():
                del vecs

            for img_path in som_info:
                som_info[img_path] = {int(k): (int(v[0]), int(v[1])) for k, v in som_info[img_path].items()}
            
            json_out_file = os.path.join(dataset_root, concept_case, "multi", case, "som_info.json")
            with open(json_out_file, "w") as f:
                json.dump(som_info, f)

            print(f"Saved som info to {json_out_file}")


if __name__ == "__main__":
    main()
