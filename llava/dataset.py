import json
import os
import random
import re
from io import BytesIO

import numpy as np
import requests
import torch
from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IMAGE_PLACEHOLDER)
from llava.conversation import  conv_templates
from llava.mm_utils import (expand2square, get_model_name_from_path, process_images)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


dataset_root = "/workspace/data/MulBench"
json_file4concepts = os.path.join(dataset_root, "concept_type.json")
with open(json_file4concepts) as f:
    data_w_case = json.load(f)

def get_test_image_paths_by_concept(dataset_root, concept_name):
    concept_case = data_w_case[concept_name]
    concept_test_images_path = os.path.join(dataset_root, concept_case, "concept", "test", concept_name)
    concept_test_image_paths = [os.path.join(concept_test_images_path, x) for x
                                in os.listdir(concept_test_images_path)
                                if x.lower().endswith(('.png', '.jpeg', '.jpg'))]
    return concept_test_image_paths


def get_test_multi_image_paths_by_concept(dataset_root, concept_name):
    concept_case = data_w_case[concept_name]
    multi_path = os.path.join(dataset_root, concept_case, "multi")
    multi_image_paths = []
    friends = [concept_name]
    for dir_name in os.listdir(multi_path):
        if not os.path.isdir(os.path.join(multi_path, dir_name)):
            continue
        if concept_name in dir_name:
            friends = list(dir_name.split('_'))
            multi_image_paths = [os.path.join(multi_path, dir_name, x) for x
                                 in os.listdir(os.path.join(multi_path, dir_name))
                                 if x.lower().endswith(('.png', '.jpeg', '.jpg'))]
            break
    return friends, multi_image_paths


class TrainingDataset(Dataset):
    def __init__(
        self,
        generate_data_root,   # /workspace/mulllava/generate_training_data
        sks_names,      # ["sks1", "sks2"]
        device,
        config,
        image_processor,
        flip_p=0.5,
        pos_image=True,
        joint_image=True,
        random_image=True,
        conversation=False,
        extreme_negative=False,
    ):
        self.generate_data_root = generate_data_root
        self.device = device
        self.config = config
        self.image_processor = image_processor
        self.flip_p = flip_p
        self.sks_names = sks_names
        
        # determine the conversation types for training        
        conversation_types = []
        if pos_image:
            conversation_types.append('recognition_positive.json')
        if random_image:
            conversation_types.append('recognition_negative_cc12m.json')
        if conversation:
            conversation_types.append('conversation.json')
        if extreme_negative:
            conversation_types.append('recognition_strong_negative.json')
        
        self.questions = []
        self.answers = []
        self.images_path = []
        self.has_image = []
        # Load data
        for sks in self.sks_names:
            if joint_image:
                sks_conversation_types = conversation_types + [f'recognition_negative_{other_sks}.json' for other_sks in sks_names if other_sks != sks]
            else:
                sks_conversation_types = conversation_types
            
            for conversation_type in sks_conversation_types:
                f = open(os.path.join(self.generate_data_root, sks, conversation_type))
                try:
                    data = json.load(f)
                except:
                    raise ValueError(f"Failed to load {os.path.join(self.generate_data_root, sks, conversation_type)}")
                file_names = [x for x in data.keys()]
                for file_name in file_names:
                    questions = []
                    answers = []
                    for conv in data[file_name]:
                        questions.append(conv['Human'])
                        answers.append(conv['AI'])
                    self.questions.extend(questions)
                    self.answers.extend(answers)
                
                    self.images_path.extend([file_name]*len(answers))
                    if 'conversation' in conversation_type:
                        self.has_image.extend([False]*len(answers))
                        print("No image")
                    else:
                        self.has_image.extend([True]*len(answers))
        print('Total: ', len(self.questions), len(self.answers), len(self.images_path), len(self.has_image))
        print(self.images_path)

        self._length = len(self.questions)
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image_path = self.images_path[i]
        images = [Image.open(image_path).convert("RGB")]
        images = [self.flip_transform(image) for image in images]
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            self.image_processor,
            self.config
        )
        example["images"] = images_tensor
        example['query'] = self.questions[i]
        example['answer'] = self.answers[i]
        example['has_image'] = self.has_image[i]
        example['image_sizes'] = image_sizes
        return example


class TestDatasetForSingleRec(Dataset):
    def __init__(
        self,
        dataset_root,   # /workspace/data/MulBench
        concept_name,   # sks
        device,
        config,
        image_processor,
        num_neg = 100,
    ):
        random.seed(1)
        self.dataset_root = dataset_root
        self.device = device
        self.config = config
        self.image_processor = image_processor
        self.concept_name = concept_name
        self.data_w_case = data_w_case
        
        # Load positive single concept images 
        pos_single_image_paths = get_test_image_paths_by_concept(self.dataset_root, concept_name)
        
        # Load positive multi concept images
        friends, pos_multi_image_paths = get_test_multi_image_paths_by_concept(self.dataset_root, concept_name)
        
        # Combine positive images
        self.pos_image_paths = pos_single_image_paths + pos_multi_image_paths
        num_pos = len(self.pos_image_paths)
        
        # Load negative images
        num_single_neg, num_multi_neg = num_neg // 2, num_neg // 2
        
        # Load neg single concept images
        neg_single_image_paths = []
        for _ in range(num_single_neg):
            neg_concept_name = random.choice(list(self.data_w_case.keys()))
            while neg_concept_name == concept_name:
                neg_concept_name = random.choice(list(self.data_w_case.keys()))
            
            neg_single_image_paths.append(random.choice(get_test_image_paths_by_concept(self.dataset_root, neg_concept_name)))
        
        # Load neg multi concept images
        neg_multi_image_paths = []
        for _ in range(num_multi_neg):
            neg_concept_name = random.choice(list(self.data_w_case.keys()))
            while neg_concept_name in friends:
                neg_concept_name = random.choice(list(self.data_w_case.keys()))
            
            neg_multi_image_paths.append(random.choice(get_test_multi_image_paths_by_concept(self.dataset_root, neg_concept_name)[1]))
        
        self.neg_image_paths = neg_single_image_paths + neg_multi_image_paths
        num_neg = len(self.neg_image_paths)
        
        self.image_paths = self.pos_image_paths + self.neg_image_paths
        self.gt = ['Yes'] * num_pos + ['No'] * num_neg
        self.num_images = len(self.image_paths)
        self._length = self.num_images

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image_path = self.image_paths[i]     
        
        images = [Image.open(image_path).convert("RGB")]
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            self.image_processor,
            self.config
        ).to(self.device)
        example["images"] = images_tensor
        example["query"] = f'Can you see <{self.concept_name}> in this photo? Answer the question using a single word Yes or No.'
            
        example["answer"] = self.gt[i]
        example['image_sizes'] = image_sizes
        example['has_image'] = True
        return example
    

class TestDatasetForMultiRec(Dataset):
    def __init__(
        self,
        dataset_root,   # /workspace/data/MulBench
        concept_names,  # ["sks1", "sks2"]
        device,
        config,
        image_processor,
        more_negative,
        num_neg = 100,
    ):
        random.seed(1)
        self.dataset_root = dataset_root
        self.device = device
        self.config = config
        self.image_processor = image_processor
        self.concept_names = concept_names
        self.data_w_case = data_w_case
        self.more_negative = more_negative

        # Load positive concept images 
        # types = ["si_sq", "mi_sq", "mi_mq"]
        pos_image_paths = []
        pos_quary_names = []
        types_query = []
        
        for concept_name in concept_names:
            tmp_single_image_paths = get_test_image_paths_by_concept(self.dataset_root, concept_name)
            pos_quary_names += [f'{concept_name}'] * len(tmp_single_image_paths)
            types_query += ["si_sq"] * len(tmp_single_image_paths)
            pos_image_paths += tmp_single_image_paths
            
            _, tmp_multi_image_paths = get_test_multi_image_paths_by_concept(self.dataset_root, concept_name)
            pos_quary_names += [f'{concept_name}'] * len(tmp_multi_image_paths)
            types_query += ["mi_sq"] * len(tmp_multi_image_paths)
            pos_image_paths += tmp_multi_image_paths
            
        pos_image_paths += tmp_multi_image_paths
        pos_quary_names += [f"{'_'.join(self.concept_names)}"] * len(tmp_multi_image_paths)
        types_query += ["mi_mq"] * len(tmp_multi_image_paths)
        
        assert len(pos_image_paths) == len(pos_quary_names), "Num of images and queries should be same"
        assert len(pos_image_paths) == len(types_query), "Num of images and queries should be same"
        
        assert len(pos_image_paths) == len(concept_names) * 5 + len(concept_names) * 5 + 5, f"So far, num of positive images should be 5 * m + 5 * m + 5, but got {len(pos_image_paths)}"
        
        # Load negative images
        num_single_neg, num_multi_neg = num_neg // 2, num_neg // 2
        neg_image_paths = []
        neg_quary_names = []
        
        # Load joint negative images
        for curr_concept_name in concept_names:
            for other_concept_name in concept_names:
                if curr_concept_name == other_concept_name:
                    continue
                temp_neg_image_paths = get_test_image_paths_by_concept(self.dataset_root, other_concept_name)
                neg_image_paths += temp_neg_image_paths
                types_query += ["si_sq"] * len(temp_neg_image_paths)
                neg_quary_names += [f'{curr_concept_name}'] * len(temp_neg_image_paths)
        
        assert len(neg_image_paths) == len(neg_quary_names), "Num of images and queries should be same"
        assert len(neg_image_paths) == len(concept_names) * (len(concept_names) - 1) * 5, f"So far, num of negative images should be 5 * m * (m - 1), but got {len(neg_image_paths)}"
        
        if self.more_negative:
            for _ in range(num_single_neg):
                neg_concept_name = random.choice(list(self.data_w_case.keys()))
                while neg_concept_name in concept_names:
                    neg_concept_name = random.choice(list(self.data_w_case.keys()))
                
                neg_image_paths.append(random.choice(get_test_image_paths_by_concept(self.dataset_root, neg_concept_name)))
                neg_quary_names.append(random.choice(concept_names))
                types_query.append("si_sq")
            
            for _ in range(num_multi_neg):
                neg_concept_name = random.choice(list(self.data_w_case.keys()))
                while neg_concept_name in concept_names:
                    neg_concept_name = random.choice(list(self.data_w_case.keys()))
                
                neg_image_paths.append(random.choice(get_test_multi_image_paths_by_concept(self.dataset_root, neg_concept_name)[1]))
                ask_multi_or_not = random.choice([True, False])
                if not ask_multi_or_not:
                    neg_quary_names.append(random.choice(concept_names))
                    types_query.append("mi_sq")
                else:
                    neg_quary_names.append(f"{'_'.join(self.concept_names)}")
                    types_query.append("mi_mq")
                    
        self.image_paths = pos_image_paths + neg_image_paths
        self.quary_names = pos_quary_names + neg_quary_names
        
        assert len(self.image_paths) == len(self.quary_names), "Num of images and queries should be same"
        
        self.gt = ['Yes'] * len(pos_image_paths) + ['No'] * len(neg_image_paths)
        self.types_query = types_query
        self.num_images = len(self.image_paths)
        self._length = self.num_images

    def __len__(self):
        return self._length
    
    def __getitem__(self, i):
        example = {}
        image_path = self.image_paths[i]      
        
        images = [Image.open(image_path).convert("RGB")]
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            self.image_processor,
            self.config
        ).to(self.device)
        example["images"] = images_tensor
        quary_name = self.quary_names[i]
        
        if "_" not in quary_name:
            example["query"] = f'Can you see <{quary_name}> in this photo? Answer the question using a single word Yes or No.'
        else:
            concept_names = quary_name.split("_")
            concept_names = [f'<{x}>' for x in concept_names]
            example["query"] = f'Can you see {" and ".join(concept_names)} in this photo? Answer the question using a single word Yes or No.'
         
        example["type"] = self.types_query[i]
        example["answer"] = self.gt[i]
        example['image_sizes'] = image_sizes
        example['has_image'] = True
        return example
    

class TestDatasetForMultiVQA(Dataset):
    def __init__(
        self,
        dataset_root,   # /workspace/data/MulBench
        concept_names,  # ["sks1", "sks2"]
        device,
        config,
        image_processor,
        type = "vqa",
    ):
        self.dataset_root = dataset_root
        self.device = device
        self.config = config
        self.image_processor = image_processor
        self.concept_names = concept_names
        self.data_w_case = data_w_case
        self.type = type

        # Load positive concept images 
        # types = ["si_sq", "mi_sq", "mi_mq"]
        # type = ["vqa", "choice"]
        
        self.image_paths = []
        self.quaries = []
        self.answers = []
        self.types_query = []
        
        for concept_name in concept_names:
            concept_case = self.data_w_case[concept_name]
            qa_file = os.path.join(self.dataset_root, concept_case, "concept/test", concept_name, f"{self.type}.json")
            try:
                with open(qa_file) as f:
                    qa_data = json.load(f)
            except:
                raise ValueError(f"Failed to load {qa_file}")
            
            for image_path, qa in qa_data.items():
                self.image_paths.append(image_path)
                self.quaries.append(qa['Human'])
                self.answers.append(qa['AI'])
                self.types_query.append("si_sq")
        
        assert len(self.image_paths) == len(self.quaries), "Num of images and queries should be same"
        assert len(self.image_paths) == len(self.answers), "Num of images and answers should be same"
        assert len(self.image_paths) == len(self.concept_names) * 5, f"So far, num of images should be 5 * m, but got {len(self.image_paths)}"
        
        # Load multi
        concept_case = self.data_w_case[self.concept_names[0]]
        multi_cases_dir = os.path.join(self.dataset_root, concept_case, "multi")
        self.case = None
        for case in os.listdir(multi_cases_dir):
            if not os.path.isdir(os.path.join(multi_cases_dir, case)):
                continue
            if self.concept_names[0] in case:
                self.case = case
                break
        if self.case is None:
            raise ValueError(f"Multi case not found for {self.concept_names[0]}")
        
        curr_case_qa_file = os.path.join(multi_cases_dir, self.case, f"{self.type}.json")
        try:
            with open(curr_case_qa_file) as f:
                curr_case_qa_data = json.load(f)
        except:
            raise ValueError(f"Failed to load {curr_case_qa_file}")
        
        for image_path, qa_pairs in curr_case_qa_data.items():
            for i, qa in enumerate(qa_pairs):
                self.image_paths.append(image_path)
                self.quaries.append(qa['Human'])
                self.answers.append(qa['AI'])
                if i < 2:
                    self.types_query.append("mi_sq")
                else:
                    self.types_query.append("mi_mq")
        
        assert len(self.image_paths) == len(self.quaries), "Num of images and queries should be same"
        assert len(self.image_paths) == len(self.answers), "Num of images and answers should be same"
        # assert len(self.image_paths) == len(self.concept_names) * 5 + (1 + len(self.concept_names)) * 5, f"So far, num of images should be 5 * m + (1 + len(self.concept_names)) * 5, but got {len(self.image_paths)}"
        
        self.num_images = len(self.image_paths)
        self._length = self.num_images

    def __len__(self):
        return self._length
    
    def __getitem__(self, i):
        example = {}
        image_path = self.image_paths[i]      
        
        images = [Image.open(image_path).convert("RGB")]
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            self.image_processor,
            self.config
        ).to(self.device)
        example["images"] = images_tensor
        if self.type == "vqa":
            example["query"] = self.quaries[i]
        elif self.type == "choice":
            example["query"] = self.quaries[i] + "\nIMPORTANT: Answer the question using a single letter A, B or C."
        else:
            raise ValueError(f"Unsupported type: {self.type}")
        example["answer"] = self.answers[i]
        example["type"] = self.types_query[i]
        example['image_sizes'] = image_sizes
        example['has_image'] = True
        return example
    

class TestDatasetForCaptionGeneration(Dataset):
    def __init__(
        self,
        dataset_root,   # /workspace/data/MulBench
        concept_names,  # ["sks1", "sks2"]
        device,
        config,
        image_processor,
    ):
        self.dataset_root = dataset_root
        self.device = device
        self.config = config
        self.image_processor = image_processor
        self.concept_names = concept_names
        self.data_w_case = data_w_case

        # Load positive concept images 
        # types = ["si", "mi"]
        
        self.image_paths = []
        self.quaries = []
        self.types_query = []
        
        # Load single test images path
        for concept_name in concept_names:
            concept_single_image_paths = get_test_image_paths_by_concept(self.dataset_root, concept_name)
            self.image_paths += concept_single_image_paths
            self.quaries += [f'<{concept_name}>'] * len(concept_single_image_paths)
            self.types_query += ["si"] * len(concept_single_image_paths)
        
        assert len(self.image_paths) == len(self.quaries), "Num of images and queries should be same"
        
        # Load multi test images path
        concept_multi_image_paths = get_test_multi_image_paths_by_concept(self.dataset_root, concept_names[0])[1]
        
        # xxxxx/0.png  -> xxxxx/position.json
        position_json_file = os.path.join(os.path.dirname(concept_multi_image_paths[0]), "position.json")
        try:
            with open(position_json_file) as f:
                position_data = json.load(f)
        except:
            raise ValueError(f"Failed to load {position_json_file}")

        self.existing_concept = []
        for image, positions in position_data.items():
            non_null_keys = [key for key, value in positions.items() if value is not None]
            self.existing_concept.append(non_null_keys)
            
        self.image_paths += concept_multi_image_paths
        self.quaries += self.existing_concept
        self.types_query += ["mi"] * len(concept_multi_image_paths)
        
        assert len(self.image_paths) == len(self.quaries), "Num of images and queries should be same"
        
        self.num_images = len(self.image_paths)
        self._length = self.num_images
        
    def __len__(self):
        return self._length
    
    def __getitem__(self, i):
        example = {}
        image_path = self.image_paths[i]      
        
        images = [Image.open(image_path).convert("RGB")]
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            self.image_processor,
            self.config
        ).to(self.device)
        example["images"] = images_tensor
        example["query"] = self.quaries[i]
        
        example["type"] = self.types_query[i]
        example['image_sizes'] = image_sizes
        example['has_image'] = True
        return example
    

class TestDatasetForTextOnlyQAChoice(Dataset):
    def __init__(
        self,
        dataset_root,   # /workspace/data/MulBench
        concept_names,  # ["sks1", "sks2"]
    ):
        self.dataset_root = dataset_root
        self.concept_names = concept_names
        self.data_w_case = data_w_case

        # Load positive concept images 
        # types = ["s", "m"]
        
        self.quaries = []
        self.ansewers = []
        self.types_query = []
        
        # Load single 
        for concept_name in self.concept_names:
            concept_case = self.data_w_case[concept_name]
            qa_file = os.path.join(self.dataset_root, concept_case, "concept/test", concept_name, "qa.json")
            try:
                with open(qa_file) as f:
                    qa_data = json.load(f)
            except:
                raise ValueError(f"Failed to load {qa_file}")
            
            for qa in qa_data:
                self.quaries.append(qa['question'])
                self.ansewers.append(qa['answer'])
                self.types_query.append("s")
        
        assert len(self.ansewers) == len(self.quaries), "Num of images and queries should be same"
        
        # Load multi 
        concept_case = self.data_w_case[self.concept_names[0]]
        cases_dir = os.path.join(self.dataset_root, concept_case, "multi")
        for case in os.listdir(cases_dir):
            if not os.path.isdir(os.path.join(cases_dir, case)):
                continue
            if self.concept_names[0] in case:
                qa_file = os.path.join(cases_dir, case, "qa.json")
                try:
                    with open(qa_file) as f:
                        qa_data = json.load(f)
                except:
                    raise ValueError(f"Failed to load {qa_file}")
                
                for qa in qa_data:
                    self.quaries.append(qa['question'])
                    self.ansewers.append(qa['answer'])
                    self.types_query.append("m")
                break
            
        assert len(self.ansewers) == len(self.quaries), "Num of images and queries should be same"
        
        self._length = len(self.quaries)
        
    def __len__(self):
        return self._length
    
    def __getitem__(self, i):
        example = {}
        example["query"] = self.quaries[i] + "\nIMPORTANT: Answer the question using a single letter A, B or C."
        example["type"] = self.types_query[i]
        example["answer"] = self.ansewers[i]
        example['has_image'] = False
        return example


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def get_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )
    return tokenizer, model, image_processor, context_len

def get_model_(model_path):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name
    )
    return tokenizer, model, image_processor, context_len


def get_query(args, query, model, sks_system_prompt=None):
    qs = query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    model_name = get_model_name_from_path(args.model_path)
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode
    
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    if sks_system_prompt is not None:
        conv.system = conv.system + " " + sks_system_prompt
    args.conv = conv
    return args


def process_masks(masks):
    processed_masks = []
    for mask in masks:
        mask = expand2square(mask, 0)
        mask = mask.resize((336, 336), Image.NEAREST)
        mask_array = np.array(mask)
        mask_tensor = torch.tensor(mask_array, dtype=torch.uint8) // 255
        mask_tensor = mask_tensor.unsqueeze(0)
        
        processed_masks.append(mask_tensor)
    
    return torch.cat(processed_masks, dim=0)


def custom_collate_fn(batch):
    return batch

if __name__ == "__main__":
    model_path = "liuhaotian/llava-v1.6-vicuna-13b"
    tokenizer, model, image_processor, context_len = get_model_(model_path)
    
    cases_file = "/workspace/data/MulBench/cases.json"
    with open(cases_file) as f:
        cases = json.load(f)
    num_2 = 0
    num_3 = 0
    num_4 = 0
    
    for case in cases:
        concepts = case.split("_")
        dataset_0 = TestDatasetForTextOnlyQAChoice(dataset_root, concepts)
        dataset_1 = TestDatasetForMultiVQA(dataset_root, concepts, model.device, model.config, image_processor, "choice")
        dataset_2 = TestDatasetForMultiVQA(dataset_root, concepts, model.device, model.config, image_processor, "vqa")
        
        if len(concepts) == 2:
            num_2 += len(dataset_0)
            num_2 += len(dataset_1)
            num_2 += len(dataset_2)
        elif len(concepts) == 3:
            num_3 += len(dataset_0)
            num_3 += len(dataset_1)
            num_3 += len(dataset_2)
        elif len(concepts) == 4:
            num_4 += len(dataset_0)
            num_4 += len(dataset_1)
            num_4 += len(dataset_2)
        else:
            raise ValueError(f"Unsupported number of concepts: {len(concepts)}")
        
    print(f"Num of cases with 2 concepts: {num_2}")
    print(f"Num of cases with 3 concepts: {num_3}")
    print(f"Num of cases with 4 concepts: {num_4}")
