import os
import json
import random


def generate_positive_qa_pairs4concept(root_dir, concept, templates, out_put_root_dir, num_qa_pairs=5):
    concept_folder_path = os.path.join(root_dir, concept)
    if not os.path.isdir(concept_folder_path):
        return
    
    concept_image_files = [f for f in os.listdir(concept_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and '_mask' not in f]
    # a list containing the image file names in the concept training folder
    
    result = {}
    for concept_image in concept_image_files:
        image_path = os.path.join(concept_folder_path, concept_image)
        qa_pairs = []
        
        selected_templates = random.sample(templates, min(num_qa_pairs, len(templates)))
        
        for template in selected_templates:
            question = template["Question"].replace("<sks>", f"<{concept}>")
            answer = template["Answer"].replace("<sks>", f"<{concept}>")
            qa_pairs.append({
                "Human": question,
                "AI": answer
            })
        
        result[image_path] = qa_pairs

    output_dir = os.path.join(out_put_root_dir, 'generate_training_data', concept)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, 'recognition_positive.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    
    print(f"The generated file has been saved as: {output_file}")


def generate_positive_qa_pairs(root_dir, template_file, out_put_root_dir, num_qa_pairs=5):
    with open(template_file, 'r', encoding='utf-8') as f:
        templates = json.load(f)
    
    for concept in os.listdir(root_dir):
        generate_positive_qa_pairs4concept(root_dir, concept, templates, out_put_root_dir, num_qa_pairs)


def generate_negative_qa_pairs4concept(random_images, concept, root_dir, templates, out_put_root_dir, num_qa_pairs=100):
    concept_folder_path = os.path.join(root_dir, concept)
    if not os.path.isdir(concept_folder_path):
        return
    
    result = {}
    for random_image in random_images:
        random_image_path = os.path.join(random_image_dir, random_image)
        
        template = random.choice(templates)
        question = template["Question"].replace("<sks>", f"<{concept}>")
        answer = template["Answer"].replace("<sks>", f"<{concept}>")
        
        result[random_image_path] = [{
            "Human": question,
            "AI": answer
        }]
    
    output_dir = os.path.join(out_put_root_dir, 'generate_training_data', concept)
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'recognition_negative_cc12m.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    
    print(f"The generated file has been saved as: {output_file}")
    

def generate_strong_negtive_qa_pairs4concept(concept, root_dir, templates, out_put_root_dir, data_size, seed = 0):
    random.seed(seed)
    concept_folder_path = os.path.join(root_dir, concept)
    if not os.path.isdir(concept_folder_path):
        return
    
    strong_neg_dir = os.path.join(concept_folder_path, 'negative_example')
    strong_neg_image_files = [f for f in os.listdir(strong_neg_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and '_mask' not in f]
    # a list containing the image file names in the concept training folder
    strong_neg_image_files = random.sample(strong_neg_image_files, data_size)
    
    result = {}
    for neg_image in strong_neg_image_files:
        image_path = os.path.join(strong_neg_dir, neg_image)
        qa_pairs = []
        
        selected_templates = random.sample(templates, min(1, len(templates)))
        
        for template in selected_templates:
            question = template["Question"].replace("<sks>", f"<{concept}>")
            answer = template["Answer"].replace("<sks>", f"<{concept}>")
            qa_pairs.append({
                "Human": question,
                "AI": answer
            })
        
        result[image_path] = qa_pairs

    output_dir = os.path.join(out_put_root_dir, 'generate_training_data', concept)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, 'recognition_strong_negative.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    
    print(f"The generated file has been saved as: {output_file}")


def generate_negative_qa_pairs(random_image_dir, root_dir, template_file, out_put_root_dir, num_qa_pairs=100):

    with open(template_file, 'r', encoding='utf-8') as f:
        templates = json.load(f)
    
    random_images = [f for f in os.listdir(random_image_dir) if f.lower().endswith(('.png', '.jpeg', '.jpg'))]
    random_images = random.sample(random_images, min(num_qa_pairs, len(random_images)))
    
    for concept in os.listdir(root_dir):
        generate_negative_qa_pairs4concept(random_images, concept, root_dir, templates, out_put_root_dir, num_qa_pairs)


def generate_convosation4concept(root_dir, concept, out_put_root_dir):
    concept_folder_path = os.path.join(root_dir, concept)
    if not os.path.isdir(concept_folder_path):
        return

    qa_file = os.path.join(concept_folder_path, 'QA.json')
    if os.path.exists(qa_file):
        with open(qa_file, 'r') as f:
            qa_data = json.load(f)
        
        conversation = {}
        for image_file in os.listdir(concept_folder_path):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(concept_folder_path, image_file)
                conversation[image_path] = []
                
                for qa in qa_data:
                    conversation = {
                        "Human": qa["Human"],
                        "AI": qa["AI"]
                    }
                    conversation[image_path].append(conversation)
    else:
        print(f"No QA file found for {concept}")
        return

    output_dir = os.path.join(out_put_root_dir, 'generate_training_data', concept)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, 'conversation.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(conversation, f, ensure_ascii=False, indent=4)
    
    print(f"The generated file has been saved as: {output_file}")


def generate_conversation(root_dir, out_put_root_dir):
    for concept in os.listdir(root_dir):
        generate_convosation4concept(root_dir, concept, out_put_root_dir)


def generate_joint_qa_pairs(root_dir, concept_names, template_file, out_put_root_dir, num_qa_pairs=10):
    with open(template_file, 'r', encoding='utf-8') as f:
        templates = json.load(f)
    
    for concept in concept_names:
        curr_concept_path = os.path.join(root_dir, concept)
        curr_concept_images = [f for f in os.listdir(curr_concept_path) if f.lower().endswith(('.png', '.jpeg', '.jpg')) and '_mask' not in f]
        
        result = {}
        
        for curr_concept_image in curr_concept_images:
            image_path = os.path.join(curr_concept_path, curr_concept_image)
            qa_pairs = []
            
            selected_templates = random.sample(templates, min(num_qa_pairs, len(templates)))
            
            for template in selected_templates:
                question = template["Question"]
                answer = template["Answer"]  
                qa_pairs.append({
                    "Human": question,
                    "AI": answer
                })
                
            result[image_path] = qa_pairs
        
        for other_concept in concept_names:
            if other_concept == concept:
                continue
            new_result = {}
            for img, qa_pairs in result.items():
                new_qa_pairs = []
                for qa_pair in qa_pairs:
                    question = qa_pair["Human"].replace(f"<sks>", f"<{other_concept}>")
                    answer = qa_pair["AI"].replace(f"<sks>", f"<{other_concept}>")
                    new_qa_pairs.append({
                        "Human": question,
                        "AI": answer
                    })
                new_result[img] = new_qa_pairs
            output_dir = os.path.join(out_put_root_dir, other_concept)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f'recognition_negative_{concept}.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(new_result, f, ensure_ascii=False, indent=4)
        
            print(f"The generated file has been saved as: {output_file}")
            
    
def generate_strong_negtive_qa_pairs(root_dir, template_file, out_put_root_dir, data_size):
    with open(template_file, 'r', encoding='utf-8') as f:
        templates = json.load(f)
    
    for concept in os.listdir(root_dir):
        generate_strong_negtive_qa_pairs4concept(concept, root_dir, templates, out_put_root_dir, data_size)

if __name__ == "__main__":
    pass
