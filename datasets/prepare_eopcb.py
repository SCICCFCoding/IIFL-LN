import argparse
import os
import shutil
import random
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from datasets_utils import create_directories, archive_directories
except ImportError:
    def create_directories(target_path, nclients):
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        for subdict in ['', '/images', '/labels']:
            if not os.path.exists(f'{target_path}/server{subdict}'):
                os.makedirs(f'{target_path}/server{subdict}')
        for k in range(1, nclients + 1):
            for subdict in ['', '/images', '/labels']:
                if not os.path.exists(f'{target_path}/client{k}{subdict}'):
                    os.makedirs(f'{target_path}/client{k}{subdict}')
    
    def archive_directories(target_path, nclients):
        import tarfile
        server_path = os.path.join(target_path, 'server')
        tar_file_name = os.path.join(target_path, 'server.tar')
        with tarfile.open(tar_file_name, 'w') as tar_handle:
            tar_handle.add(server_path, arcname='server')
        for k in range(1, nclients + 1):
            client_path = os.path.join(target_path, f'client{k}')
            tar_file_name = os.path.join(target_path, f'client{k}.tar')
            with tarfile.open(tar_file_name, 'w') as tar_handle:
                tar_handle.add(client_path, arcname=f'client{k}')
ORIGINAL_CLASS_MAP = {
    '0': 'missing_hole',   
    '1': 'short',          
    '2': 'mouse_bite',     
    '3': 'open_circuit',   
    '4': 'spurious_copper', 
    '5': 'spur'            
}


TARGET_CLASS_MAP = {
    'missing_hole': 0,    
    'short': 1,          
    'mouse_bite': 2,     
    'spur': 3,           
    'spurious_copper': 4, 
    'open_circuit': 5    
}

FOLDER_MAP = {
    'missing_hole': 'Missing_hole',
    'short': 'Short',
    'mouse_bite': 'Mouse_bite',
    'spur': 'Spur',
    'spurious_copper': 'Spurious_copper',
    'open_circuit': 'Open_circuit',
    'missing_hole_create': 'Missing_hole_create',
    'short_create': 'Short_create',
    'mouse_bite_create': 'Mouse_bite_create', 
    'spur_create': 'Spur_create',
    'spurious_copper_create': 'Spurious_copper_create',
    'open_circuit_create': 'Open_circuit_create'
}

CLASS_RELATION = {
    'missing_hole_create': 'missing_hole',
    'short_create': 'short',
    'mouse_bite_create': 'mouse_bite',
    'spur_create': 'spur',
    'spurious_copper_create': 'spurious_copper',
    'open_circuit_create': 'open_circuit'
}

def get_defect_type_from_filename(filename):
    type_priorities = [
        ("spurious_copper", "spurious_copper"),  
        ("missing_hole", "missing_hole"),
        ("open_circuit", "open_circuit"),
        ("mouse_bite", "mouse_bite"),
        ("short", "short"),
        ("spur", "spur")
    ]
    
    for keyword, defect_type in type_priorities:
        if keyword in filename:
            return defect_type
            
    return None

def get_folder_from_filename(filename):
    defect_type = get_defect_type_from_filename(filename)
    if defect_type is None:
        return None
        
    is_created = "_create" in filename
    if is_created:
        return FOLDER_MAP[f"{defect_type}_create"]
    else:
        return FOLDER_MAP[defect_type]

def get_distribution_dataframe(nclients):
    columns = ['server']
    for k in range(1, nclients + 1):
        columns.append(f'client{k}')

    classes = list(TARGET_CLASS_MAP.keys())
    objects_distribution = pd.DataFrame(0, columns=columns, index=['Samples'] + classes)
    return objects_distribution

def get_iid_splits(file_list, nclients, val_frac):
    random.seed(42)  
    total_files = len(file_list)
    client_frac = (1 - val_frac) / nclients

    available_files = file_list.copy()
    random.shuffle(available_files)
    
    splits = {}
    client_split_size = int(total_files * client_frac)

    for k in range(1, nclients + 1):
        client_data = available_files[:client_split_size]
        available_files = available_files[client_split_size:]
        
        for file in client_data:
            splits[file] = f'client{k}'

    for file in available_files:
        splits[file] = 'server'
        
    return splits

def find_image_for_label(source_path, label_filename, all_images):
    label_prefix = label_filename.rsplit('.', 1)[0]
    
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_filename = f"{label_prefix}{ext}"
        
        if image_filename in all_images:
            image_path, folder = all_images[image_filename]
            return image_path, image_filename, folder
    
    for img_name, (img_path, folder) in all_images.items():
        if label_prefix in img_name:
            return img_path, img_name, folder

    with open(os.path.join(source_path, 'labels', label_filename), 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = parts[0]
                if class_id in ORIGINAL_CLASS_MAP:
                    defect_type = ORIGINAL_CLASS_MAP[class_id]
                    is_created = "_create" in label_prefix
                    folder_to_search = None
                    if is_created:
                        folder_to_search = FOLDER_MAP[f"{defect_type}_create"]
                    else:
                        folder_to_search = FOLDER_MAP[defect_type]

                    for img_name, (img_path, folder) in all_images.items():
                        if folder == folder_to_search:
                            parts = label_prefix.split('_')
                            if len(parts) >= 3:
                                prefix_pattern = f"{parts[0]}_{defect_type}_{parts[-1]}"
                                if prefix_pattern in img_name:
                                    return img_path, img_name, folder
                    for img_name, (img_path, folder) in all_images.items():
                        if folder == folder_to_search:
                            return img_path, img_name, folder
                    
                    break  
    
    return None, None, None

def process_eopcb(source_path, target_path, data, nclients, val_frac, tar):
    print('Processing E-O-PCB dataset and splitting...')
    
    create_directories(target_path, nclients)
    
    labels_path = os.path.join(source_path, 'labels')
    label_files = [f for f in os.listdir(labels_path) if f.endswith('.txt')]
    
    print(f"Found {len(label_files)} label files")
    print("Indexing all image files...")
    all_images = {}  
    
    for folder in FOLDER_MAP.values():
        folder_path = os.path.join(source_path, 'images', folder)
        if os.path.exists(folder_path):
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files = glob.glob(os.path.join(folder_path, ext))
                for img_path in image_files:
                    img_name = os.path.basename(img_path)
                    all_images[img_name] = (img_path, folder)
    
    print(f"Found {len(all_images)} image files")
    
    splits = get_iid_splits(label_files, nclients, val_frac)
    
    objects_distribution = get_distribution_dataframe(nclients)
    
    class_counts = {cls: 0 for cls in TARGET_CLASS_MAP.keys()}
    processed_count = 0
    skipped_count = 0
    matched_files = []
    
    for label_file in tqdm(label_files):
        destination = splits[label_file]
        
        image_path, image_filename, image_folder = find_image_for_label(source_path, label_file, all_images)
        
        if image_path is None:
            print(f"Warning: Cannot find matching image for label file {label_file}, skipping")
            skipped_count += 1
            continue
        
        target_image_path = os.path.join(target_path, destination, 'images', image_filename)
        shutil.copyfile(image_path, target_image_path)

        source_label_path = os.path.join(labels_path, label_file)
        target_label_path = os.path.join(target_path, destination, 'labels', label_file)

        valid_label = False
        with open(source_label_path, 'r') as source_file, open(target_label_path, 'w') as target_file:
            for line in source_file:
                parts = line.strip().split()
                if len(parts) >= 5:  
                    orig_class_id, x, y, w, h = parts[:5]
                    
                    if orig_class_id in ORIGINAL_CLASS_MAP:
                        defect_type = ORIGINAL_CLASS_MAP[orig_class_id]
                        label_prefix = label_file.rsplit('.', 1)[0]
                        if "_create" in label_prefix:
                            defect_type_with_create = f"{defect_type}_create"
                            if defect_type_with_create in CLASS_RELATION:
                                defect_type = CLASS_RELATION[defect_type_with_create]

                        if defect_type in TARGET_CLASS_MAP:
                            new_class_id = TARGET_CLASS_MAP[defect_type]

                            objects_distribution.loc['Samples', destination] += 1
                            objects_distribution.loc[defect_type, destination] += 1

                            class_counts[defect_type] += 1
                            valid_label = True

                            target_file.write(f"{new_class_id} {x} {y} {w} {h}\n")
                        else:
                            print(f"Warning: Cannot map defect type: {defect_type}, skipping this line")
                    else:
                        print(f"Warning: Unrecognized class ID: {orig_class_id}, keeping as is")
                        target_file.write(line)
                        valid_label = True
        
        if valid_label:
            processed_count += 1
            matched_files.append((label_file, image_filename, image_folder))
        else:
            os.remove(target_image_path)
            os.remove(target_label_path)
            skipped_count += 1
    
    print(f"\nSuccessfully processed {processed_count} label-image pairs, skipped {skipped_count} invalid labels")
    
    print("\nClass distribution statistics:")
    print(objects_distribution)
    
    print("\nClass processing counts:")
    for cls, count in class_counts.items():
        print(f"{cls}: {count}")
    
    matches_csv_path = os.path.join(target_path, 'matched_files.csv')
    with open(matches_csv_path, 'w') as matches_file:
        matches_file.write("Label File,Image File,Image Folder\n")
        for label, image, folder in matched_files:
            matches_file.write(f"{label},{image},{folder}\n")
    
    print(f"\nMatched file list saved to {matches_csv_path}")
    
    objects_distribution.to_csv(f'{target_path}/objects_distribution.csv')

    if tar:
        print('Creating archive files...')
        archive_directories(target_path, nclients)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-path', type=str, default='datasets/E-O_PCB', help='Source path of E-O-PCB dataset')
    parser.add_argument('--target-path', type=str, default='datasets/eo-pcb', help='Target path of processed dataset')
    parser.add_argument('--data', type=str, default='data/eo-pcb.yaml', help='Path of data configuration file')
    parser.add_argument('--nclients', type=int, default=3, help='Number of clients in federated learning')
    parser.add_argument('--val-frac', type=float, default=0.25, help='Fraction of data reserved by server for validation')
    parser.add_argument('--tar', action='store_true', help='Whether to create archives for federated participant directories')
    args = parser.parse_args()
    process_eopcb(args.source_path, args.target_path, args.data, args.nclients, args.val_frac, args.tar) 