# FedPylot 修改版 - E-O-PCB 数据集处理脚本
# 该脚本将E-O-PCB数据集处理成联邦学习所需的格式，按客户端分割数据

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

# 添加datasets目录到系统路径，确保可以导入datasets_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from datasets_utils import create_directories, archive_directories
except ImportError:
    # 如果无法导入，提供必要的函数实现
    def create_directories(target_path, nclients):
        """创建目录结构"""
        # 创建目标目录
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        # 创建服务器子目录
        for subdict in ['', '/images', '/labels']:
            if not os.path.exists(f'{target_path}/server{subdict}'):
                os.makedirs(f'{target_path}/server{subdict}')
        # 创建客户端子目录
        for k in range(1, nclients + 1):
            for subdict in ['', '/images', '/labels']:
                if not os.path.exists(f'{target_path}/client{k}{subdict}'):
                    os.makedirs(f'{target_path}/client{k}{subdict}')
    
    def archive_directories(target_path, nclients):
        """将目录打包成tar文件"""
        import tarfile
        # 打包服务器目录
        server_path = os.path.join(target_path, 'server')
        tar_file_name = os.path.join(target_path, 'server.tar')
        with tarfile.open(tar_file_name, 'w') as tar_handle:
            tar_handle.add(server_path, arcname='server')
        # 打包客户端目录
        for k in range(1, nclients + 1):
            client_path = os.path.join(target_path, f'client{k}')
            tar_file_name = os.path.join(target_path, f'client{k}.tar')
            with tarfile.open(tar_file_name, 'w') as tar_handle:
                tar_handle.add(client_path, arcname=f'client{k}')

# 根据样本标签文件更新类别映射
# 这里将使用数据集中实际使用的类别ID，并在处理时进行映射
ORIGINAL_CLASS_MAP = {
    # 根据标签文件中的实际ID进行映射
    '0': 'missing_hole',   # 缺失孔
    '1': 'short',          # 短路
    '2': 'mouse_bite',     # 鼠咬
    '3': 'open_circuit',   # 断路
    '4': 'spurious_copper', # 多余铜箔
    '5': 'spur'            # 毛刺
}

# 我们希望使用的新类别ID - 确保与data/eo-pcb.yaml中的顺序一致
TARGET_CLASS_MAP = {
    'missing_hole': 0,    # 缺失孔
    'short': 1,          # 短路
    'mouse_bite': 2,     # 鼠咬
    'spur': 3,           # 毛刺
    'spurious_copper': 4, # 多余铜箔
    'open_circuit': 5    # 断路
}

# 确定类别名称与文件夹的映射关系
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

# 定义类别的关系 - 将create类型关联到基本类型
CLASS_RELATION = {
    'missing_hole_create': 'missing_hole',
    'short_create': 'short',
    'mouse_bite_create': 'mouse_bite',
    'spur_create': 'spur',
    'spurious_copper_create': 'spurious_copper',
    'open_circuit_create': 'open_circuit'
}

def get_defect_type_from_filename(filename):
    """从文件名提取缺陷类型"""
    # 如果文件名包含多个类型关键字，使用优先级来确定类型
    # 以防止错误匹配（例如：文件名中包含了spurious_copper但实际是spur类型）
    type_priorities = [
        ("spurious_copper", "spurious_copper"),  # 优先检查更长/更具体的名称
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
    """根据文件名推断所在文件夹"""
    # 分析文件名中的类型标识符
    defect_type = get_defect_type_from_filename(filename)
    if defect_type is None:
        return None
        
    is_created = "_create" in filename
    if is_created:
        return FOLDER_MAP[f"{defect_type}_create"]
    else:
        return FOLDER_MAP[defect_type]

def get_distribution_dataframe(nclients):
    """创建用于跟踪对象分布的DataFrame"""
    columns = ['server']
    for k in range(1, nclients + 1):
        columns.append(f'client{k}')
    
    # 只使用基本类别（不包括_create版本）
    classes = list(TARGET_CLASS_MAP.keys())
    objects_distribution = pd.DataFrame(0, columns=columns, index=['Samples'] + classes)
    return objects_distribution

def get_iid_splits(file_list, nclients, val_frac):
    """将数据集按IID（独立同分布）方式分割给多个客户端"""
    random.seed(42)  # 固定随机种子以确保结果可重复
    total_files = len(file_list)
    client_frac = (1 - val_frac) / nclients
    
    # 创建一个副本，避免修改原始列表
    available_files = file_list.copy()
    random.shuffle(available_files)
    
    splits = {}
    client_split_size = int(total_files * client_frac)
    
    # 为每个客户端分配数据
    for k in range(1, nclients + 1):
        client_data = available_files[:client_split_size]
        available_files = available_files[client_split_size:]
        
        for file in client_data:
            splits[file] = f'client{k}'
    
    # 剩余的分配给服务器
    for file in available_files:
        splits[file] = 'server'
        
    return splits

def find_image_for_label(source_path, label_filename, all_images):
    """为标签文件找到匹配的图像文件"""
    # 从标签文件名去掉扩展名
    label_prefix = label_filename.rsplit('.', 1)[0]
    
    # 尝试常见的图像扩展名
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_filename = f"{label_prefix}{ext}"
        
        # 1. 直接检查是否有完全匹配的图像文件
        if image_filename in all_images:
            image_path, folder = all_images[image_filename]
            return image_path, image_filename, folder
    
    # 2. 如果没有完全匹配，尝试部分匹配
    for img_name, (img_path, folder) in all_images.items():
        if label_prefix in img_name:
            return img_path, img_name, folder
    
    # 3. 从标签获取类型，尝试在正确的文件夹中查找类似文件
    with open(os.path.join(source_path, 'labels', label_filename), 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = parts[0]
                if class_id in ORIGINAL_CLASS_MAP:
                    defect_type = ORIGINAL_CLASS_MAP[class_id]
                    is_created = "_create" in label_prefix
                    
                    # 确定要搜索的文件夹
                    folder_to_search = None
                    if is_created:
                        folder_to_search = FOLDER_MAP[f"{defect_type}_create"]
                    else:
                        folder_to_search = FOLDER_MAP[defect_type]
                    
                    # 尝试找到该文件夹中任何包含类似前缀的图像
                    for img_name, (img_path, folder) in all_images.items():
                        if folder == folder_to_search:
                            # 提取标签前缀的关键部分
                            parts = label_prefix.split('_')
                            if len(parts) >= 3:
                                # 尝试匹配数字部分+类型+编号
                                prefix_pattern = f"{parts[0]}_{defect_type}_{parts[-1]}"
                                if prefix_pattern in img_name:
                                    return img_path, img_name, folder
                    
                    # 如果还是找不到，尝试任何与该类型相关的图像
                    for img_name, (img_path, folder) in all_images.items():
                        if folder == folder_to_search:
                            return img_path, img_name, folder
                    
                    break  # 只检查第一行
    
    # 找不到任何匹配的图像
    return None, None, None

def process_eopcb(source_path, target_path, data, nclients, val_frac, tar):
    """处理E-O-PCB数据集并按照联邦学习格式分割"""
    print('处理E-O-PCB数据集并分割...')
    
    # 创建目录结构
    create_directories(target_path, nclients)
    
    # 获取所有标签文件
    labels_path = os.path.join(source_path, 'labels')
    label_files = [f for f in os.listdir(labels_path) if f.endswith('.txt')]
    
    print(f"找到 {len(label_files)} 个标签文件")
    
    # 预先索引所有图像文件
    print("索引所有图像文件...")
    all_images = {}  # 文件名 -> (完整路径, 文件夹)
    
    for folder in FOLDER_MAP.values():
        folder_path = os.path.join(source_path, 'images', folder)
        if os.path.exists(folder_path):
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files = glob.glob(os.path.join(folder_path, ext))
                for img_path in image_files:
                    img_name = os.path.basename(img_path)
                    all_images[img_name] = (img_path, folder)
    
    print(f"找到 {len(all_images)} 个图像文件")
    
    # 获取数据分割
    splits = get_iid_splits(label_files, nclients, val_frac)
    
    # 创建分布表
    objects_distribution = get_distribution_dataframe(nclients)
    
    # 创建计数器用于诊断
    class_counts = {cls: 0 for cls in TARGET_CLASS_MAP.keys()}
    processed_count = 0
    skipped_count = 0
    matched_files = []
    
    # 处理每个标签文件
    for label_file in tqdm(label_files):
        # 确定目的地
        destination = splits[label_file]
        
        # 为标签文件查找匹配的图像
        image_path, image_filename, image_folder = find_image_for_label(source_path, label_file, all_images)
        
        if image_path is None:
            print(f"警告: 无法为标签文件 {label_file} 找到匹配的图像，跳过")
            skipped_count += 1
            continue
        
        # 复制图像到目的地
        target_image_path = os.path.join(target_path, destination, 'images', image_filename)
        shutil.copyfile(image_path, target_image_path)
        
        # 复制标签到目的地
        source_label_path = os.path.join(labels_path, label_file)
        target_label_path = os.path.join(target_path, destination, 'labels', label_file)
        
        # 读取并处理标签内容
        valid_label = False
        with open(source_label_path, 'r') as source_file, open(target_label_path, 'w') as target_file:
            for line in source_file:
                parts = line.strip().split()
                if len(parts) >= 5:  # 确保有足够的字段
                    orig_class_id, x, y, w, h = parts[:5]
                    
                    # 映射原始类别ID到目标类别ID
                    if orig_class_id in ORIGINAL_CLASS_MAP:
                        defect_type = ORIGINAL_CLASS_MAP[orig_class_id]
                        
                        # 如果是_create版本，获取对应的基本类型
                        label_prefix = label_file.rsplit('.', 1)[0]
                        if "_create" in label_prefix:
                            defect_type_with_create = f"{defect_type}_create"
                            if defect_type_with_create in CLASS_RELATION:
                                # 确保使用基本类型的类别ID
                                defect_type = CLASS_RELATION[defect_type_with_create]
                        
                        # 确保类别存在于TARGET_CLASS_MAP中
                        if defect_type in TARGET_CLASS_MAP:
                            new_class_id = TARGET_CLASS_MAP[defect_type]
                            
                            # 更新对象分布（始终使用基本类型更新分布）
                            objects_distribution.loc['Samples', destination] += 1
                            objects_distribution.loc[defect_type, destination] += 1
                            
                            # 更新类别计数
                            class_counts[defect_type] += 1
                            valid_label = True
                            
                            # 使用新的类别ID写入标签
                            target_file.write(f"{new_class_id} {x} {y} {w} {h}\n")
                        else:
                            print(f"警告: 无法映射缺陷类型: {defect_type}，跳过此行")
                    else:
                        # 如果无法识别类别ID，保持原样
                        print(f"警告: 无法识别的类别ID: {orig_class_id}，保持原样")
                        target_file.write(line)
                        valid_label = True
        
        if valid_label:
            processed_count += 1
            matched_files.append((label_file, image_filename, image_folder))
        else:
            # 如果标签无效，删除已复制的文件
            os.remove(target_image_path)
            os.remove(target_label_path)
            skipped_count += 1
    
    # 打印处理统计信息
    print(f"\n成功处理 {processed_count} 个标签-图像对，跳过 {skipped_count} 个无效标签")
    
    # 打印分布统计信息
    print("\n类别分布统计:")
    print(objects_distribution)
    
    # 打印类别计数
    print("\n类别处理计数:")
    for cls, count in class_counts.items():
        print(f"{cls}: {count}")
    
    # 保存匹配文件列表
    matches_csv_path = os.path.join(target_path, 'matched_files.csv')
    with open(matches_csv_path, 'w') as matches_file:
        matches_file.write("标签文件,图像文件,图像文件夹\n")
        for label, image, folder in matched_files:
            matches_file.write(f"{label},{image},{folder}\n")
    
    print(f"\n匹配的文件列表已保存到 {matches_csv_path}")
    
    # 保存对象分布数据
    objects_distribution.to_csv(f'{target_path}/objects_distribution.csv')
    
    # 如果需要，创建tar归档
    if tar:
        print('创建归档文件...')
        archive_directories(target_path, nclients)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-path', type=str, default='datasets/E-O_PCB', help='E-O-PCB数据集源路径')
    parser.add_argument('--target-path', type=str, default='datasets/eo-pcb', help='处理后数据集目标路径')
    parser.add_argument('--data', type=str, default='data/eo-pcb.yaml', help='数据配置文件路径')
    parser.add_argument('--nclients', type=int, default=3, help='联邦学习中的客户端数量')
    parser.add_argument('--val-frac', type=float, default=0.25, help='服务器保留用于验证的数据比例')
    parser.add_argument('--tar', action='store_true', help='是否创建联邦参与者目录的归档')
    args = parser.parse_args()
    
    # 调用主处理函数
    process_eopcb(args.source_path, args.target_path, args.data, args.nclients, args.val_frac, args.tar) 