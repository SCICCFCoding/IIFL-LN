# FedPylot by Cyprien Quéméneur, GPL-3.0 license

import os
import pandas as pd
import tarfile
import yaml

#该函数用于创建联邦学习所需的目录结构
def create_directories(target_path: str, nclients: int) -> None:
    """Create directory structure for the data."""
    # Create target directory
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    # Create server subdirectory
    for subdict in ['', '/images', '/labels']:
        if not os.path.exists(f'{target_path}/server{subdict}'):
            os.makedirs(f'{target_path}/server{subdict}')
    # Create client subdirectories
    for k in range(1, nclients + 1):
        for subdict in ['', '/images', '/labels']:
            if not os.path.exists(f'{target_path}/client{k}{subdict}'):
                os.makedirs(f'{target_path}/client{k}{subdict}')

#该函数用于将创建的目录结构打包归档，便于在计算机集群中存储和传输，
def archive_directories(target_path: str, nclients: int) -> None:
    """Archive the directories of the federated participants to store them in the computer cluster."""
    # Archive the server directory
    server_path = os.path.join(target_path, 'server')
    tar_file_name = os.path.join(target_path, 'server.tar')
    with tarfile.open(tar_file_name, 'w') as tar_handle:
        tar_handle.add(server_path, arcname='server')
    # Archive the client directories
    for k in range(1, nclients + 1):
        client_path = os.path.join(target_path, f'client{k}')
        tar_file_name = os.path.join(target_path, f'client{k}.tar')
        with tarfile.open(tar_file_name, 'w') as tar_handle:
            tar_handle.add(client_path, arcname=f'client{k}')

#该函数用于创建一个数据分布统计表格（DataFrame）
# 这对于分析联邦学习中数据分布的均衡性非常有用，有助于评估实验结果的可靠性。
def get_distribution_dataframe(data: str, nclients: int) -> pd.DataFrame:
    """Create a dataframe to store the distribution of the annotations between the nodes."""
    columns = ['server']
    with open(data, 'r') as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
    for k in range(1, nclients + 1):
        columns.append(f'client{k}')
    objects_distribution = pd.DataFrame(columns=columns, index=['Samples'] + data_dict['names'])
    objects_distribution.fillna(0, inplace=True)
    return objects_distribution

#这是一个边界框格式转换工具，用于将边界框坐标从原始格式转换为 YOLO 目标检测框架所需的格式
def convert_bbox(bbox_left: float, bbox_top: float, bbox_right: float, bbox_bottom: float, img_width: int,
                 img_height: int) -> tuple[float, float, float, float]:
    """Convert bounding box annotations to YOLO format."""
    x = (bbox_left + bbox_right) / 2 / img_width
    y = (bbox_top + bbox_bottom) / 2 / img_height
    w = (bbox_right - bbox_left) / img_width
    h = (bbox_bottom - bbox_top) / img_height
    return x, y, w, h
