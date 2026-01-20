import torch
import numpy as np
from typing import Dict, List, Tuple

class ImmuneDetector:
    """免疫负向选择检测器 - 增强版（欧氏距离 + 余弦相似度）"""
    
    def __init__(self, threshold: float = 0.5):
        """
        初始化检测器
        
        Args:
            threshold: 欧氏距离异常检测阈值，默认为0.5
            cosine_threshold: 余弦相似度阈值，默认为0.5
        """
        self.threshold = threshold
        
    def calculate_distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        计算两个张量之间的欧氏距离
        """
        a = a.to(torch.float)
        b = b.to(torch.float)
        sq_a = a ** 2
        sum_sq_a = torch.sum(sq_a, dim=1).unsqueeze(1)
        sq_b = b ** 2
        sum_sq_b = torch.sum(sq_b, dim=1).unsqueeze(0)
        bt = b.t()
        return torch.sqrt((sum_sq_a + sum_sq_b - 2 * a.mm(bt)))
    
    def calculate_cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """
        计算两个张量之间的余弦相似度
        
        Args:
            a: 第一个张量
            b: 第二个张量
            
        Returns:
            float: 余弦相似度值 [-1, 1]
        """
        a_flat = a.flatten().to(torch.float)
        b_flat = b.flatten().to(torch.float)
        
        dot_product = torch.dot(a_flat, b_flat)
        norm_a = torch.norm(a_flat)
        norm_b = torch.norm(b_flat)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        cosine_sim = dot_product / (norm_a * norm_b)
        return cosine_sim.item()  
    
    def _get_client_weight(self, client_id: int, default_weight: float = 1.0) -> float:
        """
        获取客户端权重 - 可以基于样本量或其他指标
        这里使用简单实现，你可以根据实际需求扩展
        """
        return default_weight
    
    def detect(self, 
              trusted_clients: List[int],
              client_params: Dict[int, Dict[str, torch.Tensor]],
              all_clients: List[int],
              previous_model: Dict[str, torch.Tensor] = None) -> Tuple[Dict[str, torch.Tensor], List[int]]:
        """
        使用免疫负向选择算法检测异常客户端 - 增强版
        
        Args:
            trusted_clients: 可信客户端ID列表
            client_params: 客户端参数字典
            all_clients: 所有客户端ID列表
            previous_model: 上一轮全局模型参数（用于计算更新方向）
            
        Returns:
            Tuple[更新方向delta_t, 正常客户端列表]
        """
        sum_parameters = None
        total_weight = 0.0
        
        for client_id in trusted_clients:
            if client_id not in client_params:
                continue
                
            weight = self._get_client_weight(client_id)
            if sum_parameters is None:
                sum_parameters = {}
                for key, var in client_params[client_id].items():
                    sum_parameters[key] = var.clone() * weight
            else:
                for key in sum_parameters:
                    sum_parameters[key] = sum_parameters[key] + client_params[client_id][key] * weight
            total_weight += weight
        

        if total_weight > 0 and sum_parameters is not None:
            for key in sum_parameters:
                sum_parameters[key] = sum_parameters[key] / total_weight
        

        detection_ranges = {}
        cosine_thresholds = {}  
        
        if sum_parameters is not None:
            for key in sum_parameters:
                detection_ranges[key] = 0.0  
                min_cosine_sim = 1.0  
                
                for client_id in trusted_clients:
                    if client_id not in client_params:
                        continue
                    temp_sum = torch.reshape(sum_parameters[key], (1, -1))
                    temp_param = torch.reshape(client_params[client_id][key], (1, -1))
                    distance = self.calculate_distance(temp_sum, temp_param)
                    distance_value = distance.item() if distance.numel() == 1 else torch.mean(distance).item()
                    detection_ranges[key] = max(detection_ranges[key], distance_value)
                    cosine_sim = self.calculate_cosine_similarity(sum_parameters[key], client_params[client_id][key])
                    min_cosine_sim = min(min_cosine_sim, cosine_sim)
                
                cosine_thresholds[key] = 0  
                detection_ranges[key] *= 1.7  
        
        normal_clients = trusted_clients.copy()  
        
        for client_id in all_clients:
            if client_id in trusted_clients:  
                continue
                
            if client_id not in client_params:
                continue

            abnormal_count1 = 0
            abnormal_count2 = 0
            total_params = len(sum_parameters) if sum_parameters else 0
            
            if sum_parameters is not None:
                for key in sum_parameters:
                    temp_sum = torch.reshape(sum_parameters[key], (1, -1))
                    temp_param = torch.reshape(client_params[client_id][key], (1, -1))

                    distance = self.calculate_distance(temp_sum, temp_param)

                    distance_value = distance.item() if distance.numel() == 1 else torch.mean(distance).item()
                    distance_abnormal = distance_value > detection_ranges[key]

                    cosine_sim = self.calculate_cosine_similarity(sum_parameters[key], client_params[client_id][key])
                    cosine_abnormal = cosine_sim < cosine_thresholds[key]  #

                    if distance_abnormal:
                        abnormal_count1 += 1
                    if cosine_abnormal:
                        abnormal_count2 += 1

            if abnormal_count1 <= total_params * (1 - self.threshold) and abnormal_count2 <= total_params * 0.5:
                normal_clients.append(client_id)
                print(f"[IMMUNE] 客户端{client_id} 被判定为正常节点 (欧氏距离异常参数数: {abnormal_count1}/{total_params})")
                print(f"[IMMUNE] 客户端{client_id} 被判定为正常节点 (余弦异常参数数: {abnormal_count2}/{total_params})")
                weight = self._get_client_weight(client_id)
                if sum_parameters is not None:
                    for key in sum_parameters:
                        sum_parameters[key] = (sum_parameters[key] * total_weight + 
                                             client_params[client_id][key] * weight) / (total_weight + weight)
                    total_weight += weight
            else:
                print(f"[IMMUNE] 客户端{client_id} 被判定为异常节点 (欧氏距离异常参数数: {abnormal_count1}/{total_params})")
                print(f"[IMMUNE] 客户端{client_id} 被判定为异常节点 (余弦异常参数数: {abnormal_count2}/{total_params})")

        if previous_model is not None and sum_parameters is not None:
            delta_t = {}
            for key in sum_parameters.keys():
                if key in previous_model:
                    delta_t[key] = sum_parameters[key] - previous_model[key]
                else:
                    delta_t[key] = sum_parameters[key]  
            return delta_t, normal_clients
        else:
            return sum_parameters, normal_clients


