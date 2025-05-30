"""
室内设计数据集预处理模块 
实现多模态数据增强与规范化处理
"""
 
import os
import cv2
import json 
import numpy as np
from pathlib import Path
from multiprocessing import Pool 
from typing import Tuple, Dict 
from dataclasses import dataclass 
from sklearn.model_selection  import train_test_split
 
@dataclass
class PreprocessConfig:
    """预处理配置参数"""
    input_dir: str = "./raw_data"                  # 原始数据路径 
    output_dir: str = "./processed"                # 输出路径
    target_size: Tuple[int, int] = (1024, 1024)    # 目标尺寸
    augment_ratio: float = 0.7                     # 增强概率 
    material_lib: str = "./materials"              # PBR材质库路径
    batch_size: int = 32                           # 并行处理批量 
    norm_mean: Tuple[float] = (0.485, 0.456, 0.406)# 归一化均值 
    norm_std: Tuple[float] = (0.229, 0.224, 0.225) # 归一化方差 
    test_ratio: float = 0.2                        # 测试集比例 
 
class PlanAugmentor:
    """户型图增强处理器"""
    def __init__(self, config: PreprocessConfig):
        self.cfg  = config
        self._init_material_lib()
    
    def _init_material_lib(self):
        """加载PBR材质库"""
        self.materials  = {}
        for mat_dir in Path(self.cfg.material_lib).glob("*"): 
            self.materials[mat_dir.name]  = {
                "albedo": cv2.imread(str(mat_dir/"albedo.jpg")), 
                "normal": cv2.imread(str(mat_dir/"normal.jpg")), 
                "roughness": cv2.imread(str(mat_dir/"roughness.jpg")) 
            }
    
    def _random_perspective(self, img: np.ndarray)  -> np.ndarray: 
        """随机透视变换（模拟量房视角）"""
        h, w = img.shape[:2] 
        pts1 = np.float32([[0,0],  [w,0], [0,h], [w,h]])
        pts2 = pts1 + np.random.uniform(-0.1,  0.1, pts1.shape)  * [w, h]
        M = cv2.getPerspectiveTransform(pts1,  pts2)
        return cv2.warpPerspective(img,  M, (w,h))
    
    def _material_transfer(self, img: np.ndarray)  -> np.ndarray: 
        """材质迁移增强"""
        mat = np.random.choice(list(self.materials.values())) 
        albedo = cv2.resize(mat["albedo"],  (img.shape[1],  img.shape[0])) 
        normal = cv2.resize(mat["normal"],  (img.shape[1],  img.shape[0])) 
        return cv2.addWeighted(img,  0.7, albedo, 0.3, 0) * (normal/255.0)
 
    def __call__(self, img_path: str) -> Dict[str, np.ndarray]: 
        """处理单个户型图样本"""
        # 加载原始数据 
        img = cv2.imread(img_path) 
        mask = self._extract_layout_mask(img)
        
        # 空间变换增强 
        if np.random.rand()  < self.cfg.augment_ratio: 
            img = self._random_perspective(img)
            mask = self._random_perspective(mask)
        
        # 材质增强 
        if np.random.rand()  < self.cfg.augment_ratio:  
            img = self._material_transfer(img)
        
        # 尺寸标准化
        img = cv2.resize(img,  self.cfg.target_size) 
        mask = cv2.resize(mask,  self.cfg.target_size) 
        
        return {
            "image": img,
            "mask": mask,
            "meta": self._parse_metadata(img_path)
        }
 
    def _extract_layout_mask(self, img: np.ndarray)  -> np.ndarray: 
        """提取户型结构掩模（论文第2章技术）"""
        gray = cv2.cvtColor(img,  cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray,  200, 255, cv2.THRESH_BINARY_INV)
        return cv2.morphologyEx(mask,  cv2.MORPH_CLOSE, np.ones((5,5),  np.uint8)) 
 
    def _parse_metadata(self, path: str) -> Dict:
        """解析户型元数据"""
        meta_path = Path(path).with_suffix('.json')
        with open(meta_path) as f:
            return json.load(f) 
 
class DatasetSplitter:
    """数据集分割管理器"""
    @staticmethod 
    def split(data: list, test_ratio: float = 0.2) -> Dict[str, list]:
        """按比例分割数据集"""
        train, test = train_test_split(data, test_size=test_ratio)
        return {"train": train, "test": test}
 
class Preprocessor:
    """预处理主控制器"""
    def __init__(self, config: PreprocessConfig):
        self.cfg  = config 
        self.augmentor  = PlanAugmentor(config)
        self.splitter  = DatasetSplitter()
        
    def process_dataset(self):
        """全流程处理"""
        samples = self._collect_samples()
        processed = self._batch_process(samples)
        split_data = self.splitter.split(processed,  self.cfg.test_ratio) 
        self._save_dataset(split_data)
    
    def _collect_samples(self) -> list:
        """收集原始数据样本"""
        return list(Path(self.cfg.input_dir).glob("**/*.png")) 
    
    def _batch_process(self, samples: list) -> list:
        """多进程批量处理"""
        with Pool(os.cpu_count())  as pool:
            results = pool.map(self.augmentor,  samples)
        return [r for r in results if self._validate_sample(r)]
    
    def _validate_sample(self, sample: Dict) -> bool:
        """合规性验证（论文第4章规范）"""
        meta = sample["meta"]
        # 门宽检查
        door_ok = all(w >= 900 for w in meta["door_widths"])
        # 窗地比检查
        window_ratio = meta["window_area"] / meta["total_area"]
        return door_ok and (window_ratio >= 0.2)
    
    def _save_dataset(self, data: Dict[str, list]):
        """保存处理后的数据集"""
        for split_name, samples in data.items(): 
            split_dir = Path(self.cfg.output_dir)  / split_name
            split_dir.mkdir(parents=True,  exist_ok=True)
            
            for idx, sample in enumerate(samples):
                base_path = split_dir / f"sample_{idx:04d}"
                cv2.imwrite(str(base_path.with_suffix(".jpg")),  sample["image"])
                cv2.imwrite(str(base_path.with_suffix("_mask.png")),  sample["mask"])
                with open(base_path.with_suffix(".json"),  "w") as f:
                    json.dump(sample["meta"],  f)
 
if __name__ == "__main__":
    config = PreprocessConfig(
        input_dir="./raw_plans",
        output_dir="./dataset",
        material_lib="./pbr_materials"
    )
    processor = Preprocessor(config)
    processor.process_dataset() 
