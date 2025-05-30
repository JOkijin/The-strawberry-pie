import os 
import json 
import numpy as np 
import torch 
from PIL import Image 
from torch.utils.data  import Dataset, DataLoader 
from torchvision import transforms as T 
from torchvision.transforms  import functional as F 
from utils.metrics  import validate_floorplan  # 论文提到的规范校验模块 
 
class InteriorDesignDataset(Dataset):
    """支持多模态室内设计数据加载的增强数据集"""
    
    def __init__(self, 
                 root_dir: str,
                 phase: str = 'train',
                 img_size: tuple = (512, 512),
                 augment: bool = True):
        """
        Args:
            root_dir (str): 包含floorplans和materials的根目录 
            phase (str): 训练/验证阶段 
            img_size (tuple): 输出图像尺寸 
            augment (bool): 是否启用数据增强 
        """
        self.root  = root_dir 
        self.phase  = phase 
        self.img_size  = img_size 
        self.augment  = augment 
        
        # 加载论文中提到的户型图元数据 
        with open(os.path.join(root_dir,  'metadata.json'))  as f:
            self.metadata  = json.load(f)['samples'] 
            
        # 材质库加载（论文中Substance Source预处理）
        self.materials  = self._load_materials(
            os.path.join(root_dir,  'materials')
        )
        
        # 论文中提到的三种核心增强 
        self.transform  = self._build_transforms()
        
        # 缓存验证器（来自技术架构）
        self.validator  = validate_floorplan  
 
    def _load_materials(self, mat_dir):
        """加载PBR材质贴图（论文3.2节材质合成）"""
        mat_dict = {}
        for mat_name in os.listdir(mat_dir): 
            path = os.path.join(mat_dir,  mat_name)
            mat_dict[mat_name] = {
                'albedo': Image.open(os.path.join(path,  'albedo.png')), 
                'normal': Image.open(os.path.join(path,  'normal.png')), 
                'roughness': Image.open(os.path.join(path,  'roughness.png')) 
            }
        return mat_dict 
 
    def _build_transforms(self):
        """构建符合论文技术路线的增强流水线"""
        base_transforms = [
            T.Resize(self.img_size),   # 保持户型图比例 
            T.Lambda(lambda x: x.convert('RGB')  if x.mode  != 'RGB' else x),
            T.ToTensor()
        ]
        
        if self.augment: 
            # 论文3.1节提到的空间增强 
            spatial_transforms = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.3),
                RandomPerspectiveDistortion(max_scale=0.1),  # 自定义视角变换 
                RandomRoomCut(probability=0.3),  # 论文提到的户型切割增强 
            ])
            
            # 论文3.2节材质增强 
            appearance_transforms = T.Compose([
                PBRMaterialTransform(self.materials),   # 材质替换 
                T.ColorJitter(brightness=0.2, contrast=0.3),
                GaussianNoise(p=0.2)  # 模拟扫描噪声 
            ])
            
            return T.Compose([
                spatial_transforms,
                appearance_transforms,
                *base_transforms 
            ])
        else:
            return T.Compose(base_transforms)
 
    def __len__(self):
        return len(self.metadata) 
 
    def __getitem__(self, idx):
        """返回包含多模态设计数据的字典"""
        sample = self.metadata[idx] 
        
        # 加载户型图（论文基础数据）
        fp_img = Image.open( 
            os.path.join(self.root,  'floorplans', sample['file'])
        ).convert('L')  # 灰度图加载 
        
        # 数据增强（论文第3章）
        fp_tensor = self.transform(fp_img) 
        
        # 加载关联的元数据 
        meta = {
            'area': torch.tensor(sample['area'],  dtype=torch.float), 
            'room_types': torch.tensor(sample['room_types']), 
            'style_label': torch.tensor(sample['style_code']), 
            'is_valid': self.validator(sample)   # 规范校验（论文2.3节）
        }
        
        # 添加材质参数（论文3.2节）
        if 'materials' in sample:
            mat_params = self._get_material_params(sample['materials'])
            meta.update(mat_params) 
            
        return {'image': fp_tensor, 'metadata': meta}
 
    def _get_material_params(self, mat_ids):
        """生成材质特征向量"""
        params = []
        for mat_id in mat_ids:
            mat = self.materials[mat_id] 
            # 将材质图转换为特征向量 
            params.append(torch.stack([ 
                T.functional.to_tensor(mat['albedo']).mean(dim=(1,2)), 
                T.functional.to_tensor(mat['normal']).mean(dim=(1,2)), 
                T.functional.to_tensor(mat['roughness']).mean() 
            ]))
        return {'materials': torch.cat(params)} 
        
 
# 论文3.1节提到的自定义增强类 
class RandomPerspectiveDistortion(T.RandomPerspective):
    """增强版透视变换，支持参数化控制"""
    def __init__(self, max_scale=0.1, distortion_scale=0.5, p=0.5):
        super().__init__(distortion_scale, p)
        self.max_scale  = max_scale 
 
    def forward(self, img):
        if torch.rand(1)  < self.p:
            w, h = img.size  
            startpoints, endpoints = self.get_params(w,  h)
            return F.perspective(img,  startpoints, endpoints)
        return img 
 
    def get_params(self, width, height):
        """生成动态变形参数"""
        # 论文中采用的参数化范围 
        offset = int(width * self.max_scale) 
        return (
            [(0, 0), (width, 0), (width, height), (0, height)],  # 原始点 
            [(random.randint(-offset,  offset), random.randint(-offset,  offset)) for _ in range(4)]  # 变形点 
        )
 
class RandomRoomCut:
    """论文3.1节提到的户型切割增强"""
    def __init__(self, probability=0.3):
        self.probability  = probability 
 
    def __call__(self, img):
        if random.random()  < self.probability: 
            w, h = img.size  
            # 随机选择切割方向 
            if random.choice([True,  False]):
                # 垂直切割 
                cut_pos = random.randint(int(w*0.3),  int(w*0.7))
                return img.crop((0,  0, cut_pos, h))
            else:
                # 水平切割 
                cut_pos = random.randint(int(h*0.3),  int(h*0.7))
                return img.crop((0,  0, w, cut_pos))
        return img 
 
class PBRMaterialTransform:
    """论文3.2节材质替换增强"""
    def __init__(self, material_db):
        self.materials  = material_db 
        
    def __call__(self, img):
        # 随机选择3种材质进行混合 
        selected_mats = random.sample(list(self.materials.keys()),  3)
        layers = []
        for mat_name in selected_mats:
            mat = self.materials[mat_name] 
            # 使用论文中的材质混合公式 
            blend_mode = random.choice(['overlay',  'multiply'])
            layers.append(self._apply_material(img,  mat, blend_mode))
        return Image.blend(layers[0],  layers[1], 0.3)
 
    def _apply_material(self, base_img, mat, mode):
        """应用单个材质混合"""
        # 调整材质图尺寸 
        mat_img = mat['albedo'].resize(base_img.size) 
        if mode == 'overlay':
            return ImageChops.overlay(base_img,  mat_img)
        elif mode == 'multiply':
            return ImageChops.multiply(base_img,  mat_img)
 
class GaussianNoise:
    """模拟论文中提到的扫描噪声"""
    def __init__(self, p=0.2, mean=0., std=0.05):
        self.p = p 
        self.mean  = mean 
        self.std  = std 
 
    def __call__(self, img):
        if random.random()  < self.p:
            np_img = np.array(img) 
            noise = np.random.normal(self.mean,  self.std,  np_img.shape) 
            noisy_img = np.clip(np_img  + noise * 255, 0, 255).astype(np.uint8) 
            return Image.fromarray(noisy_img) 
        return img 
