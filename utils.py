import itertools
import os
import random
import rasterio
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import io
from sklearn.metrics import confusion_matrix
from torchvision.utils import make_grid
from tqdm.auto import tqdm
from typing import Optional, Union


def fix_random_seed(seed):
    """
    固定随机种子
    :param seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # 禁用cuDNN基准测试
    torch.backends.cudnn.deterministic = True  # 强制使用确定性卷积算法
    torch.use_deterministic_algorithms(True)


def convert_to_color(arr_2d, palette):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d


def convert_from_color(arr_3d, palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d


def rgb_to_class_indices(rgb, color_map):
    """
    将RGB图像转换为类别索引。

    Args:
        rgb (torch.Tensor): RGB图像，形状为 [B, 3, H, W]，范围为 [0, 1]。
        color_map (dict): 类别到RGB颜色的映射字典。

    Returns:
        torch.Tensor: 类别索引图像，形状为 [B, H, W]。
    """
    # 将 color_map 转换为一个 [num_classes, 3] 的张量
    color_map_tensor = torch.tensor(
        list(color_map.values()), dtype=torch.float32, device=rgb.device
    ) / 255.0  # 归一化到 [0, 1] 范围内

    # 重塑 rgb 以便于与 color_map 进行比较
    # rgb: [B, 3, H, W] -> [B, H, W, 3]
    rgb = rgb.permute(0, 2, 3, 1)

    # 扩展 color_map_tensor 以匹配 rgb 的维度
    # color_map_tensor: [num_classes, 3] -> [1, num_classes, 1, 1, 3]
    color_map_tensor = color_map_tensor.view(1, -1, 1, 1, 3)

    # 扩展 rgb 以进行广播比较
    # rgb: [B, H, W, 3] -> [B, 1, H, W, 3]
    rgb_expanded = rgb.unsqueeze(1)

    # 比较每个像素与每个类别的颜色是否匹配
    matches = (rgb_expanded == color_map_tensor).all(dim=-1)  # [B, num_classes, H, W]

    # 找到每个像素的类别索引
    class_indices = torch.argmax(matches.float(), dim=1)  # [B, H, W]

    # print(class_indices.shape)

    return class_indices


def save_img(tensor, name):
    tensor = tensor.cpu().permute((1, 0, 2, 3))
    im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=2).permute((1, 2, 0))
    im = (im.data.numpy() * 255.).astype(np.uint8)
    Image.fromarray(im).save(name + '.jpg')


def format_string(input_str):
    parts = input_str.split('_')
    formatted_parts = [part.zfill(2) for part in parts]
    return '_'.join(formatted_parts)


class ISPRS_dataset(torch.utils.data.Dataset):
    def __init__(self, ids, dataset_cfg, window_size, cache=False, augmentation=True):
        super(ISPRS_dataset, self).__init__()

        self.augmentation = augmentation
        self.cache = cache
        self.window_size = tuple(window_size)
        self.dataset_cfg = dataset_cfg
        self.invert_palette = {tuple(v): k for k, v in dataset_cfg.palette.items()}

        # List of files
        self.data_files = [dataset_cfg.data_folder.format(id) for id in ids]
        if dataset_cfg.name == 'Potsdam':
            dif_ids = [format_string(id) for id in ids]
            self.dsm_files = [dataset_cfg.dsm_folder.format(id) for id in dif_ids]
        else:
            self.dsm_files = [dataset_cfg.dsm_folder.format(id) for id in ids]
        self.label_files = [dataset_cfg.label_folder.format(id) for id in ids]

        # Sanity check : raise an error if some files do not exist
        for f in self.data_files + self.dsm_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))

        # Initialize cache dicts
        self.data_cache_ = {}
        self.dsm_cache_ = {}
        self.label_cache_ = {}

    def __len__(self):
        return 10 * 1000

    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True

        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))

        return tuple(results)

    def __getitem__(self, i):
        # Pick a random image
        random_idx = random.randint(0, len(self.data_files) - 1)

        # If the tile hasn't been loaded yet, put in cache
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            # Data is normalized in [0, 1]
            if self.dataset_cfg.name == 'Potsdam':
                data = io.imread(self.data_files[random_idx])  # RGBNIR
                data = 1 / 255 * np.asarray(data.transpose((2, 0, 1)), dtype='float32')
            elif self.dataset_cfg.name == 'Vaihingen':
                data = io.imread(self.data_files[random_idx])
                data = 1 / 255 * np.asarray(data.transpose((2, 0, 1)), dtype='float32')
            if self.cache:
                self.data_cache_[random_idx] = data

        if random_idx in self.dsm_cache_.keys():
            dsm = self.dsm_cache_[random_idx]
        else:
            if self.dataset_cfg.name == 'Potsdam' or self.dataset_cfg.name == 'Vaihingen':
                # DSM is normalized in [0, 1]
                dsm = np.asarray(io.imread(self.dsm_files[random_idx]), dtype='float32')
                min = np.min(dsm)
                max = np.max(dsm)
                dsm = (dsm - min) / (max - min)
                if self.cache:
                    self.dsm_cache_[random_idx] = dsm

        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else:
            # Labels are converted from RGB to their numeric values
            label = np.asarray(convert_from_color(io.imread(self.label_files[random_idx]), self.invert_palette),
                               dtype='int64')
            if self.cache:
                self.label_cache_[random_idx] = label

        # Get a random patch
        x1, x2, y1, y2 = get_random_pos(data, self.window_size)
        data_p = data[:, x1:x2, y1:y2]
        dsm_p = dsm[x1:x2, y1:y2]
        label_p = label[x1:x2, y1:y2]

        # Data augmentation
        data_p, dsm_p, label_p = self.data_augmentation(data_p, dsm_p, label_p)

        # Return the torch.Tensor values
        return (torch.from_numpy(data_p),
                torch.from_numpy(dsm_p),
                torch.from_numpy(label_p))


# class WHUDataset(torch.utils.data.Dataset):
#     def __init__(self, data_dir: str, split: str, color_map: dict):
#         """
#         Args:
#             data_dir (str): Path to the dataset directory.
#             split (str): One of 'train', 'val', 'test'.
#             color_map (dict): Mapping from class indices to RGB colors.
#         """
#         self.data_dir = Path(data_dir) / split
#         self.color_map = color_map
#         self.transform = None
#
#         # Assuming each modality has its own subdirectory: GT, OPT, SAR
#         self.modalities = ['GT', 'OPT', 'SAR']
#
#         # List all GT files
#         gt_dir = self.data_dir / 'GT'
#         self.filenames = sorted([f.name for f in gt_dir.glob('*.tif')])
#
#     def __len__(self):
#         return len(self.filenames)
#
#     def __getitem__(self, idx):
#         filename = self.filenames[idx]
#
#         # Load modalities
#         modality_tensors = []
#         for modality in ['OPT', 'SAR']:
#             modality_path = self.data_dir / modality / filename
#             with rasterio.open(modality_path) as src:
#                 image = src.read()  # [C, H, W]
#                 image = image.astype(np.float32)
#                 image = image / 255.0
#                 image = torch.from_numpy(image)  # [C, H, W]
#                 modality_tensors.append(image)
#
#         opt_tensor = modality_tensors[0]
#         sar_tensor = modality_tensors[1].squeeze(0)
#
#         # Load Ground Truth
#         gt_path = self.data_dir / 'GT' / filename
#         with rasterio.open(gt_path) as src:
#             gt = src.read()  # [C, H, W]
#             gt = gt.astype(np.float32)
#             gt = gt / 255.0
#             gt = torch.from_numpy(gt)  # [3, H, W]
#
#         # Convert GT to class indices
#         gt_class = rgb_to_class_indices(gt.unsqueeze(0), self.color_map).squeeze(0)  # [H, W]
#
#         return opt_tensor, sar_tensor, gt_class


class WHUDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, split: str, color_map: dict, transform=None):
        """
        Loads all samples into memory on initialization.

        Args:
            data_dir (str): Path to the dataset directory.
            split (str): One of 'train', 'val', 'test'.
            color_map (dict): Mapping from class indices to RGB colors.
            transform (callable, optional): A function/transform to apply to each sample.
        """
        super().__init__()
        self.color_map = color_map
        self.transform = transform

        base_dir = Path(data_dir) / split
        gt_dir = base_dir / 'GT'
        filenames = sorted([f.name for f in gt_dir.glob('*.tif')])

        self.samples = []  # list of tuples: (opt_tensor, sar_tensor, gt_class)
        for filename in tqdm(filenames, desc=f"Loading {split} set", unit="file"):
            # Load OPT
            opt_path = base_dir / 'OPT' / filename
            opt = io.imread(opt_path).astype(np.float32) / 255.0
            if opt.ndim == 2:
                opt = np.expand_dims(opt, axis=0)  # ensure [C, H, W]
            elif opt.ndim == 3:
                opt = np.transpose(opt, (2, 0, 1))
            opt_tensor = torch.from_numpy(opt)

            # Load SAR
            sar_path = base_dir / 'SAR' / filename
            sar = io.imread(sar_path).astype(np.float32) / 255.0
            if sar.ndim == 2:
                sar_tensor = torch.from_numpy(sar)
            else:
                sar = np.transpose(sar, (2, 0, 1))
                sar_tensor = torch.from_numpy(sar.squeeze(0))

            # Load GT
            gt_path = base_dir / 'GT' / filename
            gt = io.imread(gt_path).astype(np.float32) / 255.0
            # if gt.ndim == 2:
            #     gt = np.stack([gt] * 3, axis=-1)
            gt = np.transpose(gt, (2, 0, 1))
            gt_tensor = torch.from_numpy(gt)

            # Convert GT to class indices
            gt_class = rgb_to_class_indices(gt_tensor.unsqueeze(0), self.color_map).squeeze(0)  # [H, W]

            self.samples.append((opt_tensor, sar_tensor, gt_class))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        opt, sar, gt = self.samples[idx]
        if self.transform:
            opt, sar, gt = self.transform(opt, sar, gt)
        return opt, sar, gt


class YESegDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, split: str, color_map: dict, transform=None):
        """
        Loads all samples into memory on initialization.

        Args:
            data_dir (str): Path to the dataset directory.
            split (str): One of 'train', 'val', 'test'.
            color_map (dict): Mapping from class indices to RGB colors.
            transform (callable, optional): A function/transform to apply to each sample.
        """
        super().__init__()
        self.color_map = color_map
        self.transform = transform

        base_dir = Path(data_dir) / split
        gt_dir = base_dir / 'GT'
        filenames = sorted([f.name for f in gt_dir.glob('*.png')])

        if split == 'test':
            print(filenames)

        self.samples = []  # list of tuples: (opt_tensor, sar_tensor, gt_class)
        for filename in tqdm(filenames, desc=f"Loading {split} set", unit="file"):
            # Load OPT
            opt_path = base_dir / 'OPT' / filename
            opt = io.imread(opt_path).astype(np.float32) / 255.0
            if opt.ndim == 2:
                opt = np.expand_dims(opt, axis=0)  # ensure [C, H, W]
            elif opt.ndim == 3:
                opt = np.transpose(opt, (2, 0, 1))
            opt_tensor = torch.from_numpy(opt)

            # Load SAR
            sar_path = base_dir / 'SAR' / filename
            sar = io.imread(sar_path).astype(np.float32) / 255.0
            if sar.ndim == 2:
                sar_tensor = torch.from_numpy(sar)
            else:
                sar = np.transpose(sar, (2, 0, 1))
                sar_tensor = torch.from_numpy(sar)

            # Load GT
            gt_path = base_dir / 'GT' / filename
            gt = io.imread(gt_path).astype(np.float32) / 255.0
            # if gt.ndim == 2:
            #     gt = np.stack([gt] * 3, axis=-1)
            gt = np.transpose(gt, (2, 0, 1))
            gt_tensor = torch.from_numpy(gt)

            # Convert GT to class indices
            gt_class = rgb_to_class_indices(gt_tensor.unsqueeze(0), self.color_map).squeeze(0)  # [H, W]

            self.samples.append((opt_tensor, sar_tensor, gt_class))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        opt, sar, gt = self.samples[idx]
        if self.transform:
            opt, sar, gt = self.transform(opt, sar, gt)
        return opt, sar, gt


# Utils
def get_random_pos(img, window_shape):
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2


def CrossEntropy2d(input, target, weight=None, size_average=True):
    """ 2D version of the cross entropy loss """
    dim = input.dim()
    if dim == 2:
        # return F.cross_entropy(input, target, weight, size_average)
        return F.cross_entropy(input, target, weight, reduction='mean')
    elif dim == 4:
        output = input.view(input.size(0), input.size(1), -1)
        output = torch.transpose(output, 1, 2).contiguous()
        output = output.view(-1, output.size(2))
        target = target.view(-1)
        # return F.cross_entropy(output, target, weight, size_average)
        return F.cross_entropy(output, target, weight, reduction='mean')
    else:
        raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))


def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6,
    ignore_index: Optional[int] = None,
    class_weights: Optional[torch.Tensor] = None,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    通用 Dice Loss（1 - Dice Coefficient），支持二分类和多分类。

    Args:
        logits:       网络输出原始分数，shape = (B, C, H, W) 或 (B, 1, H, W) 或 (B, H, W)
        targets:      真值标签，多分类为 (B, H, W) int；二分类可 (B, H, W) 或 (B, 1, H, W)
        smooth:       防止除 0 的平滑常数
        ignore_index: 要忽略的类别索引（仅多分类有效）
        class_weights:各类权重，shape = (C,)；用于类别不平衡
        reduction:    聚合方式，'mean' | 'sum' | 'none'
    Returns:
        torch.Tensor: 如果 reduction!='none' 则返回标量，否则返回各类/各 batch 的 Dice Loss
    """
    # 统一 shape
    B = logits.size(0)
    if logits.dim() == 3:  # (B, H, W) -> 二分类
        probs = torch.sigmoid(logits.unsqueeze(1))  # (B, 1, H, W)
    else:
        C = logits.size(1)
        if C == 1:  # 二分类 logits shape = (B,1,H,W)
            probs = torch.sigmoid(logits)
        else:      # 多分类
            probs = F.softmax(logits, dim=1)

    # 构造 one-hot
    if probs.size(1) > 1:
        C = probs.size(1)
        # targets: (B,H,W) 或 (B,1,H,W) -> (B,H,W)
        tgt = targets.squeeze(1) if targets.dim()==4 else targets
        t_onehot = F.one_hot(tgt.clamp(0, C-1), num_classes=C)  # (B,H,W,C)
        t_onehot = t_onehot.permute(0,3,1,2).float()           # (B,C,H,W)
    else:
        # 二分类: 把 targets 转成 float 0/1
        t_onehot = targets.float().unsqueeze(1)

    # 如果指定 ignore_index，则将该通道和对应 one-hot 清 0
    if ignore_index is not None and probs.size(1)>1:
        mask = (targets == ignore_index)  # (B,H,W)
        # 在每通道上把被忽略的像素置零
        t_onehot[:, ignore_index, ...][mask] = 0
        probs[:, ignore_index, ...][mask] = 0

    # 计算交并
    # flatten 到 (C, B*H*W)
    p_flat = probs.reshape(probs.size(1), -1)
    t_flat = t_onehot.reshape(t_onehot.size(1), -1)
    # per-class intersection & union
    inter = (p_flat * t_flat).sum(dim=1)
    union = p_flat.sum(dim=1) + t_flat.sum(dim=1)

    # 对“完全空”的通道特殊处理：若 union - smooth == 0，令 dice=1
    dice_score = (2*inter + smooth) / (union + smooth)
    empty_mask = (union < smooth * 0.5)  # 真值+预测都近 0
    dice_score = torch.where(empty_mask, torch.ones_like(dice_score), dice_score)

    # Dice Loss
    loss_per_class = 1.0 - dice_score  # (C,)

    # 应用 class_weights
    if class_weights is not None:
        loss_per_class = loss_per_class * class_weights.view(-1)

    # 聚合
    if reduction == 'none':
        return loss_per_class
    elif reduction == 'sum':
        return loss_per_class.sum()
    else:  # 'mean'
        # 多分类时平均各类；二分类时 C=1，mean 就是返回标量
        return loss_per_class.mean()


def accuracy(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size


def sliding_window(top, step=10, window_size=(20, 20)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image """
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c


def grouper(n, iterable):
    """ Browse an iterator by chunk of n elements """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def metrics(predictions, gts, label_values, num_classes):
    cm = confusion_matrix(
        gts,
        predictions,
        labels=range(len(label_values)))

    OA = {}
    F1 = {}
    MIoU = {}

    # print("Confusion matrix :")
    # print(cm)
    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    # print("%d pixels processed" % (total))
    # print("Total accuracy : %.2f" % (accuracy))
    OA['total'] = accuracy

    Acc = np.diag(cm) / cm.sum(axis=1)
    for l_id, score in enumerate(Acc):
        OA[label_values[l_id]] = score
    # for l_id, score in enumerate(Acc):
    #     print("%s: %.4f" % (label_values[l_id], score))
    # print("---")

    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    F1['mean'] = np.nanmean(F1Score[:num_classes - 1])
    for l_id, score in enumerate(F1Score):
        F1[label_values[l_id]] = score
    # print("F1Score :")
    # for l_id, score in enumerate(F1Score):
    #     print("%s: %.4f" % (label_values[l_id], score))
    # print('mean F1Score: %.4f' % (np.nanmean(F1Score[:N_CLASSES - 1])))
    # print("---")

    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)
    # print("Kappa: %.4f" % (kappa))
    # print("---")

    # Compute MIoU coefficient
    MIoU_ = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))
    MIoU['mean'] = np.nanmean(MIoU_[:num_classes - 1])
    for l_id, score in enumerate(MIoU_):
        MIoU[label_values[l_id]] = score

    # # Compute MIoU coefficient
    # MIoU = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))
    # print(MIoU)
    # MIoU = np.nanmean(MIoU[:N_CLASSES - 1])
    # print('mean MIoU: %.4f' % (MIoU))
    # print("---")

    results = {'OA': OA, 'F1': F1, 'Kappa': kappa, 'MIoU': MIoU}
    # print(results)

    return results
