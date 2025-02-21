#  x包含两部分 组织特征和spot-level基因表达 (batch_size, 49, 1024), (batch_size, 20, 1024) 可以是一个列表/字典
from torch.utils.data import Dataset
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from utils import get_disk_mask

def dimensionality_reduction(arr, target_size):
    x = np.arange(arr.shape[-1])
    f = interp1d(x, arr, axis=-1)
    arr_compressed = f(np.linspace(0, arr.shape[-1] - 1, target_size))
    return arr_compressed

class SpotDataset(Dataset):

    def __init__(self, x_all, y, enhance_y, locs, enhance_locs, radius):
        super().__init__()
        # x_all = x_all[:, :, 0:1000]
        #x_all = dimensionality_reduction(x_all, y.shape[-1])
        mask = get_disk_mask(radius)
        his = get_patches_flat(x_all, locs, mask)
        gene = get_patches_genes(enhance_locs, enhance_y, k=5)
        x = dict(his=his, gene=gene)
        self.x = x
        self.y = y
        self.locs = locs
        self.size = x_all.shape[:2]
        self.radius = radius
        self.mask = mask

    def __len__(self):
        return len(self.x['his'])

    def __getitem__(self, idx):
        x_item = {key: value[idx] for key, value in self.x.items()}
        y_item = self.y[idx]
        return x_item, y_item


def get_patches_flat(img, locs, mask):
    shape = np.array(mask.shape)
    mask = np.ones_like(mask, dtype=bool) #这个是按方块切 不是圆
    center = shape // 2
    r = np.stack([-center, shape-center], -1)  # offset
    x_list = []
    for s in locs:
        patch = img[
                s[0]+r[0][0]:s[0]+r[0][1],
                s[1]+r[1][0]:s[1]+r[1][1]]

        x = patch[mask]
        x_list.append(x)
    x_list = np.stack(x_list)
    return x_list

def get_patches_genes(locs, y, k=20):
    """
    根据坐标选取每个细胞的 k 个最近邻细胞（包括自身），并提取基因表达数据。

    参数:
    - locs: np.ndarray, 形状为 (n, 2)，细胞的坐标。
    - y: np.ndarray, 形状为 (n, 1000)，细胞的基因表达数据。
    - k: int, 每个细胞选取的最近邻数量，默认值为 20。

    返回:
    - patches: list, 每个元素是形状为 (k, 1000) 的 numpy 数组。
    """
    # 创建KD树用于快速寻找最近邻
    tree = cKDTree(locs)

    # 查询每个点的 k 个最近邻
    _, indices = tree.query(locs, k=k)

    # 收集每个细胞的最近邻基因表达数据
    genes_list = [y[idx] for idx in indices]
    genes_list = np.stack(genes_list)
    return genes_list

def get_patches_genes_test(locs1, locs2, y, k=20):
    """
    从 locs1 的每个坐标在 locs2 中寻找 k 个最近邻点，并提取基因表达数据。

    参数:
    - locs1: np.ndarray, 形状为 (n1, 2)，源坐标。
    - locs2: np.ndarray, 形状为 (n2, 2)，目标坐标。
    - y: np.ndarray, 形状为 (n2, 1000)，目标坐标对应的基因表达数据。
    - k: int, 每个点选取的最近邻数量，默认值为 20。

    返回:
    - patches: list, 每个元素是形状为 (k, 1000) 的 numpy 数组。
    """
    # 在 locs2 中构建 KD 树
    tree = cKDTree(locs2)

    # 查询 locs1 中的每个点在 locs2 中的 k 个最近邻
    _, indices = tree.query(locs1, k=k)

    # 收集每个点在 locs2 中的最近邻基因表达数据
    patches = [y[idx] for idx in indices]

    return patches

def get_center_coordinates_rounded(h, w, block_size):
    """
    获取每个子块的中心坐标，并通过四舍五入获得整数值。

    参数:
    - h: int, 张量在高度方向的维度大小。
    - w: int, 张量在宽度方向的维度大小。
    - block_size: int, 子块的边长（假设子块是正方形）。

    返回:
    - locs1: np.ndarray, 每个子块中心坐标组成的矩阵，形状为 (num_blocks, 2)。
    """
    # 计算子块的步长
    step = block_size

    # 确定所有子块中心的 x 和 y 坐标（四舍五入）
    center_y = np.arange(step / 2, h, step)
    center_x = np.arange(step / 2, w, step)

    # 生成网格坐标
    grid_x, grid_y = np.meshgrid(center_x, center_y)

    # 四舍五入并转换为整数
    grid_x = np.round(grid_x).astype(int)
    grid_y = np.round(grid_y).astype(int)

    # 将网格展平并组合为 (num_blocks, 2) 的矩阵
    locs1 = np.stack([grid_y.ravel(), grid_x.ravel()], axis=-1)

    return locs1