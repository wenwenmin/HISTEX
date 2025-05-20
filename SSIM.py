from skimage.metrics import structural_similarity as ssim
import numpy as np
import pickle
import os
from utils import read_lines, load_image

def normalize_data(data):

    min_vals = np.min(data, axis=(1, 2), keepdims=True)
    max_vals = np.max(data, axis=(1, 2), keepdims=True)

    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data

predicted_data = []
dir = 'E:\Liver_cancer\\'
folder_path = f"{dir}cnts-super\\"
gene_names = read_lines(f'{dir}gene-names.txt')
for name in gene_names:
    file_path = os.path.join(folder_path, name+'.pickle')
    with open(file_path, 'rb') as f:
        matrix = pickle.load(f)
        predicted_data.append(matrix)
predicted_data = np.array(predicted_data)

with open(f"{dir}genes_3D.pkl", 'rb') as f:
    actual_data = pickle.load(f)
    actual_data[actual_data < 0] = 0
    actual_data = np.transpose(actual_data, (2, 0, 1))
predicted_data = predicted_data[:, :actual_data.shape[1], :actual_data.shape[2]]

mask = load_image(f"{dir}mask.png") > 0
mask = mask[:, :, 0]

predicted_data = normalize_data(predicted_data)
actual_data = normalize_data(actual_data)

predicted_data = np.where(mask, predicted_data, 0)
actual_data = np.where(mask, actual_data, 0)

SSIM = []
for i in range(predicted_data.shape[0]):
    ssim_index = ssim(actual_data[i], predicted_data[i], data_range=1.0)
    SSIM.append(ssim_index)

x = SSIM
print('Mean:', np.nanmean(x))
print('Var:', np.nanvar(x))

folder = f"{dir}SSIM\\"
np.savetxt(f'{folder}ours.txt', x)