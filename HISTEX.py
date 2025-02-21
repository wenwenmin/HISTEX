import argparse
import multiprocessing
import torch
import numpy as np
from utils import read_lines, read_string, save_pickle, load_pickle, load_tsv, load_image
from train import get_model as train_load_model
from model import SpaPSC
from Datasets import SpotDataset, get_center_coordinates_rounded, get_patches_genes_test
from scipy.interpolate import interp1d

def dimensionality_reduction(arr, target_size):
    x = np.arange(arr.shape[-1])
    f = interp1d(x, arr, axis=-1)
    arr_compressed = f(np.linspace(0, arr.shape[-1] - 1, target_size))
    return arr_compressed

def pad(emd, num):
    h, w = emd.shape[0], emd.shape[1]

    pad_h = (num - h % num) % num
    pad_w = (num - w % num) % num

    padded_matrix = np.pad(emd,
                           ((0, pad_h), (0, pad_w), (0, 0)),
                           'constant', constant_values=0)

    new_h, new_w = padded_matrix.shape[:2]
    assert new_h % num == 0 and new_w % num == 0
    return padded_matrix

def get_locs(prefix, target_shape=None, enhance=False):
    if enhance:
        enhance_locs = load_tsv(f'{prefix}HRlocs.csv')
        locs = load_tsv(f'{prefix}locs.csv')
    else:
        enhance_locs = None
        locs = load_tsv(f'{prefix}locs.csv')

    enhance_locs = np.stack([enhance_locs['y'], enhance_locs['x']], -1)
    locs = np.stack([locs['y'], locs['x']], -1)

    if target_shape is not None:
        wsi = load_image(f'{prefix}he.jpg')
        current_shape = np.array(wsi.shape[:2])
        rescale_factor = current_shape // target_shape
        enhance_locs = enhance_locs.astype(float)
        enhance_locs /= rescale_factor
        locs = locs.astype(float)
        locs /= rescale_factor

    enhance_locs = enhance_locs.round().astype(int)
    locs = locs.round().astype(int)

    return locs, enhance_locs

def get_data(prefix, enhance=False):
    gene_names = read_lines(f'{prefix}gene-names.txt')
    if enhance:
        enhance_cnts = load_tsv(f'{prefix}HRcnts.csv')
        cnts = load_tsv(f'{prefix}cnts.csv')
    else:
        enhance_cnts = None
        cnts = load_tsv(f'{prefix}cnts.csv')
    enhance_cnts = enhance_cnts.iloc[:, enhance_cnts.var().to_numpy().argsort()[::-1]]
    enhance_cnts = enhance_cnts[gene_names]
    cnts = cnts.iloc[:, cnts.var().to_numpy().argsort()[::-1]]
    cnts = cnts[gene_names]

    embs = load_pickle(f'{prefix}embeddings-hist.pickle')
    embs = np.concatenate([embs['his'], embs['rgb'], embs['pos']]).transpose(1, 2, 0)
    #embs = np.concatenate([embs['sub'], embs['rgb'], embs['loc']]).transpose(1, 2, 0)
    #embs = np.concatenate([embs['his']]).transpose(1, 2, 0)

    locs, enhance_locs = get_locs(prefix, target_shape=embs.shape[:2], enhance=enhance)
    return embs, cnts, enhance_cnts, locs, enhance_locs

def get_model_kwargs(kwargs):
    return get_model(**kwargs)

def get_model(
        x, y, en_y, locs, en_locs, radius, prefix, batch_size, epochs, lr, num_embeddings,
        load_saved=False, device='cuda'):

    print('x:', x.shape, ', y:', y.shape)

    x = x.copy()

    dataset = SpotDataset(x, y, en_y, locs, en_locs, radius)
    model = train_load_model(
            model_class=SpaPSC,
            model_kwargs=dict(
                num_features=x.shape[-1],
                num_genes=y.shape[-1],
                num_embeddings=num_embeddings,
                radius=radius,
                lr=lr),
            dataset=dataset, prefix=prefix,
            batch_size=batch_size, epochs=epochs,
            load_saved=load_saved, device=device)
    model.eval()
    if device == 'cuda':
        torch.cuda.empty_cache()
    return model, dataset

def normalize(embs, cnts):

    embs = embs.copy()
    cnts = cnts.copy()

    # TODO: check if adjsut_weights in extract_features can be skipped
    embs_mean = np.nanmean(embs, (0, 1))
    embs_std = np.nanstd(embs, (0, 1))
    embs -= embs_mean
    embs /= embs_std + 1e-12

    cnts_min = cnts.min(0)
    cnts_max = cnts.max(0)
    cnts -= cnts_min
    cnts /= (cnts_max - cnts_min) + 1e-12

    return embs, cnts, (embs_mean, embs_std), (cnts_min, cnts_max)

def predict_single_out(model, z, x, indices, names, y_range):
    z = torch.tensor(z, device=model.device).squeeze(0).to(torch.float32)  # 32 49 1000
    x = torch.tensor(x, device=model.device)
    x = x.reshape(x.shape[0], -1, x.shape[-1]).to(torch.float32)
    #z = torch.cat((z, x), dim=2)
    y = model.get_gene(z, indices=indices)
    y = y.cpu().detach().numpy()

    y *= y_range[:, 1] - y_range[:, 0]
    y += y_range[:, 0]
    return y


def predict_single_lat(model, x, genes):
    x = torch.tensor(x, device=model.device)
    x = x.reshape(x.shape[0], -1, x.shape[-1]).to(torch.float32)
    genes = torch.tensor(genes, device=model.device)
    genes = genes.reshape(genes.shape[0], -1, genes.shape[-1]).to(torch.float32)
    z = model.get_Multi_Fea(x, genes)

    z = z.cpu().detach().numpy()
    return z

def predict(h, w,
        model_states, x_batches, genes, name_list, y_range, prefix,
        device='cuda', gra_size=7):

    batch_size_outcome = 100

    model_states = [mod.to(device) for mod in model_states]

    z_states_batches_1 = [
            [predict_single_lat(mod, x_bat, gene) for mod in model_states]
            for x_bat, gene in zip(x_batches, genes)]
    z_point = np.concatenate([
        np.median(z_states, 0)
        for z_states in z_states_batches_1])

    z_1 = np.zeros((h + (gra_size - h % gra_size), w + (gra_size - w % gra_size), z_point.shape[-1]))
    k = 0
    for i in range(0, h, gra_size):
        for j in range(0, w, gra_size):
            z_1[i:i + gra_size, j:j + gra_size, :] = z_point[k].reshape(gra_size, gra_size, z_point.shape[-1])
            k = k + 1
    z_1 = z_1[0:h, 0:w, :]

    z_dict = dict(cls=z_1.transpose(2, 0, 1))
    save_pickle(
            z_dict,
            prefix+'embeddings-gene.pickle')
    del z_point

    idx_list = np.arange(len(name_list))
    n_groups_outcome = len(idx_list) // batch_size_outcome + 1
    idx_groups = np.array_split(idx_list, n_groups_outcome)
    for idx_grp in idx_groups:
        name_grp = name_list[idx_grp]
        y_ran = y_range[idx_grp]
        y_grp = np.concatenate([
            np.median([
                predict_single_out(mod, z, x_batch, idx_grp, name_grp, y_ran)
                for mod, z in zip(model_states, z_states)], 0)
            for z_states, x_batch in zip(z_states_batches_1, x_batches)])

        z_1 = np.zeros((h + (gra_size - h % gra_size), w + (gra_size - w % gra_size), y_grp.shape[-1]))
        k = 0
        for i in range(0, h, gra_size):
            for j in range(0, w, gra_size):
                z_1[i:i + gra_size, j:j + gra_size, :] = y_grp[k].reshape(gra_size, gra_size, y_grp.shape[-1])
                k = k + 1
        z_1 = z_1[0:h, 0:w, :]
        for i, name in enumerate(name_grp):
            save_pickle(z_1[..., i], f'{prefix}cnts-super/{name}.pickle')
    print(f'All genes have been saved in {prefix}cnts-super/..')

def impute(
        embs, cnts, locs, en_cnts, en_locs, radius, epochs, batch_size, prefix, MCA_embeddings,
        n_states=1, load_saved=False, device='cuda', n_jobs=1):

    names = cnts.columns

    cnts = cnts.to_numpy()
    cnts = cnts.astype(np.float32)
    __, cnts, __, (cnts_min, cnts_max) = normalize(embs, cnts)

    en_cnts = en_cnts.to_numpy()
    en_cnts = en_cnts.astype(np.float32)
    __, en_cnts, __, (en_cnts_min, en_cnts_max) = normalize(embs, en_cnts)


    kwargs_list = [
            dict(
                x=embs, y=cnts, en_y=en_cnts, locs=locs, en_locs=en_locs, radius=radius,
                batch_size=batch_size, epochs=epochs, lr=1e-4, num_embeddings=MCA_embeddings,
                prefix=f'{prefix}states/{i:02d}/',
                load_saved=load_saved, device=device)
            for i in range(n_states)]

    if n_jobs is None or n_jobs < 1:
        n_jobs = n_states
    if n_jobs == 1:
        out_list = [get_model_kwargs(kwargs) for kwargs in kwargs_list]
    else:
        with multiprocessing.Pool(processes=n_jobs) as pool:
            out_list = pool.map(get_model_kwargs, kwargs_list)

    model_list = [out[0] for out in out_list]
    dataset_list = [out[1] for out in out_list]
    mask_size = dataset_list[0].mask.sum()

    cnts_range = np.stack([cnts_min, cnts_max], -1)
    cnts_range /= mask_size

    gra_size = 7
    embs_1 = pad(embs, gra_size)
    h, w = embs_1.shape[0], embs_1.shape[1]
    batch_size_row = gra_size
    n_batches_row = embs_1.shape[0] // batch_size_row

    batch_size_col = gra_size
    n_batches_col = embs_1.shape[1] // batch_size_col

    embs_batches = np.array_split(embs_1, n_batches_row, axis=0)

    embs_batches = [np.array_split(i, n_batches_col, axis=1) for i in embs_batches]
    del embs_1
    del embs

    locs1 = get_center_coordinates_rounded(h, w, gra_size)
    genes = get_patches_genes_test(locs1, en_locs, en_cnts, k=5)


    genes_split = [genes[i:i + n_batches_col] for i in range(0, len(genes), n_batches_col)]

    predict(h, w,
            model_states=model_list, x_batches=embs_batches, genes=genes_split,
            name_list=names, y_range=cnts_range,
            prefix=prefix, device=device, gra_size=gra_size)



def main(prefix, epoch=500, device='cuda', n_states=5, load_saved=False, enhance=False):
    embs, cnts, enhance_cnts, locs, enhance_locs = get_data(prefix, enhance=enhance)

    factor = 16
    ori_radius = int(read_string(f'{prefix}radius.txt'))
    radius = ori_radius / factor

    n_train = cnts.shape[0]
    batch_size = min(32, n_train//16)

    impute(
            embs=embs, cnts=cnts, locs=locs, en_cnts=enhance_cnts, en_locs=enhance_locs, radius=radius,
            epochs=epoch,
            batch_size=batch_size, n_states=n_states,
            prefix=prefix, MCA_embeddings=128, load_saved=load_saved,
            device=device, n_jobs=1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str, default='')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--n-states', type=int, default=5)

    args = parser.parse_args()

    main(args.prefix, epoch=args.epochs, n_states=args.n_states, enhance=True)