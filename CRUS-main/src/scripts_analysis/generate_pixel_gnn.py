import argparse
import os
import sys
import time
import warnings
from glob import glob

import numpy as np
import yaml
from tqdm import tqdm

sys.path.append('../')
from utils.attribute_hashmap import AttributeHashmap
from utils.parse import parse_settings

warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to config yaml file.', required=True)
    parser.add_argument('-o', '--overwrite', action='store_true', help='If true, overwrite previously computed results.')
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config.config_file_name = args.config
    config = parse_settings(config, log_settings=False)

    files_folder = '%s/%s' % (config.output_save_path, 'numpy_files')
    np_files_path = sorted(glob('%s/%s' % (files_folder, '*.npz')))

    save_path_numpy = '%s/%s' % (config.output_save_path, 'numpy_files_seg_pixel_gnn')
    os.makedirs(save_path_numpy, exist_ok=True)
    num_workers = 1

    dice_list = []
    for image_idx in tqdm(range(len(np_files_path))):
        load_path = np_files_path[image_idx]
        save_path = '%s/%s' % (save_path_numpy, os.path.basename(np_files_path[image_idx]))

        if os.path.exists(save_path) and not args.overwrite:
            print('File already exists: %s' % save_path)
            print('Skipping this file. If want to recompute and overwrite, use `-o`/`--overwrite`.')
            continue

        numpy_array = np.load(load_path)
        image = numpy_array['image']
        label_true = numpy_array['label']

        image = (image + 1) / 2

        H, W = label_true.shape[:2]
        latent = image.reshape(H * W, -1)  # 将图像像素数据展平成特征矩阵
        C = latent.shape[-1]

        from helper_generate_GNN import generate_gcn
        dice_score, label_pred, seg_pred = generate_gcn((H, W, C), latent, label_true, num_workers=num_workers)

        with open(save_path, 'wb+') as f:
            np.savez(f, image=image, label=label_true, latent=latent, label_gcn=label_pred, seg_gcn=seg_pred)

        print('SUCCESS! %s, dice: %s' % (load_path.split('/')[-1], dice_score))
        dice_list.append(dice_score)

    print('All pixel-based GNN results generated.')
    print('Dice: %.3f ± %.3f.' % (np.mean(dice_list), np.std(dice_list) / np.sqrt(len(dice_list))))

    os._exit(0)
