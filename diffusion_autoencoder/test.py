import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import torch
from tqdm import tqdm

import utils, dataset
from encoder import Encoder

device = torch.device('cuda:0')


def main():

    VAE_experiment_name = "Jun23_11-47_7000epochs_128latent_enc128_dec512_globalnorm"
    _, _, encoder, _ = utils.reload_model(None, None, VAE_experiment_name, 'best_train', device)


    split = 2
    match split:
        case 0:
            set_GRASP_CODE = dataset.Dataset_Latent_grasp_and_code('train')
            set_GRASP_PC = dataset.Dataset_Latent_grasp_and_pc('train')
        case 1:
            set_GRASP_CODE = dataset.Dataset_Latent_grasp_and_code('val')
            set_GRASP_PC = dataset.Dataset_Latent_grasp_and_pc('val')
        case 2:
            set_GRASP_CODE = dataset.Dataset_Latent_grasp_and_code('test')
            set_GRASP_PC = dataset.Dataset_Latent_grasp_and_pc('test')

    print(len(set_GRASP_CODE))
    print(len(set_GRASP_PC))

    for object_index in tqdm(range(len(set_GRASP_CODE))):
        
        code = set_GRASP_CODE[object_index]['code'].to(device).unsqueeze(0)
        pc = set_GRASP_PC[object_index]['point_cloud'].to(device).unsqueeze(0)
        
        mean, log_variance = encoder(pc)
        mean = mean.squeeze()
        
        max_diff = torch.abs(mean - code).max().item()
        
        if max_diff > 0.000001:
            print(f"Index {object_index:02d} mismatch! Max absolute difference: {max_diff:.6f}")

if __name__ == "__main__":
    main()