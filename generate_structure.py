import os
import argparse

import numpy as np
import torch
import shutil
from runner import Runner

def reshape_edges(edge_index_list):
    edge_list = [[] for _ in range(gen_coords_list.shape[0])]
    for i in range(len(edge_index_list[0])):
        index = int((edge_index_list[0][i] + edge_index_list[1][i])/2/24)
        edge_list[index].append([edge_index_list[0][i].item()-24*index, edge_index_list[1][i].item()-24*index])
    #print(edge_list)
    return edge_list

# from utils import smact_validity, compute_elem_type_num_wdist, get_structure, compute_density_wdist, structure_validity
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./result_0607/model_64.pth', type=str, help='The directory for storing training outputs')
    # parser.add_argument('--dataset', type=str, default='perov_5', help='Dataset name, must be perov_5, carbon_24, or mp_20')
    parser.add_argument('--dataset', type=str, default='LatticeModulus', help='Dataset name, must be perov_5, carbon_24, or mp_20, LatticeModulus, LatticeStiffness')
    parser.add_argument('--data_path', type=str, default='/data/home/wzhan24/materialgen/material_data/', help='The directory for storing training outputs')
    parser.add_argument('--save_mat_path', type=str, default='generated_mat/', help='The directory for storing training outputs')

    parser.add_argument('--num_gen', type=int, default=1, help='Number of materials to generate')
    parser.add_argument('--device', type=str, default='cuda:1')

    args = parser.parse_args()


    assert args.dataset in ['perov_5', 'carbon_24', 'mp_20', 'LatticeModulus', 'LatticeStiffness'], "Not supported dataset"


    if args.dataset in ['perov_5', 'carbon_24', 'mp_20']:
        train_data_path = os.path.join('data', args.dataset, 'train.pt')
        if not os.path.isfile(train_data_path):
            train_data_path = os.path.join('data', args.dataset, 'train.csv')

        test_data_path = os.path.join('data', args.dataset, 'test.pt')
        if not os.path.isfile(test_data_path):
            train_data_path = os.path.join('data', args.dataset, 'test.csv')

        if args.dataset == 'perov_5':
            from config.perov_5_config_dict import conf
        elif args.dataset == 'carbon_24':
            from config.carbon_24_config_dict import conf
        else:
            from config.mp_20_config_dict import conf

        score_norm_path = os.path.join('data', args.dataset, 'score_norm.txt')


    elif args.dataset in ['LatticeModulus', 'LatticeStiffness']:
        data_path = os.path.join(args.data_path, args.dataset)
        if args.dataset == 'LatticeModulus':
            from config.LatticeModulus_config_dict import conf
        elif args.dataset == 'LatticeStiffness':
            from config.LatticeStiffness_config_dict import conf

        train_data_path, val_data_path = None, None
        score_norm_path = None
    
    print('loading model...')
    runner = Runner(conf, score_norm_path)
    runner.model.load_state_dict(torch.load(args.model_path))
    print('loading data...')
    runner.load_data(data_path, args.dataset, file_name='data')

    num_gen = args.num_gen
    with open("input_property.txt", "r") as f:
        line = f.readline().strip()
    values = [float(x) for x in line.strip("[]").split(",")]
    cond = torch.tensor(values, dtype=torch.float32).repeat(num_gen,1).to(args.device).float()
    #cond = torch.tensor([0.211,0.180,0.25,0.0747,0.0776,0.0956,0.3026,0.2578,0.2516,0.3506,0.3277,0.276]).repeat(num_gen,1).to(args.device).float()
    density = torch.tensor([0.192]).repeat(num_gen,1).to(args.device).float()
    gen_coords_list, edge_index_list = runner.generate(cond=cond, density=density, latent_dim=16)
    gen_coords_list = gen_coords_list[0]
    edge_index_list = edge_index_list[0]
    edge_index_list = reshape_edges(edge_index_list)
    if not os.path.exists(args.save_mat_path):
        os.makedirs(args.save_mat_path)
    
    print('Saving lattice...')
    for i in range(args.num_gen):
        lattice_name = os.path.join('output_structure_{}.txt'.format(i))
        print('{} saved.'.format(lattice_name))
        with open(lattice_name, "w") as f:
            f.write("frac_coords:\n")
            np.savetxt(f, gen_coords_list[i], fmt="%.6f")
            f.write("\nedge_index:\n")
            np.savetxt(f, edge_index_list[i], fmt="%d")

