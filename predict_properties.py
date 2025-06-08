import os
import argparse

import numpy as np
import torch
import shutil
from runner import Runner

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
    frac_coords = []
    edge_index = []
    section = None

    with open("input_structure.txt", "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == "frac_coords:":
                section = "frac_coords"
                continue
            elif line == "edge_index:":
                section = "edge_index"
                continue
            if section == "frac_coords":
                frac_coords.append([float(x) for x in line.split()])
            elif section == "edge_index":
                edge_index.append([int(x) for x in line.split()])
    frac_coords_tensor = torch.tensor(frac_coords, dtype=torch.float32).unsqueeze(0)
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).unsqueeze(0)
    density = torch.tensor([0.192]).repeat(num_gen,1).to(args.device).float()
    properties = runner.predict_properties(frac_coords_tensor, edge_index_tensor, density)
    
    print('Saving lattice...')
    for i in range(args.num_gen):
        lattice_name = os.path.join('output_properties_{}.txt'.format(i))
        print('{} saved.'.format(lattice_name))
        with open(lattice_name, "w") as f:
            f.write("properties:\n")
            np.savetxt(f, properties[i].cpu(), fmt="%.6f")

