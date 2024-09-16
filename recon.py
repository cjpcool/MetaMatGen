import os
import argparse

import numpy as np
import torch
from torch_geometric.data import DataLoader

from runner import Runner
# from utils import smact_validity, compute_elem_type_num_wdist, get_structure, compute_density_wdist, structure_validity
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='D:/mine/researches/codes/result_0/model_999.pth', type=str, help='The directory for storing training outputs')
    # parser.add_argument('--dataset', type=str, default='perov_5', help='Dataset name, must be perov_5, carbon_24, or mp_20')
    parser.add_argument('--dataset', type=str, default='LatticeStiffness', help='Dataset name, must be perov_5, carbon_24, or mp_20, LatticeModulus, LatticeStiffness')
    parser.add_argument('--data_path', type=str, default='/home/jianpengc/datasets/metamaterial/', help='The directory for storing training outputs')
    parser.add_argument('--save_mat_path', type=str, default='recon_mat/new_data', help='The directory for storing training outputs')

    parser.add_argument('--num_gen', type=int, default=10, help='Number of materials to generate')

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



    runner = Runner(conf, score_norm_path)
    runner.model.load_state_dict(torch.load(args.model_path))
    runner.load_data(data_path, args.dataset, file_name='training')
    loader = DataLoader(runner.train_dataset, batch_size=args.num_gen, shuffle=True)
    data_batch = next(iter(loader))
    data_batch = data_batch.to('cuda')

    gen_atom_types_list, gen_lengths_list, gen_angles_list, gen_frac_coords_list, edge_index_list, prop_list = runner.recon(data_batch, args.num_gen, None, coord_num_langevin_steps=100, threshold=0.5)

    if not os.path.exists(args.save_mat_path):
        os.makedirs(args.save_mat_path)

    # print(edge_index_list)
    print('Saving lattice...')
    for i in range(args.num_gen):
        lattice_name = os.path.join(args.save_mat_path, '{}_lattice_{}.npz'.format(args.dataset, i))
        print('Saving {}, atom_num {}'.format(lattice_name, gen_atom_types_list[i].shape[0]))
        np.savez(lattice_name,
                atom_types=gen_atom_types_list[i],
                lengths=gen_lengths_list[i],
                angles=gen_angles_list[i],
                frac_coords=gen_frac_coords_list[i],
                edge_index=edge_index_list[i],
                prop_list = prop_list[i],

                origin_atom_types=data_batch[i].node_feat.cpu().numpy(),
                origin_lengths=data_batch[i].lengths.cpu().numpy(),
                origin_angles=data_batch[i].angles.cpu().numpy(),
                origin_frac_coords=data_batch[i].frac_coords.cpu().numpy(),
                origin_edge_index=data_batch[i].edge_index.cpu().numpy(),
                origin_prop = data_batch[i].y.cpu().numpy()
                )


    # is_valid, validity = smact_validity(gen_atom_types_list)
    # print("composition validity: {}".format(validity))
    #
    # elem_type_num_wdist = compute_elem_type_num_wdist(gen_atom_types_list, gt_atom_types_list)
    # print("element EMD: {}".format(elem_type_num_wdist))
    #
    # gen_structure_list = get_structure(gen_atom_types_list, gen_lengths_list, gen_angles_list, gen_frac_coords_list)
    #
    # is_valid, structure_validity = structure_validity(gen_atom_types_list, gen_lengths_list, gen_angles_list, gen_frac_coords_list, gen_structure_list)
    # print("structure validity: {}".format(structure_validity))
    #
    # density_wdist = compute_density_wdist(gen_structure_list, gt_structure_list)
    # print("density EMD: {}".format(density_wdist))