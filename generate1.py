import os
import argparse

import numpy as np
import torch
from runner import Runner
# from utils import smact_validity, compute_elem_type_num_wdist, get_structure, compute_density_wdist, structure_validity
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./result_19_node_transformer_01/model_239.pth', type=str, help='The directory for storing training outputs')
    # parser.add_argument('--dataset', type=str, default='perov_5', help='Dataset name, must be perov_5, carbon_24, or mp_20')
    parser.add_argument('--dataset', type=str, default='LatticeStiffness', help='Dataset name, must be perov_5, carbon_24, or mp_20, LatticeModulus, LatticeStiffness')
    parser.add_argument('--data_path', type=str, default='./material_data/', help='The directory for storing training outputs')
    parser.add_argument('--save_mat_path', type=str, default='generated_mat/not_decomp', help='The directory for storing training outputs')

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
    dataset = runner.load_data(data_path, args.dataset)


    gen_atom_types_list, gen_lengths_list, gen_angles_list, gen_frac_coords_list, edge_index_list, prop_list = runner.generate(args.num_gen, None, coord_num_langevin_steps=100)
    #print(gen_frac_coords_list)
    #input()    
    if not os.path.exists(args.save_mat_path):
        os.makedirs(args.save_mat_path)

    #print(gen_atom_types_list[0].shape)
    #print(gen_lengths_list[0].shape)
    #print(gen_angles_list[0].shape)
    #print(gen_frac_coords_list[0].shape)
    #print(prop_list[0].shape)
    #print(len(gen_frac_coords_list))
    
    print('Saving lattice...')
    for i in range(args.num_gen):
        lattice_name = os.path.join(args.save_mat_path, '{}_lattice_{}.npz'.format(args.dataset, i))
        print('Saving {}, atom_num {}'.format(lattice_name, gen_atom_types_list[i].shape[0]))
        #print(gen_frac_coords_list[i])
        #input()
        np.savez(lattice_name,
                atom_types=gen_atom_types_list[i],
                lengths=gen_lengths_list[i],
                angles=gen_angles_list[i],
                frac_coords=gen_frac_coords_list[i],
                #edge_index=edge_index_list[i],
                edge_index=np.array([]),
                prop_list=prop_list[i]
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