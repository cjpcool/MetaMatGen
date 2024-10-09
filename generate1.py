import os
import argparse

import numpy as np
import torch
from runner import Runner
from vis import plot_lattice_from_path

# from utils import smact_validity, compute_elem_type_num_wdist, get_structure, compute_density_wdist, structure_validity
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./result_47/model_999.pth', type=str, help='The directory for storing training outputs')
    # parser.add_argument('--dataset', type=str, default='perov_5', help='Dataset name, must be perov_5, carbon_24, or mp_20')
    parser.add_argument('--dataset', type=str, default='LatticeStiffness', help='Dataset name, must be perov_5, carbon_24, or mp_20, LatticeModulus, LatticeStiffness')
    parser.add_argument('--data_path', type=str, default='/home/jianpengc/datasets/metamaterial/', help='The directory for storing training outputs')
    parser.add_argument('--save_mat_path', type=str, default='generated_mat/47node', help='The directory for storing training outputs')

    parser.add_argument('--num_gen', type=int, default=100, help='Number of materials to generate')

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
    runner.load_data(data_path, args.dataset, file_name='training_node_num9')
    # codes to sample some conditions as input

    num_gen = args.num_gen
    ys = np.zeros((num_gen, len(runner.train_dataset[0]['y'])))
    for i in range(num_gen):
        ind = np.random.randint(0, len(runner.train_dataset))
        ys[i] = np.array(runner.train_dataset[ind]['y'])
    # np.savetxt('ys.csv', ys, delimiter=',')
    # print('finished saving y')

    cond = torch.tensor(ys).to('cuda:0').float()

    gen_atom_types_list, gen_lengths_list, gen_angles_list, gen_frac_coords_list, edge_index_list, prop_list = runner.generate(args.num_gen, None, coord_num_langevin_steps=100, cond=cond)
    print(edge_index_list)
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
                edge_index=edge_index_list[i],
                # edge_index=np.array([]),
                prop_list=prop_list[i]
                )

    print('Vis saving...')
    path = args.save_mat_path
    file_names = os.listdir(path)
    save_path = './vis/generated_mat/47node'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for file_name in file_names:
        save_dir = os.path.join(save_path, file_name[:-3] + 'jpeg')
        plot_lattice_from_path(path, file_name, save_dir=save_dir)
