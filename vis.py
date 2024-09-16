import numpy as np
from utils.lattice_utils import plot_lattice
import os
from utils.mat_utils import frac_to_cart_coords, get_pbc_cutoff_graphs
import torch

def plot_origin_lattice_from_path(path, name, cutoff=1.0,max_num_neighbors_threshold=3, save_dir=None):
    full_path = os.path.join(path,name)
    lattice_npz = np.load(full_path)
    frac_coords = lattice_npz['origin_frac_coords']
    lengths = lattice_npz['origin_lengths']
    angles = lattice_npz['origin_angles']
    atom_types = lattice_npz['origin_atom_types']
    num_atoms = torch.tensor([atom_types.shape[0]])
    frac_coords, lengths, angles = torch.from_numpy(frac_coords),torch.from_numpy(lengths), torch.from_numpy(angles)
    cart_coords = frac_to_cart_coords(frac_coords,lengths,
                                      angles, num_atoms)
    print('num_atoms', num_atoms)
    try:
        # raise Exception
        edge_index = lattice_npz['origin_edge_index']
    except:
        edge_index, _,_ = get_pbc_cutoff_graphs(cart_coords, lengths, angles, num_atoms, cutoff=cutoff, max_num_neighbors_threshold=max_num_neighbors_threshold)
    print('edge_index \n', edge_index)
    plot_lattice(cart_coords,edge_index.T, save_dir=save_dir)


def plot_lattice_from_path(path, name, cutoff=1.0,max_num_neighbors_threshold=3, save_dir=None ):
    full_path = os.path.join(path,name)
    lattice_npz = np.load(full_path)
    frac_coords = lattice_npz['frac_coords']
    lengths = lattice_npz['lengths']
    angles = lattice_npz['angles']
    atom_types = lattice_npz['atom_types']
    num_atoms = torch.tensor([atom_types.shape[0]])
    frac_coords, lengths, angles = torch.from_numpy(frac_coords),torch.from_numpy(lengths).unsqueeze(0), torch.from_numpy(angles).unsqueeze(0)
    cart_coords = frac_to_cart_coords(frac_coords, lengths,
                                      angles, num_atoms)
    # cart_coords = frac_coords

    print('num_atoms', num_atoms)
    try:
        # raise Exception
        edge_index = lattice_npz['edge_index']
    except:
        edge_index, _,_ = get_pbc_cutoff_graphs(cart_coords, lengths, angles, num_atoms, cutoff=cutoff, max_num_neighbors_threshold=max_num_neighbors_threshold)
    print('edge_index \n', edge_index)
    plot_lattice(cart_coords,edge_index.T, save_dir=save_dir)



if __name__ == '__main__':

    path = './generated_mat/new_data'
    file_names = os.listdir(path)
    save_path = './vis/generated_mat/new_data'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for file_name in file_names:
        save_dir = os.path.join(save_path,file_name[:-3]+'jpeg')
        plot_lattice_from_path(path, file_name,cutoff=1., max_num_neighbors_threshold=5, save_dir=save_dir)
    # plot_origin_lattice_from_path(path, name,cutoff=1., max_num_neighbors_threshold=5)
