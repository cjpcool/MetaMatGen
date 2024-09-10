import os

import pandas as pd
import torch
from torch_scatter import scatter
from torch_geometric.data import DataLoader
import numpy as np
from model import MatGen
from datasets.dataset_truss import LatticeStiffness, LatticeModulus
from utils import StandardScalerTorch


class Runner():
    def __init__(self, conf, score_norm_path):
        self.conf = conf
        if score_norm_path is not None:
            score_norm = np.loadtxt(score_norm_path)
        else:
            score_norm = None
        self.model = MatGen(**conf['model'], score_norm=score_norm)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), **conf['optim'])

    def _get_normalizer(self, dataset):
        normalizer = StandardScalerTorch()
        lengths = dataset.lengths.view(-1,3)
        angles = dataset.angles.view(-1,3)
        length_angles = torch.cat((lengths, angles), dim=-1)
        normalizer.fit(length_angles)
        normalizer.means, normalizer.stds = normalizer.means.to('cuda'), normalizer.stds.to('cuda')
        return normalizer

    def _get_prop_normalizer(self, dataset):
        normalizer = StandardScalerTorch()
        y = dataset.y
        normalizer.fit(y)
        normalizer.means, normalizer.stds = normalizer.means.to('cuda'), normalizer.stds.to('cuda')
        return normalizer

    def _train_epoch(self, loader, epoch):
        self.model.train()
        total_node_num_loss, total_lattice_loss, total_coord_loss = 0, 0, 0
        total_kld_loss, total_kld_loss1, total_kld_loss2, total_kld_loss3 = 0.0, 0.0, 0.0, 0.0
        total_loss = 0

        total_dist_reg_loss = 0.0
        total_property_loss = 0.0
        total_pbc_sym_reg_loss = 0.0
        total_edge_pred_loss = 0.0

        for iter_num, data_batch in enumerate(loader):
            data_batch = data_batch.to('cuda')
            loss_dict = self.model(data_batch, temp=self.conf['train_temp'], distance_reg=self.conf['distance_reg'])

            kld_loss, node_num_loss, lattice_loss, coord_loss = loss_dict['kld_loss'], loss_dict['node_num_loss'], loss_dict['lattice_loss'], loss_dict['coord_loss']
            edge_pred_loss = loss_dict['edge_pred_loss']
            dist_reg_loss = loss_dict['dist_reg_loss']
            property_loss = loss_dict['property_loss']
            pbc_sym_reg_loss = loss_dict['pbc_sym_reg_loss']
            loss = self.conf['kld_weight'] * kld_loss + self.conf['node_num_loss_weight'] * node_num_loss \
                + self.conf['lattice_weight'] * lattice_loss + self.conf['coord_weight'] * coord_loss + self.conf['edge_pred_weight'] * edge_pred_loss + \
                self.conf['dist_reg_weight'] * dist_reg_loss + self.conf['property_weight'] * property_loss + self.conf['pbc_sym_reg_weight'] * pbc_sym_reg_loss

            # if epoch > 10 and (loss < 0.1 or loss > 100):
            #     self.optimizer.zero_grad()
            #     return None
                
            if torch.isnan(loss) or torch.isinf(loss):
                self.optimizer.zero_grad()
                print('Loss is NAN')
                print(
                    'NAN Loss in iter {} | loss kld {:.4f} lattice {:.4f} coord {:.4f}, edge {:4f}, min_dist_reg {:4f}, prop {:4f}, pbc_sym_reg_loss {:4f}, node_num_loss {:4f}'.format(
                        iter_num, kld_loss.to('cpu').item(),
                         lattice_loss.to('cpu').item(), coord_loss.to('cpu').item(),
                        edge_pred_loss.item(), dist_reg_loss, property_loss.item(), pbc_sym_reg_loss.item(),
                        node_num_loss.item()))
                return None

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.conf['max_grad_value'])
            self.optimizer.step()

            total_loss += loss.to('cpu').item()
            total_kld_loss += kld_loss.to('cpu').item()

            total_node_num_loss += node_num_loss.to('cpu').item()

            total_lattice_loss += lattice_loss.to('cpu').item()
            total_coord_loss += coord_loss.to('cpu').item()
            total_dist_reg_loss += dist_reg_loss.to('cpu').item()
            total_property_loss += property_loss.to('cpu').item()
            total_pbc_sym_reg_loss += pbc_sym_reg_loss.to('cpu').item()
            total_edge_pred_loss += edge_pred_loss.to('cpu').item()


            if 'kld_loss1' in loss_dict and 'kld_loss2' in loss_dict and 'kld_loss3' in loss_dict:
                kld_loss1, kld_loss2, kld_loss3 = loss_dict['kld_loss1'].to('cpu').item(), loss_dict['kld_loss2'].to('cpu').item(), loss_dict['kld_loss3'].to('cpu').item()
                total_kld_loss1 += kld_loss1
                total_kld_loss2 += kld_loss2
                total_kld_loss3 += kld_loss3
            else:
                kld_loss1, kld_loss2, kld_loss3 = 0.0, 0.0, 0.0

            if iter_num % self.conf['verbose'] == 0:
                # print('Training iteration {} | loss kld {:.4f} kld1 {:.4f} kld2 {:.4f} kld3 {:.4f} elem_type_num {:.4f} elem_type {:.4f} elem_num {:.4f} lattice {:.4f} coord {:.4f}'.format(iter_num, kld_loss.to('cpu').item(),
                #     kld_loss1, kld_loss2, kld_loss3, elem_type_num_loss.to('cpu').item(), elem_type_loss.to('cpu').item(), elem_num_loss.to('cpu').item(), lattice_loss.to('cpu').item(), coord_loss.to('cpu').item()))
                print('Training iteration {} | loss kld {:.4f} kld1 {:.4f} kld2 {:.4f} kld3 {:.4f} lattice {:.4f} coord {:.4f}, edge {:4f}, min_dist_reg {:4f}, prop {:4f}, pbc_sym_reg_loss {:4f}, node_num_loss {:4f}'.format(iter_num, kld_loss.to('cpu').item(),
                    kld_loss1, kld_loss2, kld_loss3, lattice_loss.to('cpu').item(), coord_loss.to('cpu').item(), edge_pred_loss.item(), dist_reg_loss, property_loss.item(), pbc_sym_reg_loss.item(), node_num_loss.item()))

        iter_num += 1
        return (total_loss / iter_num, total_kld_loss / iter_num, total_kld_loss1 / iter_num, total_kld_loss2 / iter_num, total_kld_loss3 / iter_num, \
            total_node_num_loss / iter_num, total_lattice_loss / iter_num, total_coord_loss / iter_num, total_edge_pred_loss / iter_num, \
                total_dist_reg_loss/ iter_num, total_property_loss / iter_num, total_pbc_sym_reg_loss / iter_num)
    

    def load_data(self, data_path, data_name,file_name=None):
        print(data_name)
        if data_name == 'LatticeModulus':
            if file_name is None:
                dataset = LatticeModulus(data_path)
            else:
                dataset = LatticeModulus(data_path, file_name=file_name)
        elif data_name == 'LatticeStiffness':
            if file_name is None:
                dataset = LatticeStiffness(data_path)
            else:
                dataset = LatticeStiffness(data_path, file_name=file_name)

        split_idx = dataset.get_idx_split(len(dataset), train_size=self.conf['train_size'], valid_size=self.conf['valid_size'], seed=self.conf['seed'])
        self.train_dataset, self.valid_dataset, self.test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]


    def train(self, data_path, val_data_path, out_path, load_model_path=None):
        torch.nn.init.constant_(self.model.fc_var.weight, 1e-10)
        torch.nn.init.constant_(self.model.fc_var.bias, 0.)
        torch.nn.init.constant_(self.model.fc_lattice_log_var[-1].weight, 1e-10)
        torch.nn.init.constant_(self.model.fc_lattice_log_var[-1].bias, 0.)

        loader = DataLoader(self.train_dataset, batch_size=self.conf['batch_size'], shuffle=True)
        self.model.lattice_normalizer = self._get_normalizer(self.train_dataset)
        self.model.prop_normalizer = self._get_prop_normalizer(self.train_dataset)

        val_loader = DataLoader(self.valid_dataset, batch_size=self.conf['batch_size'], shuffle=False)

        if load_model_path is not None:
            print('Loading model path from: {}'.format(load_model_path))
            self.model.load_state_dict(torch.load(load_model_path))

        end_epoch = self.conf['end_epoch']
        score_loss_list = []
        score_loss_list_val = []
        if self.conf['start_epoch'] != 0 and os.path.exists(os.path.join(out_path, 'train_score_loss.csv')):
            df = pd.read_csv(os.path.join(out_path, 'train_score_loss.csv'))
            score_loss_list.extend(df['score_loss'].to_list())
            score_loss_list_val.extend(df['score_loss_val'].to_list())
        for epoch in range(self.conf['start_epoch'], end_epoch+1):
            if epoch == self.conf['start_epoch']:
                last_optim_dict = self.optimizer.state_dict().copy()
                last_model_dict = self.model.state_dict().copy()
                last_last_optim_dict, last_last_model_dict = last_optim_dict, last_model_dict
            else:
                last_last_optim_dict, last_last_model_dict = last_optim_dict, last_model_dict
                last_optim_dict = self.optimizer.state_dict().copy()
                last_model_dict = self.model.state_dict().copy()


            train_returns = self._train_epoch(loader, epoch)
            
            retry_num = 0
            while train_returns is None and retry_num <= 5:
                retry_num += 1
                self.optimizer.load_state_dict(last_optim_dict)
                self.model.load_state_dict(last_model_dict)
                train_returns = self._train_epoch(loader, epoch)
            
            if train_returns is None:
                retry_num = 0
                while train_returns is None and retry_num <= 5:
                    retry_num += 1
                    self.optimizer.load_state_dict(last_last_optim_dict)
                    self.model.load_state_dict(last_last_model_dict)
                    train_returns = self._train_epoch(loader, epoch)
                if train_returns is None:
                    exit()

            avg_loss, avg_kld_loss, avg_kld_loss1, avg_kld_loss2, avg_kld_loss3, avg_node_num_loss, avg_lattice_loss, avg_coord_loss, avg_edge_pred_loss, avg_dist_reg_loss, avg_property_loss, avg_pbc_sym_reg_loss  = train_returns

            avg_val_loss, avg_val_kld_loss, avg_val_kld_loss1, avg_val_kld_loss2, avg_val_kld_loss3, avg_val_node_num_loss, avg_val_lattice_loss,\
                avg_val_coord_loss, avg_val_edge_pred_loss, avg_val_dist_reg_loss, avg_val_property_loss, avg_val_pbc_sym_reg_loss = self.valid(val_loader)
            print('Training Epoch {:d} | Average loss {:.4f} kld {:.4f} kld1 {:.4f} kld2 {:.4f} kld3 {:.4f} node_num {:.4f} lattice {:.4f} coord {:.4f} \
                edge_pred {:.4f} dist_reg {:.4f} property {:.4f} pbc_sym_reg {:.4f}'.format(epoch, avg_loss,
                avg_kld_loss, avg_kld_loss1, avg_kld_loss2, avg_kld_loss3, avg_node_num_loss, avg_lattice_loss, avg_coord_loss, avg_edge_pred_loss, avg_dist_reg_loss,avg_property_loss, avg_pbc_sym_reg_loss))
            print('Validation Epoch {:d} | Average loss {:.4f} kld {:.4f} kld1 {:.4f} kld2 {:.4f} kld3 {:.4f} node_num {:.4f} lattice {:.4f} coord {:.4f} \
                edge_pred {:.4f} dist_reg {:.4f} property {:.4f} pbc_sym_reg {:.4f}'.format(epoch, avg_val_loss, avg_val_kld_loss, avg_val_kld_loss1, avg_val_kld_loss2,
                avg_val_kld_loss3, avg_val_node_num_loss, avg_val_lattice_loss, avg_val_coord_loss, avg_val_edge_pred_loss, avg_val_dist_reg_loss, avg_val_property_loss, avg_val_pbc_sym_reg_loss))
            
            if out_path is not None:
                if (epoch + 1) % self.conf['save_interval'] == 0:
                    print('Saving checkpoint...')
                    torch.save(self.model.state_dict(), os.path.join(out_path, 'model_{}.pth'.format(epoch)))
                
                file_obj = open(os.path.join(out_path, 'train.txt'), 'a')
                file_obj.write('Training Epoch {:d} | Average loss {:.4f} kld {:.4f} kld1 {:.4f} kld2 {:.4f} kld3 {:.4f} node_num {:.4f} lattice {:.4f} coord {:.4f} \
                edge_pred {:.4f} dist_reg {:.4f} property {:.4f} pbc_sym_reg {:.4f}'.format(epoch, avg_loss,
                avg_kld_loss, avg_kld_loss1, avg_kld_loss2, avg_kld_loss3, avg_node_num_loss, avg_lattice_loss, avg_coord_loss, avg_edge_pred_loss, avg_dist_reg_loss,avg_property_loss, avg_pbc_sym_reg_loss))
                file_obj.close()

                file_obj = open(os.path.join(out_path, 'val.txt'), 'a')
                file_obj.write('Validation Epoch {:d} | Average loss {:.4f} kld {:.4f} kld1 {:.4f} kld2 {:.4f} kld3 {:.4f} node_num {:.4f} lattice {:.4f} coord {:.4f} \
                edge_pred {:.4f} dist_reg {:.4f} property {:.4f} pbc_sym_reg {:.4f}'.format(epoch, avg_val_loss, avg_val_kld_loss, avg_val_kld_loss1, avg_val_kld_loss2,
                avg_val_kld_loss3, avg_val_node_num_loss, avg_val_lattice_loss, avg_val_coord_loss, avg_val_edge_pred_loss, avg_val_dist_reg_loss, avg_val_property_loss, avg_val_pbc_sym_reg_loss))
                file_obj.close()
            score_loss_list.append(avg_coord_loss)
            score_loss_list_val.append(avg_val_coord_loss)

            if epoch % self.conf['save_interval'] == 0:
                df = pd.DataFrame({'score_loss': score_loss_list,'score_loss_val': score_loss_list_val})
                df.to_csv(os.path.join(out_path, 'train_score_loss.csv'), index=False)



    def valid(self, loader):
        self.model.eval()
        total_kld_loss, total_node_num_loss, total_lattice_loss, total_coord_loss = 0, 0, 0, 0
        total_kld_loss1, total_kld_loss2, total_kld_loss3 = 0.0, 0.0, 0.0
        total_dist_reg_loss = 0.0
        total_property_loss = 0.0
        total_pbc_sym_reg_loss = 0.0
        total_edge_pred_loss = 0.0
        total_loss = 0

        with torch.no_grad():
            for iter_num, data_batch in enumerate(loader):
                data_batch = data_batch.to('cuda')
                loss_dict = self.model(data_batch, temp=self.conf['val_temp'], eval=True, distance_reg=self.conf['distance_reg'])

                kld_loss, node_num_loss, lattice_loss, coord_loss = loss_dict['kld_loss'], loss_dict['node_num_loss'], loss_dict['lattice_loss'], loss_dict['coord_loss']
                edge_pred_loss = loss_dict['edge_pred_loss']
                dist_reg_loss = loss_dict['dist_reg_loss']
                property_loss = loss_dict['property_loss']
                pbc_sym_reg_loss = loss_dict['pbc_sym_reg_loss']
                loss = self.conf['kld_weight'] * kld_loss + self.conf['node_num_loss_weight'] * node_num_loss \
                       + self.conf['lattice_weight'] * lattice_loss + self.conf['coord_weight'] * coord_loss + \
                       self.conf['edge_pred_weight'] * edge_pred_loss + \
                       self.conf['dist_reg_weight'] * dist_reg_loss + self.conf['property_weight'] * property_loss + self.conf['pbc_sym_reg_weight'] * pbc_sym_reg_loss
                
                total_loss += loss.to('cpu').item()
                total_kld_loss += kld_loss.to('cpu').item()
                total_node_num_loss += node_num_loss.to('cpu').item()
                total_lattice_loss += lattice_loss.to('cpu').item()
                total_coord_loss += coord_loss.to('cpu').item()

                if 'kld_loss1' in loss_dict and 'kld_loss2' in loss_dict and 'kld_loss3' in loss_dict:
                    kld_loss1, kld_loss2, kld_loss3 = loss_dict['kld_loss1'].to('cpu').item(), loss_dict['kld_loss2'].to('cpu').item(), loss_dict['kld_loss3'].to('cpu').item()
                    total_kld_loss1 += kld_loss1
                    total_kld_loss2 += kld_loss2
                    total_kld_loss3 += kld_loss3
                else:
                    kld_loss1, kld_loss2, kld_loss3 = 0.0, 0.0, 0.0

                total_node_num_loss += loss_dict['node_num_loss'].to('cpu').item()
                total_lattice_loss += lattice_loss.to('cpu').item()
                total_coord_loss += coord_loss.to('cpu').item()
                total_dist_reg_loss += dist_reg_loss.to('cpu').item()
                total_property_loss += property_loss.to('cpu').item()
                total_pbc_sym_reg_loss += pbc_sym_reg_loss.to('cpu').item()
                total_edge_pred_loss += edge_pred_loss.to('cpu').item()

        iter_num += 1
        return (total_loss / iter_num, total_kld_loss / iter_num, total_kld_loss1 / iter_num, total_kld_loss2 / iter_num, total_kld_loss3 / iter_num, \
            total_node_num_loss / iter_num, total_lattice_loss / iter_num, total_coord_loss / iter_num, total_edge_pred_loss / iter_num, \
                total_dist_reg_loss/ iter_num, total_property_loss / iter_num, total_pbc_sym_reg_loss / iter_num)


    def generate(self, num_gen, data_path, coord_num_langevin_steps=100, coord_step_rate=1e-4, threshold=0.6):
        dataset = self.train_dataset
        normalizer = self._get_normalizer(dataset)
        self.model.lattice_normalizer = normalizer
        self.model.prop_normalizer = self._get_prop_normalizer(dataset)
        
        num_atoms_list, atom_types_list, lengths_list, angles_list, frac_coords_list = [], [], [], [], []
        prop_list = []
        edge_index_list = []
        num_remain = num_gen
        one_time_gen = self.conf['chunk_size']
        temperature = self.conf['gen_temp']
        coord_noise_start = self.conf['model']['noise_start']
        coord_noise_end = self.conf['model']['noise_end']
        coord_num_diff_steps = self.conf['model']['num_time_steps']
        min_num_atom, max_num_atom = self.conf['min_atom_num'], self.conf['max_atom_num']


        # num_atoms_each = [8, 8, 9, 10, 11, 12, 13,14]
        # num_atoms = []
        # while len(num_atoms) < num_gen:
        #     num_atoms = num_atoms + num_atoms_each
        # num_gen = min(num_gen, len(num_atoms))
        # num_atoms = num_atoms[:num_gen]
        # num_atoms = torch.LongTensor(num_atoms)

        self.model.eval()
        while num_remain > 0:
            if num_remain > one_time_gen:
                mat_arrays = self.model.generate(one_time_gen, temperature, coord_noise_start, coord_noise_end, coord_num_diff_steps, coord_num_langevin_steps, coord_step_rate,None, min_num_atom, max_num_atom, threshold=threshold)
            else:
                mat_arrays = self.model.generate(num_remain, temperature, coord_noise_start, coord_noise_end, coord_num_diff_steps, coord_num_langevin_steps, coord_step_rate,None, min_num_atom, max_num_atom, threshold=threshold)
            
            num_atoms_list.append(mat_arrays[0].detach().cpu())
            atom_types_list.append(mat_arrays[1].detach().cpu())
            lengths_list.append(mat_arrays[2].detach().cpu())
            angles_list.append(mat_arrays[3].detach().cpu())
            frac_coords_list.append(mat_arrays[4].detach().cpu())
            edge_index_list.append(mat_arrays[5].detach().cpu())
            prop_list.append(mat_arrays[6].detach().cpu())


            num_mat = len(mat_arrays[0])
            num_remain -= num_mat
            print('{} materials are generated!'.format(num_gen - num_remain))
        
        all_num_atoms = torch.cat(num_atoms_list, dim=0)
        all_atom_types = torch.cat(atom_types_list, dim=0)
        all_lengths = torch.cat(lengths_list, dim=0)
        all_angles = torch.cat(angles_list, dim=0)
        all_frac_coords = torch.cat(frac_coords_list, dim=0)
        all_props = torch.cat(prop_list, dim=0)

        out_edge_index_list = []

        for idx, all_edge_index in enumerate(edge_index_list):
            batch = torch.cat([torch.ones(size=(all_num_atoms[i],), dtype=torch.long, device=all_edge_index.device) * i for i in range(num_gen)], dim=0)
            edge_num_per_node = scatter(
                torch.ones(size=(all_edge_index.shape[1],), device=all_edge_index.device).long(), all_edge_index[0],
                dim_size=len(batch), reduce='sum')
            edge_num_per_graph = scatter(edge_num_per_node, batch, reduce='sum', dim_size=num_gen)
            indices_per_graph = torch.cumsum(edge_num_per_graph, dim=0)
            indices_per_graph = torch.cat([torch.zeros((1,), dtype=torch.long), indices_per_graph], dim=0)
            node_num_indices_per_graph = torch.cat([torch.zeros((1,), dtype=torch.long), all_num_atoms], dim=0)
            node_num_indices_per_graph = torch.cumsum(node_num_indices_per_graph, dim=0)
            for idx_i in range(indices_per_graph.shape[0] - 1):
                edge_index_per_graph = all_edge_index.narrow(1, indices_per_graph[idx_i], edge_num_per_graph[idx_i]) - \
                                       node_num_indices_per_graph[idx_i]
                out_edge_index_list.append(edge_index_per_graph)



        atom_types_list, lengths_list, angles_list, frac_coords_list = [], [], [], []
        prop_list = []
        start_idx = 0
        for idx, num_atom in enumerate(all_num_atoms.tolist()):
            atom_types = all_atom_types.narrow(0, start_idx, num_atom).numpy()
            lengths = all_lengths[idx].numpy()
            angles = all_angles[idx].numpy()
            frac_coords = all_frac_coords.narrow(0, start_idx, num_atom).numpy()

            prop_list.append(all_props[idx].numpy())
            atom_types_list.append(atom_types)
            lengths_list.append(lengths)
            angles_list.append(angles)
            frac_coords_list.append(frac_coords)

            start_idx += num_atom

        return atom_types_list, lengths_list, angles_list, frac_coords_list, out_edge_index_list, prop_list

    def recon(self, data_batch, num_gen, data_path, coord_num_langevin_steps=100, coord_step_rate=1e-4, threshold=0.6):
        dataset = self.train_dataset
        normalizer = self._get_normalizer(dataset)
        self.model.lattice_normalizer = normalizer
        self.model.prop_normalizer = self._get_prop_normalizer(dataset)


        num_atoms_list, atom_types_list, lengths_list, angles_list, frac_coords_list = [], [], [], [], []
        prop_list = []
        edge_index_list = []
        num_remain = num_gen
        one_time_gen = self.conf['chunk_size']
        temperature = self.conf['gen_temp']
        coord_noise_start = self.conf['model']['noise_start']
        coord_noise_end = self.conf['model']['noise_end']
        coord_num_diff_steps = self.conf['model']['num_time_steps']
        min_num_atom, max_num_atom = self.conf['min_atom_num'], self.conf['max_atom_num']

        print(coord_num_diff_steps)

        num_graph = data_batch.batch[-1].item() + 1

        num_atoms = data_batch.num_atoms
        # num_atoms = num_atoms[:num_gen]
        # num_atoms = torch.LongTensor(num_atoms)

        self.model.eval()
        while num_remain > 0:
            if num_remain > one_time_gen:
                mat_arrays = self.model.recon(data_batch, one_time_gen, temperature, coord_noise_start, coord_noise_end,
                                                 coord_num_diff_steps, coord_num_langevin_steps, coord_step_rate,
                                                 num_atoms, min_num_atom, max_num_atom,threshold=threshold)
            else:
                mat_arrays = self.model.recon(data_batch, num_remain, temperature, coord_noise_start, coord_noise_end,
                                                 coord_num_diff_steps, coord_num_langevin_steps, coord_step_rate,
                                                 num_atoms, min_num_atom, max_num_atom, threshold=threshold)

            num_atoms_list.append(mat_arrays[0].detach().cpu())
            atom_types_list.append(mat_arrays[1].detach().cpu())
            lengths_list.append(mat_arrays[2].detach().cpu())
            angles_list.append(mat_arrays[3].detach().cpu())
            frac_coords_list.append(mat_arrays[4].detach().cpu())
            edge_index_list.append(mat_arrays[5].detach().cpu())
            if mat_arrays[6] is not None:
                prop_list.append(mat_arrays[6].detach().cpu())

            num_mat = len(mat_arrays[0])
            num_remain -= num_mat
            print('{} materials are generated!'.format(num_gen - num_remain))

        all_num_atoms = torch.cat(num_atoms_list, dim=0)
        all_atom_types = torch.cat(atom_types_list, dim=0)
        all_lengths = torch.cat(lengths_list, dim=0)
        all_angles = torch.cat(angles_list, dim=0)
        all_frac_coords = torch.cat(frac_coords_list, dim=0)

        if len(prop_list) > 0:
            all_props = torch.cat(prop_list, dim=0)
        else:
            all_props = data_batch.y.cpu()


        atom_types_list, lengths_list, angles_list, frac_coords_list = [], [], [], []
        prop_list1 = []
        out_edge_index_list = []

        num_atoms = num_atoms.cpu()
        for idx, all_edge_index in enumerate(edge_index_list):
            batch = torch.cat([torch.ones(size=(all_num_atoms[i],), dtype=torch.long, device=all_edge_index.device) * i for i in range(num_gen)], dim=0)
            edge_num_per_node = scatter(
                torch.ones(size=(all_edge_index.shape[1],), device=all_edge_index.device).long(), all_edge_index[0],
                dim_size=len(batch), reduce='sum')
            edge_num_per_graph = scatter(edge_num_per_node, batch, reduce='sum', dim_size=num_gen)
            indices_per_graph = torch.cumsum(edge_num_per_graph, dim=0)
            indices_per_graph = torch.cat([torch.zeros((1,), dtype=torch.long), indices_per_graph], dim=0)
            node_num_indices_per_graph = torch.cat([torch.zeros((1,), dtype=torch.long), all_num_atoms], dim=0)
            node_num_indices_per_graph = torch.cumsum(node_num_indices_per_graph, dim=0)
            for idx_i in range(indices_per_graph.shape[0] - 1):
                edge_index_per_graph = all_edge_index.narrow(1, indices_per_graph[idx_i], edge_num_per_graph[idx_i]) - \
                                       node_num_indices_per_graph[idx_i]
                out_edge_index_list.append(edge_index_per_graph)

        start_idx = 0
        for idx, num_atom in enumerate(all_num_atoms.tolist()):
            atom_types = all_atom_types.narrow(0, start_idx, num_atom).numpy()
            lengths = all_lengths[idx].numpy()
            angles = all_angles[idx].numpy()
            frac_coords = all_frac_coords.narrow(0, start_idx, num_atom).numpy()

            prop_list1.append(all_props[idx].numpy())
            atom_types_list.append(atom_types)
            lengths_list.append(lengths)
            angles_list.append(angles)
            frac_coords_list.append(frac_coords)

            start_idx += num_atom

        return atom_types_list, lengths_list, angles_list, frac_coords_list, out_edge_index_list, prop_list1