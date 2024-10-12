import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
from torch_cluster import radius_graph
from torch_scatter import scatter
from tqdm import tqdm
from traitlets import Int

from .features import device
from .spherenet_light import SphereNetLightDecoder
from .Nequip.nequip_decoder import NequipDecoder
from .modules import build_mlp, aggregate_to_node, res_mlp, transformer, transformer_cond
import sys
sys.path.append("..")
from utils import get_pbc_cutoff_graphs, frac_to_cart_coords, cart_to_frac_coords, correct_cart_coords, \
    get_pbc_distances, align_gt_cart_coords, calculate_to_jimages_efficient, lattice_params_to_matrix_torch

EPS = 1e-8

class CoordGen(torch.nn.Module):
    def __init__(self, backbone_params, latent_dim, num_fc_hidden_layers, fc_hidden_dim, num_time_steps, noise_start, noise_end, cutoff, max_num_neighbors, num_node, 
                 loss_type='per_node', score_upper_bound=None, use_gpu=True, score_norm=None,
                 property_loss=False, backbone_name='spherenet', property_dim=21):
        super(CoordGen, self).__init__()
        self.property_loss = property_loss

        if backbone_name == 'spherenet':
            self.backbone = SphereNetLightDecoder(**backbone_params)
        elif backbone_name == 'nequip':
            self.backbone = NequipDecoder(**backbone_params)
        elif backbone_name == 'transformer':
            self.backbone = transformer_cond(num_node=num_node, d_model=latent_dim, n_head=8, ffn_hidden=128, n_layers=3,
                                        drop_prob=0.1, cond_dim=property_dim)

        if not property_loss:
            property_dim = 0
        else:
            property_dim = latent_dim
        #self.fc_score = build_mlp(latent_dim + backbone_params['hidden_channels']+ property_dim, fc_hidden_dim, num_fc_hidden_layers, 1) #ini
        #self.fc_score = res_mlp(num_node)
        # self.fc_score = transformer(num_node=num_node,d_model=128,n_head=8,ffn_hidden=128,n_layers=3,drop_prob=0.1)
        self.fc_score = build_mlp(latent_dim, fc_hidden_dim, fc_num_layers=0, out_dim=3)
        # self.fc_score = aggregate_to_node(latent_dim + backbone_params['hidden_channels']+ property_dim, fc_hidden_dim, 3, num_fc_hidden_layers)
        if backbone_name == 'spherenet':
            self.edge_pred = SphereNetLightDecoder(**backbone_params)
        elif backbone_name == 'nequip':
            self.edge_pred = NequipDecoder(**backbone_params)
        elif backbone_name == 'transformer':
            self.edge_pred = transformer_cond(num_node=num_node, d_model=latent_dim, n_head=8, ffn_hidden=128, n_layers=3,
                                        drop_prob=0.1)

        # TODO: fc_edge_lin not used in current version
        self.fc_edge_lin = build_mlp(latent_dim, fc_hidden_dim, 1, latent_dim)

        self.fc_edge_prob = build_mlp(latent_dim, fc_hidden_dim, 0, latent_dim)
        self.binlin = nn.Bilinear(latent_dim, latent_dim, 1)

        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.num_node = num_node
        self.use_gpu = use_gpu
        self.score_norms = None
        if score_norm is not None:
            self.score_norms = torch.from_numpy(score_norm).float()
        
        sigmas = self._get_noise_params(num_time_steps, noise_start, noise_end)
        self.sigmas = nn.Parameter(sigmas, requires_grad=False)
        self.num_time_steps = num_time_steps
        self.pool = mp.Pool(16)
        self.score_upper_bound = score_upper_bound
        self.loss_type = loss_type
        if use_gpu:
            self.backbone = self.backbone.to('cuda')
            self.fc_score = self.fc_score.to('cuda')
            self.sigmas.data = self.sigmas.to('cuda')
            self.edge_pred = self.edge_pred.to('cuda')
            self.fc_edge_prob = self.fc_edge_prob.to('cuda')
            self.fc_edge_lin = self.fc_edge_lin.to('cuda')
            self.binlin = self.binlin.to("cuda")
            if score_norm is not None:
                self.score_norms = self.score_norms.to('cuda')


    def _get_noise_params(self, num_time_steps, noise_start, noise_end):
        log_sigmas = np.linspace(np.log(noise_start), np.log(noise_end), num_time_steps)
        sigmas = np.exp(log_sigmas)
        sigmas = torch.from_numpy(sigmas).float()
        return sigmas

    def _center_coords(self, coords, batch):
        coord_center = scatter(coords, batch, reduce='mean', dim=0)
        return coords - coord_center[batch]

    '''def shrink_coords(self,ini_coords,batch_size):
        reshaped_coords = ini_coords.view(batch_size,-1,3)
        num_node = reshaped_coords.shape[1]
        result = torch.zeros((batch_size,5,3))
        for i in range(batch_size):
            corners, selected_corners = self.find_corners_index(reshaped_coords[i,:,:])
            for j in range(4):
                result[i,j,:] = reshaped_coords[i,selected_corners[j],:]
            position = 4
            for j in range(num_node):
                if j not in corners:
                    result[i,position,:] = reshaped_coords[i,j,:]
                    position += 1
        return result
        
    def find_corners_index(self,coords):
        corners = []  #index of 4 corners
        center = torch.mean(coords,dim=0)
        for i in range(9):
            if torch.norm(coords[i,:] - center) < 1e-6:
                continue
            corners.append(i)
        selected_corners = [0,1] #4 selected corners to represent the lattice
        one_axis = coords[0,:] - coords[1,:]
        for i in range(2,8):
            for j in range(2,8):
                if torch.norm((coords[i,:] - coords[j,:]) - one_axis) < 1e-6:
                    if torch.norm((coords[0,:]+coords[1,:]+coords[i,:]+coords[j,:])/4 - center) > 1e-6:
                        selected_corners.append(i)
                        for k in range(2,8):
                            if torch.norm(coords[j,:] + coords[k,:] - 2*center) < 1e-6:
                                selected_corners.append(k)
                                return corners, selected_corners
        return corners, [0,1,2,3]'''
    
    def find_corners_index(self,coords):
        corners = []  #index of 4 corners
        center = torch.mean(coords,dim=0)
        for i in range(self.num_node):
            if torch.norm(coords[i,:] - center) < 1e-6:
                continue
            corners.append(i)
        selected_corners = [corners[0]] #4 selected corners to represent the lattice
        for i in range(1,8):
            for j in range(1,8):
                for k in range(1,8):
                    if torch.norm((coords[corners[0],:]-coords[corners[i],:]-coords[corners[j],:]+coords[corners[k],:])) < 1e-6:
                        if torch.norm((coords[corners[0],:]+coords[corners[i],:]+coords[corners[j],:]+coords[corners[k],:])-4*center) > 1e-6:
                            for h in range(1,8):
                                if torch.norm((coords[corners[k],:]+coords[corners[h],:])-2*center) < 1e-6:
                                    return corners, selected_corners + [corners[i],corners[j],corners[h]]
        return corners, [0,1,2,3]

    def shrink_coords(self,ini_coords,batch_size):
            reshaped_coords = ini_coords.view(batch_size,-1,3)
            num_node = reshaped_coords.shape[1]
            result = torch.zeros((batch_size,5,3))
            for i in range(batch_size):
                corners, selected_corners = self.find_corners_index(reshaped_coords[i,:,:])
                #print(corners)
                #print(selected_corners)
                for j in range(4):
                    result[i,j,:] = reshaped_coords[i,selected_corners[j],:]
                position = 4
                for j in range(num_node):
                    if j not in corners:
                        result[i,position,:] = reshaped_coords[i,j,:]
                        position += 1
            return result.view(-1,3)

    def expend_coords(self,shrinked_coords):
        num_gen = shrinked_coords.shape[0]
        coords_to_add = torch.zeros((num_gen,4,3)).to('cuda')
        coords_to_add[:,0,:] = shrinked_coords[:,1,:] + shrinked_coords[:,2,:] - shrinked_coords[:,0,:]
        coords_to_add[:,1,:] = shrinked_coords[:,2,:] + shrinked_coords[:,3,:] - shrinked_coords[:,0,:]
        coords_to_add[:,2,:] = shrinked_coords[:,3,:] + shrinked_coords[:,1,:] - shrinked_coords[:,0,:]
        coords_to_add[:,3,:] = coords_to_add[:,0,:] + shrinked_coords[:,3,:] - shrinked_coords[:,0,:]
        return torch.cat([shrinked_coords,coords_to_add],dim=1)


    def forward(self, latents, num_atoms, atom_types, gt_frac_coords, gt_cart_coords, lengths, angles, batch, edge_index=None, to_jimages=None, num_bonds=None, distance_reg=None, latent_prop=None, cond=None):
        torch.set_printoptions(threshold=torch.inf)
        '''print(torch.max(gt_frac_coords[18:27]))
        print(gt_frac_coords[18:27])
        print(gt_cart_coords[18:27])
        input()'''
        #print(gt_cart_coords.shape)
        #print(gt_cart_coords)
        num_graphs = batch[-1].item() + 1
        batch_size = num_graphs
        assert batch.shape[0] // num_graphs == self.num_node, 'args num_node is inconsistent with input data'
        time_steps = torch.randint(0, self.num_time_steps, size=(num_graphs,), device='cuda')
        #new_batch = batch.view(batch_size,-1)[:,:5]
        #new_batch = new_batch.reshape((-1,))
        #print(new_batch.shape)
        #print(batch.shape)

        sigmas_per_graph = self.sigmas.index_select(0, time_steps)
        sigmas_per_node = sigmas_per_graph.index_select(0, batch).view(-1,1)
        # gt_cart_coords = correct_cart_coords(gt_cart_coords, lengths, angles, num_atoms, batch)
        # gt_cart_coords = frac_to_cart_coords(gt_frac_coords, lengths, angles, num_atoms)
        
        #shrinked_coords = self.shrink_coords(gt_cart_coords,batch_size).to('cuda')
        cart_coords_noise = torch.randn_like(gt_cart_coords)
        #print(batch)
        #print(shrinked_coords.shape)
        #print(sigmas_per_node.shape)
        #print(cart_coords_noise.shape)
        cart_coords_perturbed = gt_cart_coords + sigmas_per_node * cart_coords_noise
        #cart_coords_perturbed = correct_cart_coords(cart_coords_perturbed, lengths, angles, num_atoms, batch) #本来有
        #ric
        cart_coords_perturbed = cart_coords_perturbed.reshape((batch_size,-1))
        fc_score = self.fc_score(self.backbone(cart_coords_perturbed, cond))
        score_loss = F.mse_loss(fc_score,-cart_coords_noise.reshape((batch_size,self.num_node,3)))
        #return score_loss, dist_reg_loss, pbs_sym_reg_loss
        return score_loss, 0.0, 0.0



    def get_score_norm(self, sigma):
        sigma_min, sigma_max = self.sigmas[0], self.sigmas[-1]
        sigma_index = (torch.log(sigma) - torch.log(sigma_min)) / (torch.log(sigma_max) - torch.log(sigma_min)) * (len(self.sigmas) - 1)
        sigma_index = torch.round(torch.clip(sigma_index, 0, len(self.sigmas)-1)).long()
        return self.score_norms[sigma_index]


    def predict_edge(self, latents, num_atoms, atom_types, lengths, angles, cart_coords, batch, latent_prop=None,cond=None, generate_mode=False):
        # cut_off_edge_index, distance_vectors, pbc_offsets = get_pbc_cutoff_graphs(cart_coords,
        #                                                                           lengths, angles,
        #                                                                           num_atoms, self.cutoff,
        #                                                                           self.max_num_neighbors)
        if not generate_mode:
            max_num_neighbors=self.num_node
            radius=5
        else:
            max_num_neighbors=self.num_node
            # max_num_neighbors=10
            radius=5
        cut_off_edge_index = radius_graph(cart_coords, radius, batch=batch, loop=False, max_num_neighbors=max_num_neighbors)
        # distance_vectors = cart_coords[cut_off_edge_index[1]] - cart_coords[cut_off_edge_index[0]]
        batch_size = batch.shape[0]//self.num_node
        coords = cart_coords.view((batch_size,-1))
        node_emb = self.backbone(coords,cond)

        node_emb = self.fc_edge_prob(node_emb).view(batch.shape[0], -1)
        edge_prob = self.binlin(node_emb[cut_off_edge_index[0]], node_emb[cut_off_edge_index[1]])
        edge_prob = F.sigmoid(edge_prob)
        return cut_off_edge_index, edge_prob


    def predict_pos_score(self, latents, num_atoms, atom_types, lengths, angles, cart_coords, batch, sigma, threshold=0.6, latent_prop=None, edge_index=None, to_jimages=None, num_bonds=None, cond=None):
        return self.fc_score(self.backbone(cart_coords.view(-1, self.num_node*3), cond=cond))



    @torch.no_grad()
    def generate(self, latents, num_atoms, atom_types, lengths, angles, noise_start, noise_end, num_gen_steps=10,
                 num_langevin_steps=100, coord_temp=0.01, step_rate=1e-4, threshold=0.6, latent_prop=None, edge_index=None,
                 to_jimages=None, num_bonds=None, cond=None):
        # log_sigmas = np.linspace(np.log(noise_start), np.log(noise_end), num_gen_steps)
        # sigmas = np.exp(log_sigmas)
        # sigmas = torch.from_numpy(sigmas).float()
        # sigmas.requires_grad = False
        sigmas = self.sigmas
        sigmas = torch.cat([torch.zeros([1], device=sigmas.device), sigmas])


        num_of_samples_gen = num_atoms.shape[0]
        batch = torch.repeat_interleave(torch.arange(num_of_samples_gen, device=num_atoms.device), self.num_node)
        #frac_coords_init = torch.rand(size=(batch.shape[0], 3), device=lengths.device) - 0.5
        #frac_coords_init = torch.rand(size=(angles.shape[0], 45), device=lengths.device) - 0.5
        # frac_coords_init = 10*(torch.rand((num_of_samples_gen,self.num_node,3),device='cuda') - 0.5)
        frac_coords_init = torch.distributions.normal.Normal(torch.zeros([self.num_node,3]),
                                          10*torch.ones([self.num_node,3])).sample([num_of_samples_gen]).to('cuda')
        #cart_coords_init = frac_to_cart_coords(frac_coords_init, lengths, angles, num_atoms)
        cart_coords_init = frac_coords_init

        cart_coords = cart_coords_init
        
        for t in tqdm(range(num_gen_steps, 0, -1)):
            #current_alpha = step_rate * (sigmas[t] / sigmas[1]) ** 2 #ini
            #current_alpha = step_rate
            current_alpha = 0.00000008*(sigmas[t] / sigmas[1])**2
            for _ in range(num_langevin_steps):
                scores_per_node_pos = self.predict_pos_score(latents, num_atoms, atom_types, lengths, angles, cart_coords, batch, sigmas[t], threshold=threshold,latent_prop=latent_prop, edge_index=edge_index, to_jimages=to_jimages, num_bonds=num_bonds, cond=cond)
                #print('adfjkahsdfkj')
                #print(current_alpha)
                #print(scores_per_node_pos)
                #input()
                #cart_coords += current_alpha * scores_per_node_pos + (2 * current_alpha).sqrt() * (coord_temp * torch.randn_like(cart_coords))
                cart_coords += current_alpha * scores_per_node_pos
                #cart_coords += current_alpha * scores_per_node_pos + (2 * current_alpha).sqrt() * (coord_temp * torch.randn_like(cart_coords))
                #cart_coords = correct_cart_coords(cart_coords, lengths, angles, num_atoms, batch)
                

        #frac_coords = cart_to_frac_coords(cart_coords, lengths, angles, num_atoms)
        frac_coords = cart_coords.reshape((num_of_samples_gen,-1,3))

        if edge_index is None:
            cutoff_ind, edge_prob = self.predict_edge(latents, num_atoms, atom_types, lengths, angles, frac_coords.reshape((-1,3)), batch, latent_prop=latent_prop, cond=cond,generate_mode=True)
            edge_prob = edge_prob.view(-1)
            edge_index = cutoff_ind[:, edge_prob >= threshold]
            edge_index = edge_index[:, edge_index[0] < edge_index[1]]

        # edge_index=cutoff_ind
        #return frac_coords, edge_index
        #print(frac_coords)
        #input()
        #return self.expend_coords(frac_coords).reshape((-1,3)), 0
        return frac_coords.reshape((-1,3)), edge_index

