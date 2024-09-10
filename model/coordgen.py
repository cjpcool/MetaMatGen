import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
from torch_cluster import radius_graph
from torch_scatter import scatter
from tqdm import tqdm
from traitlets import Int

from .spherenet_light import SphereNetLightDecoder
from .Nequip.nequip_decoder import NequipDecoder
from .modules import build_mlp
import sys
sys.path.append("..")
from utils import get_pbc_cutoff_graphs, frac_to_cart_coords, cart_to_frac_coords, correct_cart_coords, \
    get_pbc_distances, align_gt_cart_coords, calculate_to_jimages_efficient, lattice_params_to_matrix_torch

EPS = 1e-8

class CoordGen(torch.nn.Module):
    def __init__(self, backbone_params, latent_dim, num_fc_hidden_layers, fc_hidden_dim, num_time_steps, noise_start, noise_end, cutoff, max_num_neighbors, loss_type='per_node', score_upper_bound=None, use_gpu=True, score_norm=None,
                 property_loss=False, backbone_name='spherenet'):
        super(CoordGen, self).__init__()
        self.property_loss = property_loss

        if backbone_name == 'spherenet':
            self.backbone = SphereNetLightDecoder(**backbone_params)
        elif backbone_name == 'nequip':
            self.backbone = NequipDecoder(**backbone_params)

        if not property_loss:
            property_dim = 0
        else:
            property_dim = latent_dim
        self.fc_score = build_mlp(latent_dim + backbone_params['hidden_channels']+ property_dim, fc_hidden_dim, num_fc_hidden_layers, 1)

        if backbone_name == 'spherenet':
            self.edge_pred = SphereNetLightDecoder(**backbone_params)
        elif backbone_name == 'nequip':
            self.edge_pred = NequipDecoder(**backbone_params)
        self.fc_edge_lin = build_mlp(latent_dim + backbone_params['hidden_channels'] + property_dim, fc_hidden_dim, 1, latent_dim)
        self.fc_edge_prob = build_mlp(latent_dim, fc_hidden_dim, 1, latent_dim)
        self.binlin = nn.Bilinear(latent_dim, latent_dim, 1)


        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.use_gpu = use_gpu
        self.score_norms = None
        if score_norm is not None:
            self.score_norms = torch.from_numpy(score_norm).float()
        
        self._get_noise_params(num_time_steps, noise_start, noise_end)
        self.num_time_steps = num_time_steps
        self.pool = mp.Pool(16)
        self.score_upper_bound = score_upper_bound
        self.loss_type = loss_type
        if use_gpu:
            self.backbone = self.backbone.to('cuda')
            self.fc_score = self.fc_score.to('cuda')
            self.sigmas = self.sigmas.to('cuda')
            self.edge_pred = self.edge_pred.to('cuda')
            self.fc_edge_prob = self.fc_edge_prob.to('cuda')
            self.fc_edge_lin = self.fc_edge_lin.to('cuda')
            self.binlin = self.binlin.to("cuda")
            if score_norm is not None:
                self.score_norms = self.score_norms.to('cuda')


    def _get_noise_params(self, num_time_steps, noise_start, noise_end):
        log_sigmas = np.linspace(np.log(noise_start), np.log(noise_end), num_time_steps)
        sigmas = np.exp(log_sigmas)
        self.sigmas = torch.from_numpy(sigmas).float()
        self.sigmas.requires_grad = False


    def _center_coords(self, coords, batch):
        coord_center = scatter(coords, batch, reduce='mean', dim=0)
        return coords - coord_center[batch]


    def forward(self, latents, num_atoms, atom_types, gt_frac_coords, lengths, angles, batch, edge_index=None, to_jimages=None, num_bonds=None, distance_reg=None, latent_prop=None):
        num_graphs = batch[-1].item() + 1
        time_steps = torch.randint(0, self.num_time_steps, size=(num_graphs,), device=atom_types.device)
        
        sigmas_per_graph = self.sigmas.index_select(0, time_steps)
        sigmas_per_node = sigmas_per_graph.index_select(0, batch).view(-1,1)
        gt_cart_coords = frac_to_cart_coords(gt_frac_coords, lengths, angles, num_atoms)
        cart_coords_noise = torch.randn_like(gt_cart_coords)
        cart_coords_perturbed = gt_cart_coords + sigmas_per_node * cart_coords_noise
        cart_coords_perturbed = correct_cart_coords(cart_coords_perturbed, lengths, angles, num_atoms, batch)

        if edge_index is None or to_jimages is None or num_bonds is None:
            edge_index, distance_vectors, pbc_offsets = get_pbc_cutoff_graphs(cart_coords_perturbed, lengths, angles, num_atoms, self.cutoff, self.max_num_neighbors)
        else:
            _, distance_vectors, pbc_offsets = get_pbc_distances(cart_coords_perturbed, edge_index, lengths, angles, to_jimages, num_atoms, num_bonds, True)

        edge_features = self.backbone(atom_types, edge_index, distance_vectors)
        num_multi_edge_per_graph = scatter(torch.ones(size=(edge_index.shape[1],), device=edge_index.device).long(), batch[edge_index[0]], dim_size=num_graphs, reduce='sum')
        latents_per_multi_edge = latents.repeat_interleave(num_multi_edge_per_graph, dim=0)
        edge_features = torch.cat((edge_features, latents_per_multi_edge), dim=1)
        if self.property_loss and latent_prop is not None:
            latent_prop_per_multi_edge = latent_prop.repeat_interleave(num_multi_edge_per_graph, dim=0)
            edge_features = torch.cat((edge_features, latent_prop_per_multi_edge), dim=1)

        j, i = edge_index
        no_iden_mask = (i != j)
        j, i, edge_features, distance_vectors = j[no_iden_mask], i[no_iden_mask], edge_features[no_iden_mask], distance_vectors[no_iden_mask]
        scores_per_multi_edge = self.fc_score(edge_features)
        
        # if edge_index is None or to_jimages is None or num_bonds is None:
        #     pbc_offsets = pbc_offsets[no_iden_mask]
        #     aligned_gt_cart_coords = align_gt_cart_coords(gt_cart_coords, cart_coords_perturbed, lengths, angles, num_atoms)
        #     gt_distance_vectors = aligned_gt_cart_coords[i] - aligned_gt_cart_coords[j] - pbc_offsets
        #     gt_dists_per_multi_edge = torch.linalg.norm(gt_distance_vectors, dim=-1, keepdim=True)
        #
        # else:
        #     aligned_gt_cart_coords = align_gt_cart_coords(gt_cart_coords, cart_coords_perturbed, lengths, angles, num_atoms)
        #     _, gt_distance_vectors, _ = get_pbc_distances(aligned_gt_cart_coords, edge_index, lengths, angles, to_jimages, num_atoms, num_bonds, True)
        #     gt_distance_vectors = gt_distance_vectors[no_iden_mask]
        #     gt_dists_per_multi_edge = torch.linalg.norm(gt_distance_vectors, dim=-1, keepdim=True)

        pbc_offsets = pbc_offsets[no_iden_mask]
        aligned_gt_cart_coords = align_gt_cart_coords(gt_cart_coords, cart_coords_perturbed, lengths, angles, num_atoms)
        gt_distance_vectors = aligned_gt_cart_coords[i] - aligned_gt_cart_coords[j] - pbc_offsets
        gt_dists_per_multi_edge = torch.linalg.norm(gt_distance_vectors, dim=-1, keepdim=True)
        perturb_dists_per_multi_edge = torch.linalg.norm(distance_vectors, dim=-1, keepdim=True)
        gt_scores_per_multi_edge = gt_dists_per_multi_edge - perturb_dists_per_multi_edge

        if self.score_norms is not None:
            score_norms_per_graph = self.score_norms.index_select(0, time_steps)
            score_norms_per_node = score_norms_per_graph.index_select(0, batch).view(-1,1)
            score_norms_per_multi_edge = score_norms_per_node.index_select(0, i).view(-1,1)

            if self.score_upper_bound is not None:
                upper_bound_mask = (gt_scores_per_multi_edge <= self.score_upper_bound * score_norms_per_multi_edge).view(-1)
                j, i = j[upper_bound_mask], i[upper_bound_mask]
                scores_per_multi_edge = scores_per_multi_edge[upper_bound_mask]
                gt_scores_per_multi_edge = gt_scores_per_multi_edge[upper_bound_mask]
                score_norms_per_multi_edge = score_norms_per_multi_edge[upper_bound_mask]
                distance_vectors = distance_vectors[upper_bound_mask]
                perturb_dists_per_multi_edge = perturb_dists_per_multi_edge[upper_bound_mask]
        else:
            score_norms_per_node = sigmas_per_node
            score_norms_per_multi_edge = sigmas_per_node.index_select(0, i).view(-1,1)


        if self.loss_type == 'per_edge':
            score_loss = F.mse_loss(scores_per_multi_edge, gt_scores_per_multi_edge / score_norms_per_multi_edge, reduction='none')
            edge_to_graph = batch[i]
            score_loss = scatter(score_loss, edge_to_graph, dim=0, reduce='mean').mean()
        
        elif self.loss_type == 'per_node':
            num_multi_edges = len(i)
            new_edge_start_mask = torch.logical_or(i[:-1] != i[1:], j[:-1] != j[1:])
            new_edge_start_id = torch.nonzero(new_edge_start_mask).view(-1) + 1
            num_multi_edges_per_edge = torch.cat([new_edge_start_id[0:1], new_edge_start_id[1:] - new_edge_start_id[:-1], num_multi_edges - new_edge_start_id[-1:]])
            multi_edge_to_edge_idx = torch.repeat_interleave(torch.arange(len(num_multi_edges_per_edge), device=num_multi_edges_per_edge.device), num_multi_edges_per_edge)

            scores_per_multi_edge = scores_per_multi_edge * distance_vectors / (perturb_dists_per_multi_edge+EPS)
            scores_per_edge = scatter(scores_per_multi_edge, multi_edge_to_edge_idx, dim=0, reduce='mean')
            gt_scores_per_multi_edge = gt_scores_per_multi_edge * distance_vectors / (perturb_dists_per_multi_edge+EPS)
            gt_scores_per_edge = scatter(gt_scores_per_multi_edge, multi_edge_to_edge_idx, dim=0, reduce='mean')
            unique_edge_receiver_index = scatter(i, multi_edge_to_edge_idx, dim=0, reduce='mean').long()
            scores_per_node_pos = scatter(scores_per_edge, unique_edge_receiver_index, dim=0, dim_size=len(batch), reduce='sum')
            gt_scores_per_node_pos = scatter(gt_scores_per_edge, unique_edge_receiver_index, dim=0, dim_size=len(batch), reduce='sum')

            score_loss = F.mse_loss(scores_per_node_pos, gt_scores_per_node_pos / score_norms_per_node, reduction='none')
            score_loss = scatter(score_loss, batch, dim=0, reduce='mean').mean()
        if torch.any(torch.isnan(score_loss)):
            print('NaN in score_loss')
            print('edge feature', torch.any(torch.isnan(edge_features)))

        # compute predicted distances
        dist_reg_loss = 0.
        if distance_reg is not None and distance_reg > 0:
            predicted_dist = perturb_dists_per_multi_edge + torch.linalg.norm(scores_per_multi_edge, dim=-1, keepdim=True) / score_norms_per_multi_edge
            predicted_dist = scatter(predicted_dist, multi_edge_to_edge_idx, dim=0, reduce='mean')
            dist_reg_loss = (distance_reg - torch.clamp(predicted_dist, min=0., max=distance_reg)).sum()
        pbs_sym_reg_loss = 0.
        pbc_sym_reg=True
        if pbc_sym_reg:
            cart_coords_preds = cart_coords_perturbed + scores_per_node_pos / score_norms_per_node
            dist_vec_to_center = self._center_coords(cart_coords_preds, batch)
            x_max = dist_vec_to_center[:, 0].max()
            x_min = dist_vec_to_center[:, 0].min()
            y_max = dist_vec_to_center[:, 1].max()
            y_min = dist_vec_to_center[:, 1].min()
            z_max = dist_vec_to_center[:, 2].max()
            z_min = dist_vec_to_center[:, 2].min()
            pbs_sym_reg_loss = torch.mean(torch.abs(x_max + x_min) + torch.abs(y_max + y_min) + torch.abs(z_max + z_min))

        # score_loss, dist_reg_loss, pbs_sym_reg_loss = torch.nan_to_num(score_loss,0.0), torch.nan_to_num(dist_reg_loss,0.0), torch.nan_to_num(pbs_sym_reg_loss,0.0)

        return score_loss, dist_reg_loss, pbs_sym_reg_loss



    def get_score_norm(self, sigma):
        sigma_min, sigma_max = self.sigmas[0], self.sigmas[-1]
        sigma_index = (torch.log(sigma) - torch.log(sigma_min)) / (torch.log(sigma_max) - torch.log(sigma_min)) * (len(self.sigmas) - 1)
        sigma_index = torch.round(torch.clip(sigma_index, 0, len(self.sigmas)-1)).long()
        return self.score_norms[sigma_index]


    def predict_edge(self, latents, num_atoms, atom_types, lengths, angles, cart_coords, batch, latent_prop=None):
        # cut_off_edge_index, distance_vectors, pbc_offsets = get_pbc_cutoff_graphs(cart_coords,
        #                                                                           lengths, angles,
        #                                                                           num_atoms, self.cutoff,
        #                                                                           self.max_num_neighbors)
        cut_off_edge_index = radius_graph(cart_coords, self.cutoff, batch=batch, loop=False, max_num_neighbors=self.max_num_neighbors)
        distance_vectors = cart_coords[cut_off_edge_index[1]] - cart_coords[cut_off_edge_index[0]]
        edge_features = self.edge_pred(atom_types, cut_off_edge_index, distance_vectors)
        num_graphs = batch[-1].item() + 1
        num_multi_edge_per_graph = scatter(
            torch.ones(size=(cut_off_edge_index.shape[1],), device=cut_off_edge_index.device).long(),
            batch[cut_off_edge_index[0]], dim_size=num_graphs, reduce='sum')
        latents_per_multi_edge = latents.repeat_interleave(num_multi_edge_per_graph, dim=0)
        edge_features = torch.cat((edge_features, latents_per_multi_edge), dim=1)
        if self.property_loss and latent_prop is not None:
            latent_prop_per_multi_edge = latent_prop.repeat_interleave(num_multi_edge_per_graph, dim=0)
            edge_features = torch.cat((edge_features, latent_prop_per_multi_edge), dim=1)

        edge_features = self.fc_edge_lin(edge_features)
        node_emb = scatter(edge_features, cut_off_edge_index[0], dim=0, dim_size=cart_coords.shape[0], reduce='sum')
        node_emb = self.fc_edge_prob(node_emb)

        edge_prob = self.binlin(node_emb[cut_off_edge_index[0]], node_emb[cut_off_edge_index[1]])

        edge_prob = F.sigmoid(edge_prob)
        return cut_off_edge_index, edge_prob


    def predict_pos_score(self, latents, num_atoms, atom_types, lengths, angles, cart_coords, batch, sigma, threshold=0.6, latent_prop=None, edge_index=None, to_jimages=None, num_bonds=None):

        if edge_index is not None and to_jimages is not None and num_bonds is not None:
            _, distance_vectors, _ = get_pbc_distances(cart_coords, edge_index, lengths, angles, to_jimages, num_atoms, num_bonds, True)
        else:
            edge_index, distance_vectors, pbc_offsets = get_pbc_cutoff_graphs(cart_coords, lengths, angles,num_atoms, self.cutoff,self.max_num_neighbors)

        num_graphs = batch[-1].item() + 1

        edge_features = self.backbone(atom_types, edge_index, distance_vectors)
        num_multi_edge_per_graph = scatter(torch.ones(size=(edge_index.shape[1],), device=edge_index.device).long(), batch[edge_index[0]], reduce='sum')
        latents_per_multi_edge = latents.repeat_interleave(num_multi_edge_per_graph, dim=0)
        edge_features = torch.cat((edge_features, latents_per_multi_edge), dim=1)
        if self.property_loss and latent_prop is not None:
            latent_prop_per_multi_edge = latent_prop.repeat_interleave(num_multi_edge_per_graph, dim=0)
            edge_features = torch.cat((edge_features, latent_prop_per_multi_edge), dim=1)

        j, i = edge_index
        no_iden_mask = (i != j)
        j, i, edge_features, distance_vectors = j[no_iden_mask], i[no_iden_mask], edge_features[no_iden_mask], distance_vectors[no_iden_mask]
        dists_per_multi_edge = torch.linalg.norm(distance_vectors, dim=-1, keepdim=True)
        scores_per_multi_edge = self.fc_score(edge_features) * distance_vectors / dists_per_multi_edge

        num_multi_edges = len(i)
        new_edge_start_mask = torch.logical_or(i[:-1] != i[1:], j[:-1] != j[1:])
        new_edge_start_id = torch.nonzero(new_edge_start_mask).view(-1) + 1
        num_multi_edges_per_edge = torch.cat([new_edge_start_id[0:1], new_edge_start_id[1:] - new_edge_start_id[:-1], num_multi_edges - new_edge_start_id[-1:]])
        multi_edge_to_edge_idx = torch.repeat_interleave(torch.arange(len(num_multi_edges_per_edge), device=num_multi_edges_per_edge.device), num_multi_edges_per_edge)
        scores_per_edge = scatter(scores_per_multi_edge, multi_edge_to_edge_idx, dim=0, reduce='mean')
        unique_edge_receiver_index = scatter(i, multi_edge_to_edge_idx, dim=0, reduce='mean').long()
        scores_per_node_pos = scatter(scores_per_edge, unique_edge_receiver_index, dim=0, dim_size=len(batch), reduce='sum')

        if self.score_norms is not None:
            score_norm = self.get_score_norm(sigma)
        else:
            score_norm = sigma

        return scores_per_node_pos / score_norm



    @torch.no_grad()
    def generate(self, latents, num_atoms, atom_types, lengths, angles, noise_start, noise_end, num_gen_steps=10,
                 num_langevin_steps=50, coord_temp=0.01, step_rate=1e-4, threshold=0.6, latent_prop=None, edge_index=None,
                 to_jimages=None, num_bonds=None):
        log_sigmas = np.linspace(np.log(noise_start), np.log(noise_end), num_gen_steps)
        sigmas = np.exp(log_sigmas)
        sigmas = torch.from_numpy(sigmas).float()
        sigmas = torch.cat([torch.zeros([1], device=sigmas.device), sigmas])
        sigmas.requires_grad = False
        # sigmas = self.sigmas
        
        batch = torch.repeat_interleave(torch.arange(len(num_atoms), device=num_atoms.device), num_atoms)
        frac_coords_init = torch.rand(size=(batch.shape[0], 3), device=lengths.device) - 0.5
        cart_coords_init = frac_to_cart_coords(frac_coords_init, lengths, angles, num_atoms)

        cart_coords = cart_coords_init
        for t in tqdm(range(num_gen_steps, 0, -1)):
            current_alpha = step_rate * (sigmas[t] / sigmas[1]) ** 2
            for _ in range(num_langevin_steps):
                scores_per_node_pos = self.predict_pos_score(latents, num_atoms, atom_types, lengths, angles, cart_coords, batch, sigmas[t], threshold=threshold,latent_prop=latent_prop, edge_index=edge_index, to_jimages=to_jimages, num_bonds=num_bonds)
                cart_coords += current_alpha * scores_per_node_pos + (2 * current_alpha).sqrt() * (coord_temp * torch.randn_like(cart_coords))
                cart_coords = correct_cart_coords(cart_coords, lengths, angles, num_atoms, batch)

        frac_coords = cart_to_frac_coords(cart_coords, lengths, angles, num_atoms)

        if edge_index is None:
            cutoff_ind, edge_prob = self.predict_edge(latents, num_atoms, atom_types, lengths, angles, cart_coords, batch, latent_prop=latent_prop)
            edge_prob = edge_prob.view(-1)
            edge_index = cutoff_ind[:, edge_prob > threshold]
            edge_index = edge_index[:, edge_index[0] < edge_index[1]]

        # edge_index=cutoff_ind
        return frac_coords, edge_index

