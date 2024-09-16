conf = {}

conf_data = {}
conf_data['prop_name'] = 'heat_ref'
conf_data['graph_method'] = 'crystalnn'

conf_model = {}

conf_enc_spherenet = {}
conf_enc_nequip = {}
conf_enc_spherenet['num_layers'] = 4
conf_enc_spherenet['hidden_channels'] = 128
conf_enc_spherenet['out_channels'] = 256
conf_enc_spherenet['cutoff'] = 1.5
conf_enc_spherenet['int_emb_size'] = 64
conf_enc_spherenet['basis_emb_size_dist'] = 8
conf_enc_spherenet['basis_emb_size_angle'] = 8
conf_enc_spherenet['basis_emb_size_torsion'] = 8
conf_enc_spherenet['out_emb_channels'] = 256
conf_enc_spherenet['num_spherical'] = 7
conf_enc_spherenet['num_radial'] = 6


conf_enc_nequip['n_elems'] = 100
conf_enc_nequip['conv_feature_size'] = 128
conf_enc_nequip['parity'] = True
conf_enc_nequip['lmax'] = 1
conf_enc_nequip['cutoff'] = 1.5
conf_enc_nequip['n_radial_basis'] = 8
conf_enc_nequip['poly_degree'] = 6
conf_enc_nequip['n_conv_layers'] = 3
conf_enc_nequip['radial_network_hidden_dim'] = 64
conf_enc_nequip['radial_network_layers'] = 2
conf_enc_nequip['average_num_neigh'] = 25
conf_enc_nequip['out_hidden_channels'] = 128
conf_enc_nequip['out_channels'] = 256



conf_dec_spherenet = {}
conf_dec_nequip = {}
conf_dec_nequip['n_elems'] = 1
conf_dec_nequip['hidden_channels'] = 128
conf_dec_nequip['parity'] = True
conf_dec_nequip['lmax'] = 1
conf_dec_nequip['n_radial_basis'] = 8
conf_dec_nequip['cutoff'] = 1.5
conf_dec_nequip['poly_degree'] = 6
conf_dec_nequip['n_conv_layers'] = 3
conf_dec_nequip['radial_network_hidden_dim'] = 64
conf_dec_nequip['radial_network_layers'] = 2
conf_dec_nequip['average_num_neigh'] = 10

conf_dec_spherenet['cutoff'] = 1.5
conf_dec_spherenet['num_layers'] = 4
conf_dec_spherenet['hidden_channels'] = 128
conf_dec_spherenet['out_channels'] = 256
conf_dec_spherenet['int_emb_size'] = 64
conf_dec_spherenet['basis_emb_size_dist'] = 8
conf_dec_spherenet['basis_emb_size_angle'] = 8
conf_dec_spherenet['basis_emb_size_torsion'] = 8
conf_dec_spherenet['out_emb_channels'] = 256
conf_dec_spherenet['num_spherical'] = 7
conf_dec_spherenet['num_radial'] = 6

conf_model['backbone_name'] = 'nequip' # 'spherenet'
if conf_model['backbone_name'] == 'nequip':
    conf_model['enc_backbone_params'] = conf_enc_nequip
    conf_model['dec_backbone_params'] = conf_dec_nequip
elif conf_model['backbone_name'] == 'spherenet':
    conf_model['enc_backbone_params'] = conf_enc_spherenet
    conf_model['dec_backbone_params'] = conf_dec_spherenet

conf_model['latent_dim'] = 128
conf_model['num_fc_hidden_layers'] = 1
conf_model['fc_hidden_dim'] = 256
conf_model['max_num_atoms'] = 50
conf_model['min_num_atoms'] = 7
conf_model['use_gpu'] = True
conf_model['lattice_scale'] = False
conf_model['pred_prop'] = True
conf_model['use_multi_latent'] = True
conf_model['logvar_clip'] = 6.0
conf_model['mu_clip'] = 14.0
conf_model['num_time_steps'] = 50
conf_model['noise_start'] = 0.01
conf_model['noise_end'] = 10
conf_model['cutoff'] = 1.5
conf_model['max_num_neighbors'] = 50
conf_model['coord_loss_type'] = 'per_node'
conf_model['use_node_num_loss'] = True
conf_model['property_dim'] = 21


conf_optim = {'lr': 3e-03, 'betas': [0.9, 0.999], 'weight_decay': 0.0}

conf['edge_pred_weight'] = 1.0
conf['kld_weight'] = 0.01
conf['node_num_loss_weight'] = 10
conf['lattice_weight'] = 10.0
conf['coord_weight'] = 10.0
conf['max_grad_value'] = 0.5
# regularizer:
conf['distance_reg'] = 0.05
conf['dist_reg_weight'] = .0
conf['property_weight'] = 1.0
conf['pbc_sym_reg_weight'] = .0


conf['data'] = conf_data
conf['train_size'] = 8000
conf['valid_size'] = 1000
conf['seed'] = 0
conf['model'] = conf_model
conf['optim'] = conf_optim
conf['verbose'] = 10
conf['batch_size'] = 128
conf['start_epoch'] = 0
conf['end_epoch'] = 700
conf['save_interval'] = 20
conf['chunk_size'] = 1000
# temperature: encode(latent_comp, latent_pos, latent_lattice) , decode(latent_coords
conf['train_temp'] = [1., 1., 1.0, 1.0] # comp/node_num, pos, prop, lattice
conf['gen_temp'] = [1., 1., 0.01, 1.0, 0.01]  # comp/node_num, pos, prop, lattice, decode_pos
conf['val_temp'] = [1., 1., 1.0, 1.0]   # comp/node_num, pos, prop, lattice
# conf['loss_thre'] = [3.0, 10.0]

conf['max_atom_num'] = 9
conf['min_atom_num'] = 8