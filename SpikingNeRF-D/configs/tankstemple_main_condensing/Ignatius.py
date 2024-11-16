_base_ = '../default.py'

stepsize = 0.5

depth = 3
width = 128
dim = 12
fine_iter = 40000

condensing_fine = True
ray_tempo_unmatch_fine = False
time_step = 1
lif_type = 'lif'

# tail = '_add_layernorm2d_use_channel_affine'
# tail = '_train_iter_for_finestage'
# tail = '_ann_baseline'
tail = '_rebutt0_main_fair'


expname = 'dvgo_Ignatius_trial_on_spiking_nerf_hyperparams_depth{}_width{}_dim{}_fine_iter{}_condensing{}_ray_tempo_unmatch_fine{}_lif_type{}'\
              .format(depth, width, dim, fine_iter, condensing_fine, ray_tempo_unmatch_fine, lif_type) + tail
basedir = './logs/tanks_and_temple'

data = dict(
    datadir='./data/TanksAndTemple/Ignatius',
    dataset_type='tankstemple',
    inverse_y=True,
    load2gpu_on_the_fly=True,
    white_bkgd=True,
    spiking_mode=True,
)

coarse_train = dict(
    pervoxel_lr_downrate=2,
)


fine_train = dict(
    N_iters=fine_iter,
)

coarse_model_and_render = dict(
    stepsize=0.5,                 # sampling stepsize in volume rendering
)

#
fine_model_and_render = dict(
    rgbnet_depth=depth,               # depth of the colors MLP (there are rgbnet_depth-1 intermediate features)
    rgbnet_width=width,             # width of the colors MLP
    rgbnet_dim=dim,
    stepsize=stepsize,                 # sampling stepsize in volume rendering

    condensing=condensing_fine,
    ray_tempo_unmatch=ray_tempo_unmatch_fine,
    time_step=time_step,
    lif_type=lif_type,

)