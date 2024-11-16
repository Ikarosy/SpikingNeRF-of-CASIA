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

tail = '_rebutt0_main_fair'

expname = 'dvgo_Lifesyle_trial_on_spiking_nerf_hyperparams_depth{}_width{}_dim{}_fine_iter{}_condensing{}_ray_tempo_unmatch_fine{}_lif_{}'\
              .format(depth, width, dim, fine_iter, condensing_fine, ray_tempo_unmatch_fine, lif_type) + tail

basedir = './logs/nsvf_synthetic'
data = dict(
    datadir='./data/Synthetic_NSVF/Lifestyle',
    dataset_type='nsvf',
    inverse_y=True,
    white_bkgd=True,
    spiking_mode=True,
)

coarse_train = dict(
    N_iters=5000,                 # number of optimization steps
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