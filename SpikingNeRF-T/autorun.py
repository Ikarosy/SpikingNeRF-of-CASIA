import subprocess
import argparse
import os

def list2str(list):
    j = ''
    for i in list:
        if not i == '':
            j += (i + ' ')
    return j

######################################################################################################################

#nerf
######################################################################################################################

nerf_baseline_cofigs = [
    'configs/nerf_baseline/lego.txt',
    'configs/nerf_baseline/chair.txt',
    'configs/nerf_baseline/drums.txt',
    'configs/nerf_baseline/ficus.txt',
    'configs/nerf_baseline/hotdog.txt',
    'configs/nerf_baseline/materials.txt',
    'configs/nerf_baseline/mic.txt',
    'configs/nerf_baseline/ship.txt',
]

#baseline_2和 baseline之间的区别是，前者的 feature embedding = 0（fea_pe = 0）
nerf_baseline_2_cofigs = [
    'configs/nerf_baseline_2/lego.txt',
    'configs/nerf_baseline_2/chair.txt',
    'configs/nerf_baseline_2/drums.txt',
    'configs/nerf_baseline_2/ficus.txt',
    'configs/nerf_baseline_2/hotdog.txt',
    'configs/nerf_baseline_2/materials.txt',
    'configs/nerf_baseline_2/mic.txt',
    'configs/nerf_baseline_2/ship.txt',
]

nerf_baseline_4_cofigs = [
    'configs/nerf_baseline_4/lego.txt',
    'configs/nerf_baseline_4/chair.txt',
    'configs/nerf_baseline_4/drums.txt',
    'configs/nerf_baseline_4/ficus.txt',
    'configs/nerf_baseline_4/hotdog.txt',
    'configs/nerf_baseline_4/materials.txt',
    'configs/nerf_baseline_4/mic.txt',
    'configs/nerf_baseline_4/ship.txt',
]


nerf_baseline_3_cofigs = [
    'configs/nerf_baseline_3/lego.txt',
    'configs/nerf_baseline_3/chair.txt',
    'configs/nerf_baseline_3/drums.txt',
    'configs/nerf_baseline_3/ficus.txt',
    'configs/nerf_baseline_3/hotdog.txt',
    'configs/nerf_baseline_3/materials.txt',
    'configs/nerf_baseline_3/mic.txt',
    'configs/nerf_baseline_3/ship.txt',
]

nerf_condensing_cofigs = [
    'configs/nerf_condensing/lego.txt',
    'configs/nerf_condensing/chair.txt',
    'configs/nerf_condensing/drums.txt',
    'configs/nerf_condensing/ficus.txt',
    'configs/nerf_condensing/hotdog.txt',
    'configs/nerf_condensing/materials.txt',
    'configs/nerf_condensing/mic.txt',
    'configs/nerf_condensing/ship.txt',
]

nerf_condensing2_cofigs = [
    'configs/nerf_condensing2/lego.txt',
    'configs/nerf_condensing2/chair.txt',
    'configs/nerf_condensing2/drums.txt',
    'configs/nerf_condensing2/ficus.txt',
    'configs/nerf_condensing2/hotdog.txt',
    'configs/nerf_condensing2/materials.txt',
    'configs/nerf_condensing2/mic.txt',
    'configs/nerf_condensing2/ship.txt',
]


nerf_condensing3_cofigs = [
    'configs/nerf_condensing3/lego.txt',
    'configs/nerf_condensing3/chair.txt',
    'configs/nerf_condensing3/drums.txt',
    'configs/nerf_condensing3/ficus.txt',
    'configs/nerf_condensing3/hotdog.txt',
    'configs/nerf_condensing3/materials.txt',
    'configs/nerf_condensing3/mic.txt',
    'configs/nerf_condensing3/ship.txt',
]


nerf_condensing4_cofigs = [
    'configs/nerf_condensing4/lego.txt',
    'configs/nerf_condensing4/chair.txt',
    'configs/nerf_condensing4/drums.txt',
    'configs/nerf_condensing4/ficus.txt',
    'configs/nerf_condensing4/hotdog.txt',
    'configs/nerf_condensing4/materials.txt',
    'configs/nerf_condensing4/mic.txt',
    'configs/nerf_condensing4/ship.txt',
]


######################################################################################################################

#nsvf
######################################################################################################################

nsvf_cofigs_baseline = [
    'configs/nsvf_baseline/Bike.txt',
    'configs/nsvf_baseline/Lifestyle.txt',
    'configs/nsvf_baseline/Palace.txt',
    'configs/nsvf_baseline/Robot.txt',
    'configs/nsvf_baseline/Spaceship.txt',
    'configs/nsvf_baseline/Steamtrain.txt',
    'configs/nsvf_baseline/Toad.txt',
    'configs/nsvf_baseline/Wineholder.txt',
]


nsvf_cofigs_condensing = [
    'configs/nsvf_condensing/Bike.txt',
    'configs/nsvf_condensing/Lifestyle.txt',
    'configs/nsvf_condensing/Palace.txt',
    'configs/nsvf_condensing/Robot.txt',
    'configs/nsvf_condensing/Spaceship.txt',
    'configs/nsvf_condensing/Steamtrain.txt',
    'configs/nsvf_condensing/Toad.txt',
    'configs/nsvf_condensing/Wineholder.txt',
]

nsvf_cofigs_condensing2 = [
    'configs/nsvf_condensing2/Bike.txt',
    'configs/nsvf_condensing2/Lifestyle.txt',
    'configs/nsvf_condensing2/Palace.txt',
    'configs/nsvf_condensing2/Robot.txt',
    'configs/nsvf_condensing2/Spaceship.txt',
    'configs/nsvf_condensing2/Steamtrain.txt',
    'configs/nsvf_condensing2/Toad.txt',
    'configs/nsvf_condensing2/Wineholder.txt',
]


nsvf_cofigs_condensing3 = [
    'configs/nsvf_condensing3/Bike.txt',
    'configs/nsvf_condensing3/Lifestyle.txt',
    'configs/nsvf_condensing3/Palace.txt',
    'configs/nsvf_condensing3/Robot.txt',
    'configs/nsvf_condensing3/Spaceship.txt',
    'configs/nsvf_condensing3/Steamtrain.txt',
    'configs/nsvf_condensing3/Toad.txt',
    'configs/nsvf_condensing3/Wineholder.txt',
]


nsvf_cofigs_condensing4 = [
    'configs/nsvf_condensing4/Bike.txt',
    'configs/nsvf_condensing4/Lifestyle.txt',
    'configs/nsvf_condensing4/Palace.txt',
    'configs/nsvf_condensing4/Robot.txt',
    'configs/nsvf_condensing4/Spaceship.txt',
    'configs/nsvf_condensing4/Steamtrain.txt',
    'configs/nsvf_condensing4/Toad.txt',
    'configs/nsvf_condensing4/Wineholder.txt',
]



nsvf_cofigs_baseline_2 = [
    'configs/nsvf_baseline_2/Bike.txt',
    'configs/nsvf_baseline_2/Lifestyle.txt',
    'configs/nsvf_baseline_2/Palace.txt',
    'configs/nsvf_baseline_2/Robot.txt',
    'configs/nsvf_baseline_2/Spaceship.txt',
    'configs/nsvf_baseline_2/Steamtrain.txt',
    'configs/nsvf_baseline_2/Toad.txt',
    'configs/nsvf_baseline_2/Wineholder.txt',
]


nsvf_cofigs_baseline_3 = [
    'configs/nsvf_baseline_3/Bike.txt',
    'configs/nsvf_baseline_3/Lifestyle.txt',
    'configs/nsvf_baseline_3/Palace.txt',
    'configs/nsvf_baseline_3/Robot.txt',
    'configs/nsvf_baseline_3/Spaceship.txt',
    'configs/nsvf_baseline_3/Steamtrain.txt',
    'configs/nsvf_baseline_3/Toad.txt',
    'configs/nsvf_baseline_3/Wineholder.txt',
]

nsvf_cofigs_baseline_4 = [
    'configs/nsvf_baseline_4/Bike.txt',
    'configs/nsvf_baseline_4/Lifestyle.txt',
    'configs/nsvf_baseline_4/Palace.txt',
    'configs/nsvf_baseline_4/Robot.txt',
    'configs/nsvf_baseline_4/Spaceship.txt',
    'configs/nsvf_baseline_4/Steamtrain.txt',
    'configs/nsvf_baseline_4/Toad.txt',
    'configs/nsvf_baseline_4/Wineholder.txt',
]

######################################################################################################################

#T&T
######################################################################################################################

tankstemple_configs_baseline = [
    'configs/tankstemple/Barn.txt',
    'configs/tankstemple/Caterpillar.txt',
    'configs/tankstemple/Family.txt',
    'configs/tankstemple/Ignatius.txt',
    'configs/tankstemple/Truck.txt',
]


tankstemple_configs_baseline_2 = [
    'configs/tankstemple_baseline2/Barn.txt',
    'configs/tankstemple_baseline2/Caterpillar.txt',
    'configs/tankstemple_baseline2/Family.txt',
    'configs/tankstemple_baseline2/Ignatius.txt',
    'configs/tankstemple_baseline2/Truck.txt',
]


tankstemple_configs_baseline_3 = [
    'configs/tankstemple_baseline3/Barn.txt',
    'configs/tankstemple_baseline3/Caterpillar.txt',
    'configs/tankstemple_baseline3/Family.txt',
    'configs/tankstemple_baseline3/Ignatius.txt',
    'configs/tankstemple_baseline3/Truck.txt',
]


tankstemple_configs_baseline_4 = [
    'configs/tankstemple_baseline4/Barn.txt',
    'configs/tankstemple_baseline4/Caterpillar.txt',
    'configs/tankstemple_baseline4/Family.txt',
    'configs/tankstemple_baseline4/Ignatius.txt',
    'configs/tankstemple_baseline4/Truck.txt',
]

tankstemple_configs_condensing = [
    'configs/tankstemple_condensing/Barn.txt',
    'configs/tankstemple_condensing/Caterpillar.txt',
    'configs/tankstemple_condensing/Family.txt',
    'configs/tankstemple_condensing/Ignatius.txt',
    'configs/tankstemple_condensing/Truck.txt',
]


tankstemple_configs_condensing2 = [
    'configs/tankstemple_condensing2/Barn.txt',
    'configs/tankstemple_condensing2/Caterpillar.txt',
    'configs/tankstemple_condensing2/Family.txt',
    'configs/tankstemple_condensing2/Ignatius.txt',
    'configs/tankstemple_condensing2/Truck.txt',
]

tankstemple_configs_condensing3 = [
    'configs/tankstemple_condensing3/Barn.txt',
    'configs/tankstemple_condensing3/Caterpillar.txt',
    'configs/tankstemple_condensing3/Family.txt',
    'configs/tankstemple_condensing3/Ignatius.txt',
    'configs/tankstemple_condensing3/Truck.txt',
]


tankstemple_configs_condensing4 = [
    'configs/tankstemple_condensing4/Barn.txt',
    'configs/tankstemple_condensing4/Caterpillar.txt',
    'configs/tankstemple_condensing4/Family.txt',
    'configs/tankstemple_condensing4/Ignatius.txt',
    'configs/tankstemple_condensing4/Truck.txt',
]
##############################on forward-facing ##############################
##############################on forward-facing ##############################
##############################on forward-facing ##############################
##############################on forward-facing ##############################
##############################on forward-facing ##############################

llff_configs_baseline = [
    'configs/llff_baseline/fern.txt',
    'configs/llff_baseline/flower.txt',
    'configs/llff_baseline/fortress.txt',
    'configs/llff_baseline/horns.txt',

    'configs/llff_baseline/leaves.txt',
    'configs/llff_baseline/orchids.txt',
    'configs/llff_baseline/room.txt',
    'configs/llff_baseline/trex.txt',
]

llff_configs_condensing = [
    'configs/llff_condensing/fern.txt',
    'configs/llff_condensing/flower.txt',
    'configs/llff_condensing/fortress.txt',
    'configs/llff_condensing/horns.txt',

    'configs/llff_condensing/leaves.txt',
    'configs/llff_condensing/orchids.txt',
    'configs/llff_condensing/room.txt',
    'configs/llff_condensing/trex.txt',
]

# nohup python -u autorun.py > /dev/null 2>&1 &
if __name__ == "__main__":
    # main_shot_name = 'tensorf_On_nerf_baseline'
    # main_shot_name = 'tensorf_On_nsvf_baseline'
    # main_shot_name = 'tensorf_On_tankstemple_baseline'
    # main_shot_name = 'vogo_On_LLFF_baseline'

    #feat_pe=0 baselines
    # main_shot_name = 'tensorf_On_nerf_baseline_2'
    # main_shot_name = 'tensorf_On_nsvf_baseline_2'
    # main_shot_name = 'tensorf_On_tankstemple_baseline_2'


    #feat_pe=0 view_pe=1 baselines
    # main_shot_name = 'tensorf_On_nerf_baseline_3'
    # main_shot_name = 'tensorf_On_nsvf_baseline_3'
    # main_shot_name = 'tensorf_On_tankstemple_baseline_3'

    #feat_pe=0 view_pe=0 baselines
    # main_shot_name = 'tensorf_On_nerf_baseline_4'
    # main_shot_name = 'tensorf_On_nsvf_baseline_4'
    # main_shot_name = 'tensorf_On_tankstemple_baseline_4'


    #spiking
    # main_shot_name = 'tensorf_On_nerf_condensing'
    # main_shot_name = 'tensorf_On_nsvf_condensing'
    # main_shot_name = 'tensorf_On_tankstemple_condensing'


    #spiking feat_pe=0 baselines
    # main_shot_name = 'tensorf_On_nerf_condensing2'
    # main_shot_name = 'tensorf_On_nsvf_condensing2'
    # main_shot_name = 'tensorf_On_tankstemple_condensing2'


    #spiking feat_pe=0 view_pe=1 baselines
    # main_shot_name = 'tensorf_On_nerf_condensing3'
    # main_shot_name = 'tensorf_On_nsvf_condensing3'
    # main_shot_name = 'tensorf_On_tankstemple_condensing3'


    if not os.path.isdir(os.path.join('./RunningLogs', main_shot_name)):
        os.makedirs(os.path.join('./RunningLogs', main_shot_name))

    gpu = '7'
    for conf in tankstemple_configs_condensing3[3:]:
        config = conf
        runningtag = config.replace('/', '_').replace('.txt', '').replace('configs_', '')
        subprocess.check_call(list2str(["CUDA_VISIBLE_DEVICES=" + gpu, "python", "-u", "train.py",

                                        "--config", config,

                                        # "--render_test",
                                        # "--eval_ssim",
                                        # "--eval_lpips_vgg",
                                        # "--eval_lpips_alex",
                                        # '--dump_images',

                                        # '--op_count',

                                        # "--use_ann_trained_fine",

                                        # '--render_only',


                                        ">",
                                        './RunningLogs/' + main_shot_name + '/' + runningtag + '.log',
                                        # 'test.log',
                                        # '/dev/null',
                                        "2>&1"
                                        ]), shell=True)

# python run.py - -config configs/nerf_main_squeeze_poisson_timestep1/lego.py - -render_test --op_count --render_only

    # #for render and eval
    # main_shot_name = 'lif_depth3_width128_dim12_squeeze_On_nerf_main'
    # if not os.path.isdir(os.path.join('./RunningLogs', main_shot_name, 'test')):
    #     os.makedirs(os.path.join('./RunningLogs', main_shot_name, 'test'))
    #
    # for conf in nerf_cofigs[7:]:
    #     config = conf
    #     runningtag = config.replace('/', '_').replace('.py', '').replace('configs_', '')
    #     subprocess.check_call(list2str(["CUDA_VISIBLE_DEVICES=" + gpu, "python", "-u", "run.py",
    #
    #                                     "--config", config,
    #
    #                                     "--render_only",
    #                                     "--render_test",
    #
    #                                     "--eval_ssim",
    #                                     "--eval_lpips_vgg",
    #                                     "--eval_lpips_alex",
    #                                     '--dump_images',
    #
    #                                     '--op_count',
    #
    #                                     ">",
    #                                     './RunningLogs/' + main_shot_name + '/test/' + runningtag + '.log',
    #                                     # 'test.log',
    #                                     # '/dev/null',
    #                                     "2>&1"
    #                                     ]), shell=True)