python train.py --dataroot ./datasets/cityscapes --name  cityscapes_cyclegan_sn --model cycle_gan --sn_gan 1 --wgan 0 --with_gp 0
python train.py --dataroot ./datasets/cityscapes --name cityscapes_cyclegan_wgan_gp --model cycle_gan --sn_gan 0 --wgan 1 --with_gp 1
