# cloud prediction

Pytorch implementation of video prediction based on MIM

This projects aims to predict future frames for genneral video.  

# Prerequisite

```
python==3.6(maybe above)
torch==1.6.0 with cuda 10.2(maybe above)
torchvision==0.7.0(maybe above)
tensorboardX==2.2
scikit-image==0.17.2 
opencv-python==4.4.0.46 
```


# Training and Testing

Usage for the code:

```
python main.py --root /path/to/cloud_dataset -K 10 -T 10
```

Models will be saved in `--out_dir`, videos and images are saved in `--img_dir`.

If you want to see the loss curve, simple type

```
tensorboard --logdir=/path/to/checkoints
```

