# Sketch2Shape

A project to utilize differentialbe rendering to optimize the latent code from DeepSDF. The loss signal is computed with a Siamese Neuronal Network trained on the ShapeNetV2 dataset. You can download the ShapeNetV2 dataset on Huggingface [here](https://huggingface.co/datasets/ShapeNet/ShapeNetCore) (requires to get approved).

## Requirenments

- Ubuntu 20.04 LTS
- Python 3.11
- Pytorch 2.1

## Installation

In order install the dependencies please execute the following.

```bash
conda create -n sketch2shape python=3.11
conda activate sketch2shape
pip install -r requirements.txt
pip install -e .
```

## Data Preprocessing

In order to use the training and evaluation skripts you need to have the data setup. There are two
steps that you need to follow.

### 1) Copy Shapenet

This is the simplest way to setup the data folder for the general preprocessing. You do not need to have
a shapenet folder, you can simply use your own skripts, but you need to make sure that the data folder has the following structure, e.g. for `shapnet_chair_16` it would look like that:

```
/data
    /shapenet_chair_16
        /shapes
            /1d6f4020cab4ec1962d6a66a1a314d66
                mesh.obj
            /1eab4c4c55b8a0f48162e1d15342e13b
                mesh.obj
            ...
        metainfo.csv
```

The metainfo.csv describes the splits and the labels that are used for the DeepSDF module and the embedding table, it should contain the following:

```
obj_id,label,split
52310bca00e6a3671201d487ecde379e,0,train
...
19861e56a952fe97b8230112437913fd,15,train
f2dae367e56200a7d9b53420a5458c53,16,val
...
9dac39c51680daa2f71e06115e9c3b3e,19,val
d66fe5dc263064a2bc38fb3cb9934c71,20,test
...
d2af105ee87bc66dae981a300c94a911,23,test
```

If you have downloaded the ShapeNetV2 dataset you can simply utilize the `copy_shapnet.py` in order to create that kind of folder structure.

```bash
python scripts/copy_shapenet.py data=shapenet_chair_16 +source=/path/to/ShapeNetCore/03001627 
```

### 2) Data Preprocessing

After the data is stored correct you can utilize the data preprocessing skript. Note that the project highly build ontop of `hydra` configurations. In order to preprocess the the data you can run:

```bash
python scripts/preprocess_data.py data=shapenet_chair_16
```

This will create the full dataset for the SNN and the DeepSDF module. It will look like:

```
/data
    /shapenet_chair_16
        /shapes
            /1d6f4020cab4ec1962d6a66a1a314d66
                    /normals
                        00000.png
                        00001.png
                        ...
                    /sketches
                        00000.png
                        00001.png
                        ...
                mesh.obj
                normalized_mesh.obj
                sdf_samples.npy
                surface_samples.npy
            ...
        metainfo.csv
```

After an intial preprocessing you can also change some settings and only recalculate some parts, e.g. for just recallculating the siamese data you can do:

```bash
python scripts/preprocess_data.py data=shapenet_chair_16 data.preprocess_siamese.skip=False
```
## DeepSDF

This project contains a custom implementation to train and evaluate the DeepSDF module.

### Training DeepSDF

In order to train a DeepSDF module you can execute the following: 

```bash
python scripts/train_deepsdf.py +data=shapenet_chair_16
```

In order to run specific experiments you can utilize the configuration system and write your own experiments. Please check out `./conf/experiment/train_deepsdf/shapenet_chair_16.yaml` how this could look like. You can then run an experiment like that:

```bash
python scripts/train_deepsdf.py +experiment/train_deepsdf=shapenet_chair_16
```

### Evaluating DeepSDF

To evaluate DeepSDF, we utilize an trained DeepSDF and try to optimize the latent code, while freezing the MLP. So this can be done for train, val and test splits. The goal is to look how well the MLP can generlize, while also being flexible and allow to optimize to novel shapes. This is also the upperbound for our evaluation metrics for the differentiable rendering, because we direktly optimize with the signed distance values from the `sdf_samples.npy` file.

```bash
python scripts/optimize_deepsdf.py +data=shapenet_chair_16 ckpt_path=/path/to/deepsdf.ckpt
```

In the default settings multiple wandb runs are spawned. In order to get the mean metric you can look at the logs generated during the run.

```
/logs
    /optimize_deepsdf
        /runs
            /YYYY-MM-DD_H-M-S
                metrics.csv
```

The metrics file contains all of the individual runs, aswell as the mean metric for all the runs on the last row.

```
val/chamfer,obj_id
0.00032423,1eab4c4c55b8a0f48162e1d15342e13b
...
0.00013423,52310bca00e6a3671201d487ecde379e
0.00023020,mean_metric
```

### Latent Traversal

In order to visualize the DeepSDF latent space you can traverse between latent codes from the training set or the mean latent code gathered during training. The most basic setting is to interpolate from the mean to the target latent code you can do. 

```bash
python scripts/traverse_latent.py data=shapenet_chair_16 ckpt_path=/path/to/deepsdf.ckpt model.prior_idx_end=0
```

Another interpolation is between two latent codes you can do. 

```bash
python scripts/traverse_latent.py data=... ckpt_path=...  model.prior_idx_start=0 model.prior_idx_end=1
```

The interpolated meshes are stored in the logs directory generated by hydra:

```
/logs
    /traverse_latent
        /runs
            /YYYY-MM-DD_H-M-S
                /mesh
                    step=000.obj
                    ...
                    step=019.obj
```

## Siamese

In order to utilize the differentiable rendering, we propose to use a SNN to learn a distance between sketches and rendered normal maps. Assume you have such a network to get a distance, you can define a simple 2D loss between the rendered normal maps from DeepSDF and a 2D sketch. This loss can then be used to optimize the latent codes for the DeepSDF model.

### Training Siamese

Note that the current version uses a simplified assumption, that free-hand drawn sketches are similar to edge maps. In order to train the network you need to preprocess the data, as described above, then you can run. 

```bash
python scripts/train_siamese.py +data=shapenet_chair_16
```

### Evaluating Siamese 

To evaluate the SNN, we first infere all normal map embeddings from the train and validation set for all views and store it in an index. To compute the metrics we then infere the embedding for a sketch in the validation set and calculate the distance between this embedding and all the embeddings in the index. We then retrieve the top-k entries and check how much are from the same shape, e.g. `Recall@k`. You can execture.

```bash
python scripts/eval_siamese.py +data=shapenet_chair_16
```

## Differentiable Rendering 

In order to support the generation of 3D-Shapes, we want to optimize the latent code from DeepSDF during Inference. 

### Normal Map
In the first setting, we simply optimize the ground truth normal maps, that are used for the SNN to guid the optimization. This should be an upperbound for the follwing SNN optimization, however does not reflact the task of generating 3D-Shapes from 2D-Sketches. This is more like a debbuging for the differentiable rendering:

```bash
python scripts/optimize_normals.py +data=shapenet_chair_16 ckpt_path=...
```

### Sketch with SNN

In ordre to generate from 2D-Sketches a 3D-Shape we utilize the pretrained SNN. You can optimize the hole validation set via:

```bash
python scripts/optimize_sketch.py +data=shapenet_chair_16 split=val ckpt_path=... 
```