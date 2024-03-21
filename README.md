# Sketch2Shape: Single-View Sketch-Based 3D Reconstruction via Multi-View Differentiable Rendering

![Model Architecture](https://github.com/robinborth/sketch2shape/blob/main/docs/static/images/model_architecture.jpg?raw=true)

A project to utilize differentialbe rendering to optimize the latent code from DeepSDF. The project page can be found [robinborth.github.io/Sketch2Shape/](https://robinborth.github.io/Sketch2Shape/).

## Requirenments

- Ubuntu 20.04 LTS
- Python 3.11
- PyTorch 2.1
- PyTorch Lightning 2.2
- Hydra 1.3

## Installation

In order to install the dependencies, please execute the following.

```bash
conda create -n sketch2shape python=3.11
conda activate sketch2shape
pip install -r requirements.txt
pip install -e .
```

## Hydra Configuraion

The project heavily relies on hydra and pytorch lightning to configure and run the experiments and alignes mostly with the lightning-hydra-template. For further documentation please look at [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template).

## Makefile

The Makefile contains the main configurations and abblations how to execute the project. It also demostrates how to use hydra with the cli and usefull examples. In the following examples if we refere with execute something like:

```bash
make train_deepsdf
```

You can see the exact cli arguments that where passed in the Makefile provided in the root of the project. For example the full cli expression would look like that:

```bash
python scripts/train_deepsdf.py +experiment/train_deepsdf=shapenet_chair_4096
```

## Data

In order to use the training and evaluation scripts, you need to have the data set up. You can download the ShapeNetV2 dataset on Huggingface [here](https://huggingface.co/datasets/ShapeNet/ShapeNetCore) (requires to get approved)). There are two
steps that you need to follow.

### 1) Copy Shapenet

This is the simplest way to setup the data folder for the general preprocessing. You do not need to have a shapenet folder, you can simply use your own skripts, but you need to make sure that the data folder has the following structure, e.g. for `shapnet_chair_4096` it would look like that:

```bash
/data
    /shapenet_chair_4096
        /shapes
            /1d6f4020cab4ec1962d6a66a1a314d66
                mesh.obj
            /1eab4c4c55b8a0f48162e1d15342e13b
                mesh.obj
            ...
        metainfo.csv
```

The metainfo.csv describes the splits and the labels that are used for the DeepSDF module and the embedding table. The splits are provided in `data/shapenet_chair_4096/metainfo.csv`. It should contain the following:

```csv
obj_id,label,split
52310bca00e6a3671201d487ecde379e,0,train
...
19861e56a952fe97b8230112437913fd,4095,train
f2dae367e56200a7d9b53420a5458c53,4096,val
...
9dac39c51680daa2f71e06115e9c3b3e,4223,val
d66fe5dc263064a2bc38fb3cb9934c71,4224,test
...
d2af105ee87bc66dae981a300c94a911,4479,test
```

If you have downloaded the ShapeNetV2 dataset you can simply utilize the `copy_shapnet.py` in order to create that kind of folder structure.

```bash
python scripts/copy_shapenet.py dataset=shapenet_chair_4096 +source=path/to/shapenet/chairs/folder
```

### 2) Copy Hand-Drawn Sketch

In order to evaluate the performance on real-world hand-drawn sketches we provide a test dataset consisting of 256 sketches from the test split. The data can be found in `data/shapenet_chair_4096/handdrawn`. In order to preprocess the data and insert it into the shapes folder please execute the following:

```bash
python scripts/copy_hand_drawn.py dataset=shapenet_chair_4096 +source=data/shapenet_chair_4096/handdrawn
```

### 3) Data Preprocessing DeepSDF

After the data is stored correct you can utilize the data preprocessing script. In order to preprocess the the data you can run:

```bash
python scripts/preprocess_data.py dataset=shapenet_chair_4096
```

This will normalize the meshes into a unit shere and make them watertight. After that we sample points near the surface and in the hole shere, for more details refere to `lib/data/preprocess/PreprocessSDF` and `conf/dataset/preprocess_data.yaml` for more details. The resulting dataset should like that. In order to evaluate the reconstructions we further sample points on the normalized mesh, this is used for CD and EMD, please refere to evaluation for more details.

```bash
/data
    /shapenet_chair_4096
        /shapes
            /1d6f4020cab4ec1962d6a66a1a314d66
                mesh.obj
                normalized_mesh.obj
                sdf_samples.npy
                surface_samples.npy
            ...
        metainfo.csv
```

### 4) Data Preprocessing Encoder

After the deepsdf auto-decoder was trained, you can create the trainings data for the encoder. This creates sketch/latent pairs with the respective camera settings. There are three different images that you can generate `sketch`, `normal` and `grayscale` images. Further you can create the images from the ground truth mesh or render with sphere tracing the latent code from deepsdf. We differentiate between `synthetic`, `rendered` and `traverse`. Where the first is simply rendering from the ground truth mesh. We can also traverse between latent codes and then render the traversed latent code in order to generate sketch/latent pairs, this increases the diversity of unique latent codes.

![Dataset Traversal](https://github.com/robinborth/sketch2shape/blob/main/docs/static/images/dataset_traversal.png?raw=true)

The previous figure shows interpolations between the seed chair (left) and the target chairs (bottom) to create new chairs (top). For more details how to generate novel chairs see `lib/data/preprocess/PreprocessRenderings`.

In order to generate the full dataset you need to specify the deepsdf_ckpt to extract the latent codes and render the images.

```bash
python scripts/preprocess_data.py dataset=shapenet_chair_4096 deepsdf_ckpt_path=path/to/deepsdf.ckpt
```

The full folder structure should look like:

```bash
/data
    /shapenet_chair_4096
        /shapes
            /1d6f4020cab4ec1962d6a66a1a314d66
                /synthetic_sketch
                /synthetic_normal
                /synthetic_grayscale
                /rendered_sketch
                /rendered_normal
                /rendered_grayscale
                /traverse_sketch
                /traverse_normal
                /traverse_grayscale
                /eval_synthetic_drawn
                /eval_hand_drawn
                mesh.obj
                normalized_mesh.obj
                sdf_samples.npy
                surface_samples.npy
            ...
        metainfo.csv
```

In order to select the different image datasets you often need to specify a `mode`, the full mapping can be seen here:

```yaml
{
    0: "synthetic_sketch",
    1: "synthetic_normal",
    2: "synthetic_grayscale",
    3: "rendered_sketch",
    4: "rendered_normal",
    5: "rendered_grayscale",
    6: "traverse_sketch",
    7: "traverse_normal",
    8: "traverse_grayscale",
    9: "eval_synthetic_drawn",
    10: "eval_hand_drawn"
}
```

### Optional: Partial Recalculation

After an intial preprocessing you can also change some settings and only recalculate some parts, e.g. for just recallculating the synthetic sketch data you can do:

```bash
python scripts/preprocess_data.py data=shapenet_chair_4096 data.preprocess_synthetic.skip=False
```

## DeepSDF

This project contains a custom implementation to train and evaluate the DeepSDF module.

### Training DeepSDF

In order to train a DeepSDF module you can execute the following:

```bash
python scripts/train_deepsdf.py 
```

In order to run specific experiments you can utilize the configuration system and write your own experiments. Please check out `./conf/experiment/train_deepsdf/shapenet_chair_4096.yaml` how this could look like. You can then run an experiment like that:

```bash
python scripts/train_deepsdf.py +experiment/train_deepsdf=shapenet_chair_4096
```

### Latent Traversal

In order to visualize the DeepSDF latent space you can traverse between latent codes from the training set or the mean latent code gathered during training. The most basic setting is to interpolate between two chairs.

```bash
python scripts/traverse_latent.py +experiment/traverse_latent=train_train_1
```

The interpolated meshes are stored in the logs directory generated by hydra:

```bash
/logs
    /traverse_latent
        /runs
            /YYYY-MM-DD_H-M-S
                /mesh
                    step=000.obj
                    ...
                    step=019.obj
```

## Encoder

In order to train the view-agnostic encoder, with the latent traverse dataset you can run the following.

```bash
python scripts/train_loss.py +experiment/train_loss=latent_traverse
```

## Latent Code Optimization

In order to support the generation of 3D-Shapes, we want to optimize the latent code from DeepSDF during Inference.

### DeepSDF (Debug)

To evaluate DeepSDF, we utilize an trained DeepSDF and try to optimize the latent code, while freezing the MLP. So this can be done for train, val and test splits. The goal is to look how well the MLP can generlize, while also being flexible and allow to optimize to novel shapes. This is also the upperbound for our evaluation metrics for the differentiable rendering, because we direktly optimize with the signed distance values from the `sdf_samples.npy` file.

```bash
python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=mean_train
```

### Differentiable Rendering (Normal)

In the first setting, we simply optimize the ground truth normal maps, that are used for the SNN to guid the optimization. This should be an upperbound for the follwing SNN optimization, however does not reflact the task of generating 3D-Shapes from 2D-Sketches. This is more like a debbuging for the differentiable rendering:

```bash
python scripts/optimize_normals.py +experiment/optimize_normals=mean_train
```

### Differentiable Rendering (Silhouette)

```bash
python scripts/optimize_sketch.py +experiment/optimize_sketch=silhouette
```

### Differentiable Rendering (Global)

```bash
python scripts/optimize_sketch.py +experiment/optimize_sketch=global
```

## Results

The quantitative evaluation for hand-drawn sketches for more details please refere to the full [report](https://github.com/robinborth/sketch2shape/blob/main/docs/static/report/report_borth_korth.pdf):

| Method | CD | EMD | FID | CLIPScore |
| --- | --- | --- | --- | --- |
| Retrieval | 14.29 | 13.41 | **37.14** | 92.82 |
| Encoder | 8.99 | 11.42| 49.60 | 92.86 |
| Optimization | **8.59** | **11.21**| 49.60| **93.14**|

### Metrics

In order to see the implementation details for evaluation please refere to `scripts/optimize_latent.py` at the end.

- Chamfer Distance (CD)
- Earth Mover's Distance (EMD)
- Frechet Inception Distance (FIC)
- CLIPScore

### Retrieval Baseline

That the baseline uses ground truth meshes, that are retrieved from the training set. In order to train the retrieval baseline execute:

```bash
python scripts/train_loss.py +experiment/train_loss=triplet_traverse
```

To evaluate the retrieval baseline, we first infere all sketch embeddings from the test set for a single views and store it in an index. To compute the metrics we then infere the embedding for a sketch in the test set and calculate the distance between this embedding and all the embeddings in the index. We then retrieve the top-k entries and check how much are from the same shape, e.g. `Recall@k`. You can execture.

```bash
python scripts/eval_loss.py +experiment/eval_loss=triplet_loss
```

### Evaluate

In order to evaluate the **retrieval baseline** execute the following:

```bash
python scripts/optimize_deepsdf.py +experiment/eval_deepsdf=triplet_traverse
```

In order to evaluate the **encoder baseline** execute the following:

```bash
python scripts/optimize_deepsdf.py +experiment/eval_deepsdf=latent_traverse
```

In order to evaluate the **differentiable rendering** execute the following:

```bash
python scripts/optimize_sketch.py +experiment/optimize_sketch=silhouette
```

## Streamlit

In order to run the demo, please make sure that `checkpoints/latent_traverse.ckpt` and `checkpoints/deepsdf.ckpt` are in the checkpoints folder and all of the dependencies are installed correctly. To start the application you can then run:

```bash
python -m streamlit run lib/demo/app.py
```

Note if you run the applciation on a server you need to use a ssh tunnel to access the application on your browser:

```bash
ssh -L 8000:127.0.0.1:8501 borth@tuini15-vc04.vc.in.tum.de
```

## BibTeX

```bash
@article{borth2024sketch2shape,
  author    = {Robin Borth, Daniel Korth},
  title     = {Sketch2Shape: Single-View Sketch-Based 3D Reconstruction via Multi-View Differentiable Rendering},
  year      = {2024},
}
```
