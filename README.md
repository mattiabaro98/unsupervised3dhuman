# Run Docker

```bash
docker build -t torch-docker .
```

```bash
docker run --gpus all -it -v .:/home/unsupervised3dhuman torch-docker
```

```
pip install git+https://github.com/facebookresearch/pytorch3d smplx==0.1.28 trimesh==4.1.0 chumpy==0.70 numpy==1.23.1
```