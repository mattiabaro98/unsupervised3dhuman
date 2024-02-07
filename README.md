# 3d body scan and measure

## GPU drivers
In a GPU virtual machine install GPU drivers with Google script you can find [here](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu#installation_scripts).

## Docker
Install docker ([guide](https://gcore.com/learning/how-to-install-docker-engine-debian/)), use [this](https://docs.docker.com/engine/install/linux-postinstall/) to configure docker correctly.

Install `nvidia-container-toolkit` from [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Build Docker image:
```bash
docker build -t torch-docker .
```
Run Docker image
```bash
docker run --gpus all -it -v ./GITHUB_REPO:/home/GITHUB_REPO torch-docker
```
Once in the container install required packages:
```bash
pip install git+https://github.com/facebookresearch/pytorch3d smplx==0.1.28 trimesh==4.1.0 chumpy==0.70 numpy==1.23.1
```
`pytorch3d` should be in version `v0.7.5`.

## Fit PLY to SMPL
To fit a PLY file use `SMPLfitter_test.py` script modifing file path.

## Measure SMPL
To get body measure from the fitted SMPL model run `measure_smpl.py` script modifing file path. 