FROM nvcr.io/nvidia/pytorch:23.12-py3

SHELL ["/bin/bash", "-c"]

WORKDIR /home
# RUN mkdir unsupervised3dhuman/
# COPY . /home/unsupervised3dhuman/
# RUN pip install -r ./unsupervised3dhuman/requirements.txt

ENTRYPOINT pip install git+https://github.com/facebookresearch/pytorch3d smplx==0.1.28 trimesh==4.1.0 chumpy==0.70 numpy==1.23.1
CMD ["bash"]

