FROM nvcr.io/nvidia/pytorch:23.12-py3

SHELL ["/bin/bash", "-c"]

WORKDIR /home
# RUN mkdir unsupervised3dhuman/
# COPY . /home/unsupervised3dhuman/
# RUN pip install -r ./unsupervised3dhuman/requirements.txt

# ENTRYPOINT pip install git+https://github.com/facebookresearch/pytorch3d@v0.7.5 
CMD ["bash"]

