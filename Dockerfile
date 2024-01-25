FROM nvcr.io/nvidia/pytorch:23.12-py3

WORKDIR /home
COPY . /home

# RUN python -m pip install --upgrade pip
# RUN pip install -r requirements.txt
# RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git"

CMD ["bash"]

