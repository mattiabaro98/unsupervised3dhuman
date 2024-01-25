FROM nvcr.io/nvidia/pytorch:23.12-py3

WORKDIR /home

SHELL ["/bin/bash", "-c"]

COPY ./requirements.txt /home

RUN pip install -r requirements.txt
RUN git clone https://github.com/facebookresearch/pytorch3d \
    && cd pytorch3d \
    && pip install -e .

CMD ["bash"]

