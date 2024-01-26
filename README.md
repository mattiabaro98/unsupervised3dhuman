# Run Docker

```bash
docker build -t torch-docker .
```

```bash
docker run --gpus all -it -v .:/home/unsupervised3dhuman torch-docker

```