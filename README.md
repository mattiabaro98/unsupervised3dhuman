# Run Docker

```bash
docker build -t torch-docker .
```

```bash
docker run --gpus all -it --rm -v .:/home torch-docker
```