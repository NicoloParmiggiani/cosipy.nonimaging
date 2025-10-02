### COSI Docker for GRB simulations and BCtools IRF

This Docker setup provides a reproducible environment to run COSI GRB simulations (COSIMa) and post-processing tools (BCtools). It includes a non-root user `cosi` and mounts host folders for source code and data.

### Environment variables (recommended)

Define these variables in your shell to customize host and container paths:

```bash
export HOST_COSI_CODE="$HOME/cosi-grb"          # host folder containing this repository
export HOST_COSI_DATA_PATH="$HOME/cosi-grb/data"    # host folder containing data/models/outputs
export CONTAINER_COSI_CODE="/home/cosi/cosi" # container path where code is mounted
export PORT=8888                                      # port for Jupyter (customize)
```

Notes:
- You may change `HOST_COSI_CODE` and `HOST_COSI_DATA_PATH`, but you must install/clone this repository under `HOST_COSI_CODE` because it is mounted into the container at `CONTAINER_COSI_CODE`.
- Scripts in this repo expect geometry/models under the container path `/data/...`; therefore keep your models under `HOST_COSI_DATA_PATH` so they are visible in the container as `/data/...`.

### Build the Docker image

```bash
cd cosidl/docker
docker build -t cosi_grb_bgo:v1.1.0 .
```

### Bootstrap the image for your user

```bash
cd ..
sh bootstrap.sh cosi_grb_bgo:v1.1.0 <username>
```

What bootstrap does:
- Re-tags the base image temporarily and builds a new image `cosi_grb_bgo:v1.1.0_<username>`.
- Aligns the `cosi` user inside the image to your host UID/GID, so files created in mounted volumes are owned by you on the host.

### Run the container

```bash
docker run -d --entrypoint="" \
  -v "$HOST_COSI_CODE":"$CONTAINER_COSI_CODE" \
  -v "$HOST_COSI_DATA_PATH":"/data" \
  --name cosi_grb_bgo_container \
  -p $PORT:$PORT \
  cosi_grb_bgo:v1.1.0_<username> \
  tail -f /dev/null
```

### Enter the container

```bash
docker exec --user cosi -it cosi_grb_bgo_container bash
```

### Start Jupyter Notebook in the container

```bash
export CONTAINER_COSI_CODE="/home/cosi/cosi" # container path where code is mounted
export PORT=8888    
cd "$CONTAINER_COSI_CODE"
nohup jupyter notebook --ip=* --port=$PORT --NotebookApp.token='<password>' --no-browser &
```

### SSH port forwarding from your local machine

```bash
ssh -N -L $PORT:localhost:$PORT <user>@<remote-host>
```

### Tips and troubleshooting

- If you change the mount points, update both the host and container paths consistently. The scripts expect geometry paths like `/data/models/.../*.geo.setup` inside the container.
- Use a unique container name if you run multiple instances.
- If you see file permission issues on the host, ensure you ran the bootstrap step so container user IDs match your host.
- Keep `HOST_COSI_DATA_PATH` on a fast disk if possible; it stores large simulation outputs.
