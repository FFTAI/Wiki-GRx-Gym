# rl_docker

[中文文档](README_CN.md)

This project is used to configure a Reinforcement Learning Docker environment based on [isaac_gym](https://developer.nvidia.com/isaac-gym).

Using Docker allows for the rapid deployment of isolated, virtual, and identical development environments, eliminating the situation of "it runs on my computer, but not on yours."

## How to Use

### Build the Image

```bash
bash build.sh
```

### Run the Image

```bash
bash run.sh -g <gpus, should be num 1~9 or all> -d <true/false>
# example: bash run.sh -g all -d true
```

These two newly created files will not be tracked by Git. If needed, please modify`.gitignore`.

Use `Ctrl+P+Q` to exit the current terminal and use `exit` to stop the container.

## Check Resource Usage

The image comes with `nvitop` installed. Open a new window, run `bash exec.sh` to enter the container, and use `nvitop` to view the system resource usage. Use `exit` or `Ctrl+P+Q` to exit the current terminal without stopping the container.

## Troubleshooting

### GPU Issues

The default Dockerfile supports NVIDIA RTX 4090, please find the supported versions of pytorch for other GPUs in the link below

[Frameworks Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)

Modify the first line of the `docker/Dockerfile` file:

  ```dockerfile
  nvcr.io/nvidia/pytorch:22.12-py3
  ```

### Permission Issues

If you encounter the following error when running the `run.sh` script:

```
Error response from daemon: failed to create task for container: failed to create shim task: OCI runtime create failed: runc create failed: unable to start container process: error during container init: error running hook #0: error running hook: exit status 1, stdout: , stderr: Auto-detected mode as 'legacy'
nvidia-container-cli: initialization error: load library failed: libnvidia-ml.so.1: cannot open shared object file: no such file or directory: unknown
```

This error is mostly due to not running the container with root privileges. Here are a few solutions:

* Prefix the bash command with `root`.
* Switch to the root user.
* Add the current user to the root group.

If you cannot find the pre-built isaacgym image, you need to rebuild the image with root permissions.

### Runtime Issue

If you encounter the following error when running the `run.sh` script:

```
docker: Error response from daemon: could not select device driver "" with capabilities:[[gpu]].
```

You need to install the `nvidia-container-runtime` and `nvidia-container-toolkit` packages, and modify the Docker daemon startup parameter to change the default runtime to `nvidia-container-runtime`:

```bash
vi /etc/docker/daemon.json
```

Update the content to:

```json
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```
