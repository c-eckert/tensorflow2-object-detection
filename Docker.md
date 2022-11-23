
# Tensorflow with GPU on Docker

**Tested Version:**

**Hostsystem**
| Software    | Version     |
| ----------- | ----------- |
| OS          | Ubuntu 20.04|
| Docker      | 20.10.17    |


**Container**
| Software    | Version     |
| ----------- | ----------- |
| OS          | Ubuntu 18.04|
| python      | 3.6.9       |
| tensorflow  | 2.6.0       |
| CUDA Toolkit| 11.7        |


## Install Docker

```
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker
```

### Setup Docker for GPU
https://www.tensorflow.org/install/docker

Docker is the easiest way to enable TensorFlow GPU support on Linux since only the NVIDIA® GPU driver is required on the host machine (the NVIDIA® CUDA® Toolkit does not need to be installed).

Find out Docker version with `docker -v`
- version < 19.03 --> install `nvidia-docker2` package and use `--runtime=nvidia` flag
- version \>= 19.03 --> install `nvidia-container-toolkit` package and use `--gpus all` flag

**Setup the package repository and the GPG key**:
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
&& curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/experimental/$distribution/libnvidia-container.list | \
sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```
**Install Packages**
```
sudo apt-get update
```
```
sudo apt-get install -y nvidia-container-toolkit
```
Restart the Docker daemon to complete the installation after setting the default runtime:
```
sudo systemctl restart docker
```
At this point, a working setup can be tested by running a base CUDA container:
```
sudo docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```
This should result in a console output shown below:
```
++-----------------------------------------------------------------------------+
| NVIDIA-SMI 515.65.01    Driver Version: 515.65.01    CUDA Version: 11.7     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:09:00.0  On |                  N/A |
| 33%   28C    P8    26W / 215W |    456MiB /  8192MiB |      2%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```

## Install Tensorflow
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md

**Setup Dockerimage**

You can install the TensorFlow Object Detection API either with Python Package Installer (pip) or Docker. For local runs we recommend using Docker and for Google Cloud runs we recommend using pip.

Clone the TensorFlow Models repository and proceed to one of the installation options.
```
git clone https://github.com/tensorflow/models.git
```
Change Dockerfile like in Repo
```
# From the root of the git repository
docker build -f research/object_detection/dockerfiles/tf2/Dockerfile -t od .
# With Data from a level higher
docker build -f models/research/object_detection/dockerfiles/tf2/Dockerfile -t od .

```

**Run Container**
```
docker run --gpus all -it od
```

**File Structure**
```
|\_models (cloned tf-od repo)
|\_scripts
|   \_scripts
|      \_preprocessing
|         \_generate_tfrecord.py
\_workspace
   \_training_demo
      |\_annotations
      |  \_label_map.pbtxt
      |\_images
      |  \_test (all jpg and xml Files)
      |  \_train
      |\_model_main_tf2.py
      \_pre-trained-models
        \_ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8
          \_checkpoint (cp)
          \_pipeline.config
          \_saved_model (.pb)
```

**Test the Installation**
```
python object_detection/builders/model_builder_tf2_test.py
```
## Train Model

**Generate TF-Record Files**
```
python ~/scripts/preprocessing/generate_tfrecord.py -x ~/workspace/training_demo/images/train/ -l ~/workspace/training_demo/annotations/label_map.pbtxt -o ~/workspace/training_demo/annotations/train.record
python ~/scripts/preprocessing/generate_tfrecord.py -x ~/workspace/training_demo/images/test/ -l ~/workspace/training_demo/annotations/label_map.pbtxt -o ~/workspace/training_demo/annotations/test.record
```

**Start Training**
```
python model_main_tf2.py --model_dir=models/my_ssd_mobilenet_v1_fpn --pipeline_config_path=models/my_ssd_mobilenet_v1_fpn/pipeline.config
```
