# tensorflow2-object-detection

## 1. Install anaconda

## 2. Create new Environment with 

```
conda create --name tfod tensorflow-gpu
```
or in 3 steps 
```
conda create --name tfod
activate tfod
conda install tensorflow-gpu
```
## 3. Check Installation
```
conda activate tfod
python
>>> import tensorflow as tf
>>> tf.config.list_physical_devices('GPU')
```
## 4. Install Object Detection API
### Method 1: Install the TensorFlow Model Garden pip package

**tf-models-official** is the stable Model Garden package. Please check out the releases to see what are available modules.

pip3 will install all models and dependencies automatically.
```
pip3 install tf-models-official
```
Note that **tf-models-official** may not include the latest changes in the master branch of this github repo. To include latest changes, you may install **tf-models-nightly**, which is the nightly Model Garden package created daily automatically.
```
pip3 install tf-models-nightly
```
### Method 2: Clone the source
```
git clone https://github.com/tensorflow/models.git
```

conda install jupyter

### Method 3: Docker

**Install Docker**

```
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker
```

**Setup Docker for GPU**
https://www.tensorflow.org/install/docker

Find out Docker version with `docker -v`
- version < 19.03 --> install `nvidia-docker2` package and use `--runtime=nvidia` flag
- version \>= 19.03 --> install `nvidia-container-toolkit` package and use `--gpus all` flag

Setup the package repository and the GPG key:
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
&& curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/experimental/$distribution/libnvidia-container.list | \
sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```
Install Packages
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

**Setup Dockerimage** https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md
You can install the TensorFlow Object Detection API either with Python Package Installer (pip) or Docker. For local runs we recommend using Docker and for Google Cloud runs we recommend using pip.

Clone the TensorFlow Models repository and proceed to one of the installation options.
```
git clone https://github.com/tensorflow/models.git
```
Change Dockerfile like in Repo
```
# From the root of the git repository
docker build -f research/object_detection/dockerfiles/tf2/Dockerfile -t od .
```

**Run Container**
```
docker run --gpus all -it od
```

**Test Installation**
```
# Test the installation.
python object_detection/builders/model_builder_tf2_test.py
```
