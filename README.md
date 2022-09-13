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
