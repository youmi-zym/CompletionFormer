FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel
ENV CUDA_HOME=/usr/local/cuda-11.1/
# requirements
RUN apt update --allow-unauthenticated --allow-insecure-repositories && DEBIAN_FRONTEND=noninteractive apt install -y python3-opencv
RUN pip install mmcv-full==1.4.4 mmsegmentation==0.22.1  
RUN pip install timm tqdm thop tensorboard ipdb h5py ipython Pillow==9.5.0 
RUN pip install -U numpy 

WORKDIR /completion_former 
COPY . /completion_former
RUN cd src/model/deformconv/ && CUDA_HOME=/usr/local/cuda-11.1/ python setup.py build install
