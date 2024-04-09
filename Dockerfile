FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel
ENV CUDA_HOME=/usr/local/cuda-11.1/
# requirements
RUN apt update --allow-unauthenticated --allow-insecure-repositories && \
    DEBIAN_FRONTEND=noninteractive apt install -y python3-opencv
RUN pip install mmcv-full==1.4.4 mmsegmentation==0.22.1  
RUN pip install timm tqdm thop tensorboard ipdb h5py ipython Pillow==9.5.0 
RUN pip install -U numpy 

WORKDIR /completion_former 
COPY . /completion_former
RUN cd src/model/deformconv/ && CUDA_HOME=/usr/local/cuda-11.1/ python setup.py build install
# nvidia apex
RUN apt install -y wget unzip
ARG COMMIT=4ef930c1c884fdca5f472ab2ce7cb9b505d26c1a
RUN wget https://github.com/NVIDIA/apex/archive/${COMMIT}.zip && unzip ${COMMIT}.zip && \
    rm ${COMMIT}.zip && cd apex-${COMMIT} && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ 

