FROM ubuntu:22.10
FROM python:3.9.13

ADD . .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --upgrade pip
RUN pip install torch

RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'
RUN pip install opencv-python
RUN pip install mss
RUN pip install timm
RUN pip install dataclasses
RUN pip install ftfy
RUN pip install regex
RUN pip install fasttext
RUN pip install scikit-learn
RUN pip install lvis
RUN pip install nltk
RUN pip install git+https://github.com/openai/CLIP.git

CMD ["python3", "my_test.py"]