# Project for course UNN Deep-Learning
 

## How to run Docker image 
I have already built and pushed an Image to [DockerHub](https://hub.docker.com/u/andreylebedev1) 

## Running from DockerHub: 

Then pull my image

```
docker pull andreylebedev1/detic
```

And run the container

```
docker run andreylebedev1/detic
```

## How to build Docker image

Firstly clone this repository

```
git clone git@github.com:AndreyLebedev1/Detic.git
cd Detic
```

Build an image
```
docker build -t detic .
```

Run it with 
```
docker run detic
