IMAGE_NAME=registry.cn-hangzhou.aliyuncs.com/cuizihan/torch-imagent:latest

build:
	sudo docker build -t ${IMAGE_NAME} .

push:
	sudo docker push ${IMAGE_NAME}

all: build push