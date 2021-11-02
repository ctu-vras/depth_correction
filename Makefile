APP_NAME=torch_points3d:depth_correction
CONTAINER_NAME=torch_points3d

build:
	docker build -t $(APP_NAME) -f docker/Dockerfile .

inference: ## Run container for inference
	#xhost + "local:docker@" # for GUI visualization
	docker run \
		--privileged \
	    --runtime=nvidia \
		-itd \
		--name=${CONTAINER_NAME} \
		-v $(shell pwd):/depth_correction \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-e DISPLAY=${DISPLAY} \
	    --network=host \
	    -e JUPYTER_ENABLE_LAB=yes \
	    $(APP_NAME) bash

exec: ## Run a bash in a running container
	docker exec -it ${CONTAINER_NAME} bash

stop: ## Stop and remove a running container
	docker stop ${CONTAINER_NAME}; docker rm ${CONTAINER_NAME}
