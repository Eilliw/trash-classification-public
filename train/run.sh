cd .
pwd
docker run -d -rm --name tc-traning-container --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -p 8888:8888 tc-training
sleep 2
docker logs tc-traning-container