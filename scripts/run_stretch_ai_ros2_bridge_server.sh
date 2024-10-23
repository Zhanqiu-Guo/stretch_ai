#!/bin/bash
# Description: Run the docker container with GPU support

# Make sure it fails if we see any errors
set -e

echo "Starting Stretch AI ROS2 Bridge Server on $HELLO_FLEET_ID"
echo "========================================================="
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
parent_dir="$(dirname "$script_dir")"

echo "Reading version from $parent_dir/src/stretch/version.py"
VERSION=`python3 $parent_dir/src/stretch/version.py`

# sudo chown -R $USER:$USER /home/$USER/stretch_user
# sudo chown -R $USER:$USER /home/$USER/ament_ws/install/stretch_description/share/stretch_description/urdf

echo "Running docker image hellorobotinc/stretch-ai-ros2-bridge:$VERSION"
sudo docker run -it \
    --net=host \
    --privileged \
    -v /dev:/dev \
    -v /home/$USER/stretch_user:/home/hello-robot/stretch_user_copy \
    -v /home/$USER/ament_ws/install/stretch_description/share/stretch_description/urdf:/home/hello-robot/stretch_description/share/stretch_description/urdf \
    -e HELLO_FLEET_ID=$HELLO_FLEET_ID \
    hellorobotinc/stretch-ai-ros2-bridge:$VERSION \
    bash -c "source /home/hello-robot/.bashrc; cp -rf /home/hello-robot/stretch_user_copy/* /home/hello-robot/stretch_user; export HELLO_FLEET_ID=$HELLO_FLEET_ID; ros2 launch stretch_ros2_bridge server.launch.py"