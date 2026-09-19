# syntax=docker/dockerfile:1
#
# Layered so that dependency installation is cached and a code change rebuilds
# in seconds. The previous Dockerfile COPY'd the whole repo before installing
# anything (so every edit reinstalled everything), left both dependency installs
# commented out (so the image did not actually work), and used `sh`-only `source`
# lines in RUN steps that had no effect on later layers.
#
#   docker build -t forklift .
#   docker run -it --rm --net=host \
#       -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
#       -v "$PWD":/ws -w /ws forklift

FROM osrf/ros:humble-desktop AS base
SHELL ["/bin/bash", "-lc"]
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# --- system + gazebo ---------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip python3-colcon-common-extensions python3-rosdep \
        ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control \
        ros-humble-ros2-control ros-humble-ros2-controllers \
        ros-humble-xacro ros-humble-robot-state-publisher \
        ros-humble-tf-transformations \
    && rm -rf /var/lib/apt/lists/*

# --- python deps (cached independently of the source) ------------------------
WORKDIR /ws
COPY requirements.txt requirements-viz.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

# --- ROS deps declared in package.xml ----------------------------------------
COPY src/forklift_gym_env/package.xml   src/forklift_gym_env/package.xml
COPY src/forklift_robot/package.xml     src/forklift_robot/package.xml
COPY src/ros_gazebo_plugins/package.xml src/ros_gazebo_plugins/package.xml
RUN rosdep update --rosdistro humble \
    && rosdep install --from-paths src --ignore-src -y --rosdistro humble \
    || echo "rosdep reported missing keys; continuing (apt layer above covers the essentials)"

# --- source last, so edits only invalidate this layer ------------------------
COPY . /ws
RUN source /opt/ros/humble/setup.bash && colcon build --symlink-install

RUN echo 'source /opt/ros/humble/setup.bash'            >> /root/.bashrc \
 && echo 'source /usr/share/gazebo/setup.bash'          >> /root/.bashrc \
 && echo '[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash' >> /root/.bashrc

# Gazebo runs headless by default; `make sim-gui` needs an X socket mounted in.
ENV GAZEBO_MODEL_PATH=/ws/src/forklift_gym_env/models:${GAZEBO_MODEL_PATH} \
    GAZEBO_PLUGIN_PATH=/ws/build/ros_gazebo_plugins:${GAZEBO_PLUGIN_PATH}

CMD ["bash"]

# --- a slim image for CI / pure-python work, no ROS --------------------------
FROM python:3.11-slim AS lite
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/ws/src/forklift_gym_env
WORKDIR /ws
COPY requirements.txt requirements-viz.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY . /ws
CMD ["python", "-m", "pytest"]
