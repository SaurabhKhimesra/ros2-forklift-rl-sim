# Task shortcuts. Run `make help` for the list.
SHELL := /bin/bash
.DEFAULT_GOAL := help
WS := $(CURDIR)
PY := python3
export PYTHONPATH := $(WS)/src/forklift_gym_env:$(PYTHONPATH)

CONFIG ?= td3_kinematic.yaml
RUN    ?=

.PHONY: help
help:  ## show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS=":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- python only
.PHONY: install
install:  ## install python deps for the fast sim (no ROS needed)
	$(PY) -m pip install -r requirements-dev.txt

.PHONY: test
test:  ## run the test suite (no ROS, no GPU, a few seconds)
	$(PY) -m pytest

.PHONY: coverage
coverage:  ## test suite with a coverage report
	$(PY) -m pytest --cov=forklift_gym_env --cov-report=term-missing

.PHONY: lint
lint:  ## ruff check + format check
	$(PY) -m ruff check src
	$(PY) -m ruff format --check src

.PHONY: format
format:  ## autoformat
	$(PY) -m ruff check --fix src && $(PY) -m ruff format src

.PHONY: train
train:  ## train on the fast sim: make train [CONFIG=td3_kinematic.yaml]
	$(PY) -m forklift_gym_env train -c $(CONFIG)

.PHONY: evaluate
evaluate:  ## evaluate a checkpoint: make evaluate RUN=runs/<dir>
	@test -n "$(RUN)" || { echo "usage: make evaluate RUN=runs/<dir>"; exit 1; }
	$(PY) -m forklift_gym_env eval -c $(RUN)/config.yaml $(RUN)/best.pt \
	  --plot $(RUN)/trajectories.png --gif $(RUN)/rollout.gif

.PHONY: report
report:  ## plot learning curves: make report RUN="runs/a runs/b"
	@test -n "$(RUN)" || { echo 'usage: make report RUN="runs/a runs/b"'; exit 1; }
	$(PY) -m forklift_gym_env report $(RUN) -o learning_curves.png

.PHONY: configs
configs:  ## list the configs this checkout can find
	$(PY) -m forklift_gym_env list-configs

.PHONY: tensorboard
tensorboard:  ## open tensorboard on the runs directory
	tensorboard --logdir runs

# ------------------------------------------------------------------ ROS 2
.PHONY: deps
deps:  ## install ROS dependencies declared in package.xml
	rosdep install --from-paths src --ignore-src -y

.PHONY: build
build:  ## colcon build the ROS 2 workspace
	colcon build --symlink-install

.PHONY: rebuild
rebuild: clean build  ## clean + build from scratch

.PHONY: clean
clean:  ## remove colcon build artefacts
	rm -rf build install log

.PHONY: sim
sim:  ## launch gazebo + forklift + pallet (terminal 1)
	source install/setup.bash && ros2 launch forklift_robot forklift_sim.launch.py

.PHONY: sim-gui
sim-gui:  ## same, with the gzclient GUI
	source install/setup.bash && ros2 launch forklift_robot forklift_sim.launch.py gui:=true

.PHONY: train-gazebo
train-gazebo:  ## train against gazebo (terminal 2, after `make sim`)
	source install/setup.bash && ros2 run forklift_gym_env forklift train -c td3_gazebo.yaml

.PHONY: teleop
teleop:  ## drive the forklift by hand
	source install/setup.bash && ros2 run forklift_robot teleop

.PHONY: rviz
rviz:  ## open rviz with the sensor config
	rviz2 -d src/forklift_robot/rviz/forklift_with_sensors.rviz

.PHONY: kill-gazebo
kill-gazebo:  ## kill leftover gazebo processes
	-pkill -9 -f 'gzserver|gzclient|gazebo' 2>/dev/null; true
