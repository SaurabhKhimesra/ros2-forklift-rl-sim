"""Package definition for the forklift_gym_env ROS 2 (ament_python) package.

Note ``find_packages()`` rather than a hand-written package list: naming only
the top-level package leaves ``.envs``, ``.rl`` and the rest uninstalled, which
still works under ``colcon build --symlink-install`` and nowhere else. The
``config/``, ``worlds/`` and ``models/`` trees are mirrored into ``share/`` so
:mod:`forklift_gym_env.paths` can find them through the ament index.
"""

from pathlib import Path

from setuptools import find_packages, setup

PACKAGE = "forklift_gym_env"
HERE = Path(__file__).parent


def data_tree(source: str) -> list[tuple[str, list[str]]]:
    """Mirror a directory into ``share/<package>/`` preserving its structure."""
    entries: list[tuple[str, list[str]]] = []
    root = HERE / source
    if not root.is_dir():
        return entries
    for directory in sorted({p.parent for p in root.rglob("*") if p.is_file()}):
        install_dir = Path("share") / PACKAGE / directory.relative_to(HERE)
        files = sorted(str(p.relative_to(HERE)) for p in directory.iterdir() if p.is_file())
        entries.append((str(install_dir), files))
    return entries


setup(
    name=PACKAGE,
    version="1.0.0",
    # find_packages() picks up every subpackage; the test package is excluded
    # from the install but still runnable from a source checkout.
    packages=find_packages(exclude=["*.tests", "*.tests.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE}"]),
        (f"share/{PACKAGE}", ["package.xml"]),
        *data_tree("config"),
        *data_tree("worlds"),
        *data_tree("models"),
    ],
    install_requires=["setuptools", "numpy>=1.23", "gymnasium>=0.29", "torch>=2.0", "PyYAML>=6.0"],
    extras_require={
        "viz": ["matplotlib>=3.6", "pandas>=1.5", "imageio>=2.25"],
        "dev": ["pytest>=7.2", "pytest-cov>=4.0", "ruff>=0.4"],
    },
    zip_safe=True,
    maintainer="Saurabh Khimesra",
    maintainer_email="learningkhimesra@gmail.com",
    description=(
        "Deep reinforcement learning forklift simulation: a ROS 2 / Gazebo Gymnasium "
        "environment plus a fast ROS-free surrogate simulator, with TD3/DDPG and HER."
    ),
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # One entry point; the subcommands live behind it in cli.py.
            f"forklift = {PACKAGE}.cli:main",
        ],
    },
)
