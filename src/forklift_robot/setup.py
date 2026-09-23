"""Package definition for the forklift_robot description package.

``data_tree`` mirrors ``launch/``, ``urdf/``, ``rviz/`` and ``config/`` into
``share/forklift_robot`` keeping their structure. Flattening them into one
share directory happens to make ``<xacro:include filename="lidar.xacro"/>``
resolve, by accident of the files landing side by side, and breaks the moment
two packages disagree about a filename.
"""

from pathlib import Path

from setuptools import find_packages, setup

PACKAGE = "forklift_robot"
HERE = Path(__file__).parent


def data_tree(source: str) -> list[tuple[str, list[str]]]:
    root = HERE / source
    if not root.is_dir():
        return []
    return [
        (
            str(Path("share") / PACKAGE / d.relative_to(HERE)),
            sorted(str(p.relative_to(HERE)) for p in d.iterdir() if p.is_file()),
        )
        for d in sorted({p.parent for p in root.rglob("*") if p.is_file()})
    ]


setup(
    name=PACKAGE,
    version="1.0.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE}"]),
        (f"share/{PACKAGE}", ["package.xml"]),
        *data_tree("launch"),
        *data_tree("urdf"),
        *data_tree("rviz"),
        *data_tree("config"),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Saurabh Khimesra",
    maintainer_email="learningkhimesra@gmail.com",
    description="URDF description, ros2_control configuration and launch files for the forklift.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            f"teleop = {PACKAGE}.teleop:main",
        ],
    },
)
