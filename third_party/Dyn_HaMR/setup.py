from setuptools import find_packages, setup

setup(
    name="dyn-hamr",
    packages=find_packages(
        where="dyn-hamr",
    ),
    package_dir={"": "dyn-hamr"},
)