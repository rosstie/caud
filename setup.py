from setuptools import find_packages, setup

setup(
    name="caud",
    version="1.0.0",
    description="Model code for Landscape Complexity and Collective Adaptation",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.26.4",
        "pandas>=1.5.3",
        "matplotlib>=3.4.3",
        "scipy>=1.10.1",
        "networkx>=2.6.3",
        "xarray>=0.20.0",
        "zarr>=2.14.2",
        "pyarrow>=14.0.1",
        "dask>=2021.11.0",
        "psutil>=5.9.8",
        "pybdm>=0.1.0",
    ],
)
