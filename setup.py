from setuptools import setup, find_packages

setup(
    name="jupyter_unet",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "tensorflow",
        "keras",
        "scikit-learn",
        "jupyter",
    ],
)