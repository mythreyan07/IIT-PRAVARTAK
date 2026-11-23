from setuptools import setup, find_packages

setup(
    name="eagle-dar",
    version="0.1.0",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.10",
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
    ],
)
