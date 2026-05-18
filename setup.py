from setuptools import setup, find_packages

setup(
    name="micrograd",
    version="0.1.0",
    author="Muhammad Ahmad",
    description="A tiny scalar-valued autograd engine and small neural net library.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=["micrograd"],
    python_requires=">=3.7",
    install_requires=[],
)