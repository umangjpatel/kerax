from setuptools import setup, find_packages

__version__ = None

setup(
    name='kerax',
    version=__version__,
    packages=find_packages(exclude=["examples"]),
    python_requires="==3.7",
    install_requires=[
        "pandas>=1.2.0",
        "tensorflow==2.4.0",
        "tensorflow_datasets==4.2.0",
        "matplotlib>=3.3.3",
        "jax>=0.2.7",
        "jaxlib>=0.1.57",
        "numpy>=1.19.5",
        "dill>=0.3.3",
        "tqdm>=4.55.1",
        "msgpack_python>=0.5.6".
        "msgpack>=1.0.2"
    ],
    url='https://github.com/umangjpatel/kerax',
    license='',
    author='Umang Patel (umangjpatel)',
    author_email='umangpatel1947@gmail.com',
    description='Keras-like APIs powered with JAX library',
    classifiers=[
        "Programming Language :: Python :: 3.7"
    ]
)
