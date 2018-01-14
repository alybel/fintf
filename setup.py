import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="fintf",
    version="0.1.0",
    author="Alexander beck",
    author_email="al.d.beck@gmail.com",
    description=("Tensorflow for financial markets"),
    license="",
    keywords="machine learning",
    url="",
    packages=['fintf'],
    long_description='long description goes here',  # read('README'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
    ], install_requires=[]
)
from fintf import settings

os.makedirs(settings.data_path)