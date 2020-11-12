from setuptools import setup, find_packages

setup(
    name='digcn',
    version='0.1.0',
    description='Implement of DiGCN (NeurIPS 2020)',
    author='Zekun Tong',
    author_email='zekuntong@u.nus.edu',
    url='https://github.com/flyingtango/DiGCN',
    install_requires=['scikit-learn','pandas','torch-geometric==1.5.0'],
    packages=find_packages())
