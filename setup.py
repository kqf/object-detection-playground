from setuptools import setup, find_packages

setup(
    name="vin-big-data",
    version="0.0.1",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'train=models.main:main',
            'generate=models.mc:generate',
        ],
    },
)
