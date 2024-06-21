from setuptools import setup, find_packages

with open("README.md", "r") as f:
    description = f.read()

setup(
    name='CurvedSurfaceReconstraction',
    version='0.0.5',
    packages=find_packages(),
    install_requires=[
        'opencv-python == 4.5.5.64',
        'numpy<=1.26.4',
        'matplotlib>=3.8.0',
        'datetime',
        'screeninfo',
    ],
    
    long_description = description,
    long_description_content_type='text/markdown',
    url="",
    author = "Ahmed H. Hanfy, Pawel Flaszyński, Piotr Kaczyński, Piotr Doerffer",
    author_email= "ahmed.hady.hanfy92@gmail.com",
    license="MIT",
    python_requires = ">= 3.11"
)