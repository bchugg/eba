from setuptools import setup

# load long description from README
with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='eba',
    version='0.01',
    description='Extreme Bounds Analysis',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/bchugg/eba',
    author='Ben Chugg',
    author_email='benchugg@cmu.edu',
    license='MIT',
    packages=['eba'],
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)