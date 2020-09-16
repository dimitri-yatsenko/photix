import setuptools

with open("README.md") as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.read().split()

setuptools.setup(
    name="photix", 
    version="0.0.1",
    author="Dimitri Yatsenko",
    author_email="dimitri.yatsenko@gmail.com",
    description="Neurophotonics probe simulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dimitri-yatsenko/photix",
	install_requires=requirements,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",],
    python_requires='>=3.6',
)
