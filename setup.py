import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="yeet-pythtb",                     # This is the name of the package
    version="1.0.0",                        # The initial release version
    author="Mikel García Díez",                     # Full name of the author
    description="A JIT-compiled version of PythTB for solving tight-binding models",
    long_description=long_description,      # Long description read from the the readme file
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),    # List of all python modules to be installed
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],                                      # Information to filter the project on PyPi website
    python_requires='>=3.6',                # Minimum version requirement of the package
    py_modules=["quicksample"],             # Name of the python package
    package_dir={'':'quicksample/src'},     # Directory of the source code of the package
    install_requires=[]                     # Install other dependencies if any
)