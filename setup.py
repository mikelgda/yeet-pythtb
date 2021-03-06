import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="yeet-pythtb",                     # This is the name of the package
    version="1.0.2",                        # The initial release version
    author="Mikel García Díez",                     # Full name of the author
    description="A JIT-compiled version of PythTB for solving tight-binding models",
    long_description=long_description,      # Long description read from the the readme file
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),    # List of all python modules to be installed
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Operating System :: OS Independent",
    ],                                      # Information to filter the project on PyPi website
    python_requires='>=3.6',                # Minimum version requirement of the package
    py_modules=["yeet_pythtb","fast_scalar",
    "fast_spin","fast_wfarray"],             # Name of the python package
    package_dir={'':'yeet-pythtb/src'},     # Directory of the source code of the package
    install_requires=["numba","matplotlib","numpy","scipy"]                     # Install other dependencies if any
)