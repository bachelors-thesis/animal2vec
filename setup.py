from setuptools import setup, find_packages

# Read the long description from README.md for a detailed project description
with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="animal2vec",
    version="0.1.0",
    description="A package for training animal2vec embeddings.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Rene Nowotny",
    author_email="you@example.com",
    python_requires=">=3.9.20",
    packages=find_packages(where="."),
    install_requires=[],  # Add runtime dependencies here, if any
    license="MIT",  # Adjust as necessary; this should match your LICENSE file
    license_files=("LICENSE",),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
