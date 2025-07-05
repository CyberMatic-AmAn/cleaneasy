from setuptools import setup, find_packages

setup(
    name="cleaneasy",
    version="0.4.2",
    packages=find_packages(),
    install_requires=[
        'pandas>=1.5.0',
        'numpy>=1.23.0',
        'scipy>=1.9.0',
        'scikit-learn>=1.1.0',
        'nltk>=3.7'
    ],
    author="Aman Sonwani",
    author_email="exehyper999@gmail.com",
    description="A comprehensive data cleaning toolkit for various data structures",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/CyberMatic-AmAn/cleaneasy",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)