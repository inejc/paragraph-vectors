from setuptools import setup, find_packages

description = 'A PyTorch implementation of Paragraph Vectors (doc2vec).'

with open('README.md') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requires = f.read().splitlines()

setup(
    name='paragraph-vectors',
    version='0.0.1',
    author='Nejc Ilenic',
    description=description,
    long_description=long_description,
    license='MIT',
    keywords='nlp documents embedding machine-learning',
    install_requires=requires,
    packages=find_packages(),
    test_suite='tests',
    classifiers=[
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.5',
    ],
)
