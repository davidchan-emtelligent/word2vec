from setuptools import setup, find_packages

setup(
    name='word2vec',
    version='0.0.0',
    python_requires='>=2.7',

    packages=find_packages(where='src'),
    package_dir={'': 'src'},

    install_requires=[
        'segtok==1.5.2',
        'gensim==3.5.0'
    ],
)
