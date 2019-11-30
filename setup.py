from setuptools import setup, find_packages

setup(
	name='word2vec',
	version='0.0.1',
	python_requires='>=3.5',

	packages=find_packages(where='src'),
	package_dir={'': 'src'},

	install_requires=[
		'segtok',
		'gensim==3.5.0'
	],
	entry_points={
		'console_scripts': [
			'w2v=word2vec.w2v:main',
		],
	},
)
