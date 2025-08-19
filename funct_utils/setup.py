from setuptools import setup, find_packages

setup(
    name='funct_utils',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
		'uncertainties',
		#'sklearn',
    ],
    author='Alberto',
    description='Libreria per calcolo FWHM, rilevamento picchi e altro',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Alberto260699/funct_utils',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
