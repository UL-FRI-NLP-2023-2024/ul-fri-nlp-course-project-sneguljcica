from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='This project seeks to improve document representation in specialized domains by adapting sentence-transformer models, which, while effective, are not inherently tuned to specific fields. The focus will be on investigating two advanced adaptation techniques: TSDAE (Transformer-based Denoising AutoEncoder) and GPL (generative pseudo labeling). These methods aim to refine the representation space, making it more sensitive and accurate within a given domain. The students will evaluate the effect of the adaptation on the classification result.',
    author='Sneguljcica',
    license='',
)
