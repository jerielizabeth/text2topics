# -*- coding: utf-8 -*-

from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='text2topics',
      version='0.1',
      description='Dissertation related functions and utilities',
      long_description=readme(),
      url='',
      author='Jeri Wieringa',
      author_email='',
      license='MIT',
      packages=['text2topics'],
      install_requires=[
          'beautifulsoup4 == 4.5.3',
          'nltk == 3.2.1',
          'pandas == 0.23.4',
          'gensim == 3.1.0',
          'textblob == 0.15.1',
          'scipy == 0.18.1',
          'seaborn == 0.8',
      ],
      zip_safe=False)