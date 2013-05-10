# Copyright (C) 2013 Wesley Baugh
from distutils import log
from setuptools import setup, find_packages
from setuptools.command.install import install


PROGRAM_NAME = 'infertweet'
VERSION = '0.3'
DESCRIPTION = ('Infer information from Tweets. Useful for human-centered '
               'computing tasks, such as sentiment analysis, location '
               'prediction, authorship profiling and more!')
with open('requirements.txt') as f:
    REQUIREMENTS = f.read()
with open('dependency_links.txt') as f:
    DEPENDENCY_LINKS = [x.strip() for x in f]
with open('README.md') as f:
    LONG_DESCRIPTION = f.read()
NLTK_DEPENDENCIES = []  # None yet.


class InstallWithPostCommand(install):
    def run(self):
        install.run(self)
        log.info('running post install function')
        post_install()


def post_install():
    import nltk
    for resource in NLTK_DEPENDENCIES:
        if not nltk.download(resource):
            log.error('ERROR: Could not download required NLTK resource: '
                      '{0}'.format(resource))


setup(
    name=PROGRAM_NAME,
    version=VERSION,
    packages=find_packages(),

    install_requires=REQUIREMENTS,
    dependency_links=DEPENDENCY_LINKS,
    cmdclass={'install': InstallWithPostCommand},

    author="Wesley Baugh",
    author_email="wesley@bwbaugh.com",
    url="http://www.github.com/bwbaugh/{0}".format(PROGRAM_NAME),
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    license='Creative Commons Attribution-NonCommercial-ShareAlike 3.0 '
            'Unported License',
    classifiers=["Intended Audience :: Developers",
                 "Intended Audience :: Science/Research",
                 "Natural Language :: English",
                 "Programming Language :: Python",
                 "Programming Language :: Python :: 2.7",
                 "Programming Language :: Python :: Implementation :: PyPy",
                 "Topic :: Scientific/Engineering :: Artificial Intelligence",
                 "Topic :: Scientific/Engineering :: Information Analysis",
                 "Topic :: Software Development :: Libraries :: Python Modules",
                 "Topic :: Text Processing :: Linguistic"],
)
