""" Simple check list from AllenNLP repo: https://github.com/allenai/allennlp/blob/master/setup.py
To create the package for pypi.
1. Change the version in __init__.py, setup.py as well as docs/source/conf.py.
2. Commit these changes with the message: "Release: VERSION"
3. Add a tag in git to mark the release: "git tag VERSION -m'Adds tag VERSION for pypi' "
   Push the tag to git: git push --tags origin master
4. Build both the sources and the wheel. Do not change anything in setup.py between
   creating the wheel and the source distribution (obviously).
   For the wheel, run: "python setup.py bdist_wheel" in the top level directory.
   (this will build a wheel for the python version you use to build it).
   For the sources, run: "python setup.py sdist"
   You should now have a /dist directory with both .whl and .tar.gz source versions.
5. Check that everything looks correct by uploading the package to the pypi test server:
   twine upload dist/* -r pypitest
   (pypi suggest using twine as other methods upload files via plaintext.)
   You may have to specify the repository url, use the following command then:
   twine upload dist/* -r pypitest --repository-url=https://test.pypi.org/legacy/
   Check that you can install it in a virtualenv by running:
   pip install -i https://testpypi.python.org/pypi transformers
6. Upload the final version to actual pypi:
   twine upload dist/* -r pypi
7. Copy the release notes from RELEASE.md to the tag in github once everything is looking hunky-dory.
8. Update the documentation commit in .circleci/deploy.sh for the accurate documentation to be displayed
9. Update README.md to redirect to correct documentation.
"""

from setuptools import find_packages, setup
from src.classification_report.version import __version__

# normal dependencies ###
#
# these get resolved and installed via either of these two:
#
#   pip install classification-report
#   pip install -e .
#
user_deb = ["torch", "numpy", "seaborn", "matplotlib", "sklearn", "tensorboard"]


# developer dependencies ###
#
# anything else that's not required by a user to run the library, but
# either is an enhancement or a developer-build requirement goes here.
#
# the [dev] feature is documented here:
# https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras-optional-features-with-their-own-dependencies
#
# these, including the normal dependencies, get installed with:
#
#   pip install "classification-report[dev]"
#
# or via an editable install:
#
#   pip install -e ".[dev]"
#
# some of the listed modules appear in test_requirements as well, as explained below.
#
dev_deb = {"dev": ["sphinx",  # documentation
                   "sphinx_rtd_theme",  # read the docs theme
                   "recommonmark",  # markdown support for sphinx
                   "black",  # Code Formatting
                   "flake8",  # linting
                   "isort",  # formatting sort
                   "twine",
                   "ipython",
                   "jupyter"]}

setup(
    name="classification_report",
    version=__version__,
    description="This repo helps to track model Weights, Biases and Gradients during training with loss tracking and gives detailed insight for Classification-Model Evaluation",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="Model-weight-tracking Tensorboard Tensorboard-visualization Model-Evaluation Loss-Tracking Metrics-Visualization Classification-Model",
    license="GPLv3",
    url="https://github.com/aman5319/Classification-Report",


    author="Aman Pandey",
    author_email="amanpandey5319@gmail.com",

    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,

    install_requires=user_deb,
    extras_requires=dev_deb,
    python_requires=">=3.6.0",

    classifiers=["Development Status :: 5 - Production/Stable",
                 "Intended Audience :: Developers",
                 "Intended Audience :: Education",
                 "Intended Audience :: Science/Research",
                 "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
                 'Natural Language :: English',
                 "Operating System :: OS Independent",
                 "Programming Language :: Python :: 3",
                 "Programming Language :: Python :: 3.6",
                 "Programming Language :: Python :: 3.7",
                 "Topic :: Scientific/Engineering :: Artificial Intelligence"],
    project_urls={
        "Documentation": "https://github.com/aman5319/Classification-Report/blob/master/README.md",
        "Source": "https://github.com/aman5319/Classification-Report",
        "Bug Trackers": "https://github.com/aman5319/Classification-Report/issues"
    },

)
