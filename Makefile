.PHONY: clean-build clean-pycache quality style clean build

# clean the build
clean-build:
	rm -rf build
	rm -rf dist
	find . -name *.egg-info | xargs rm -rf
	@echo "\e[32mAll the build, dist and egg-info cleaned\e[0m"

# clean the cache
clean-pycache:
	find . -name __pycache__ | xargs rm -rf
	find . -name *.pyc | xargs rm -rf
	find . -name *.pyo | xargs rm -rf
	@echo "\e[32mAll python cache is cleaned.\e[0m"

# check code quality
quality:
	black --check --line-length 120 --target-version py36 src | cat
	isort --check-only --recursive -tc -up --line-width=120 src | cat
	flake8 --exclude=.git,__pycache__,docs/source/conf.py,old,build,dist --statistics src | cat

#Format source code
style:
	black --line-length 120 --target-version py36  src
	isort -rc -tc -up --line-width=120 src

# clean
clean: clean-build clean-pycache quality

build: style
	python setup.py sdist
