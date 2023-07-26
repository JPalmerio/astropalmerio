VERSION = "0.1.1"
PKG_NAME = 'astropalmerio'

upload2pip: version pybuild twine

version:
	@echo 'Using version value defined in Makefile'
	@echo 'Updating to version ${VERSION} in following files:'
	@echo 'src/${PKG_NAME}/__init__.py'
	@./update_version.sh ${VERSION}

lint:
	@echo 'Linting'
	python -m black src/
	ruff check src/

test:
	@echo 'Launching tests'
	python -m pytest tests

coverage:
	@echo 'Running coverage'
	coverage run --source src/ -m pytest
	@coverage xml -o reports/coverage.xml
	@coverage report

install:
	@echo 'Installing ${PKG_NAME} in editable mode'
	pip install -e .

pybuild:
	@echo 'Building ${PKG_NAME}...'
	@python -m build

twine:
	@echo 'Uploading ${PKG_NAME} to official pip'
	@python -m twine upload dist/*
	
