VERSION = "0.0.1"

version:
	@echo 'Using version value defined in Makefile'
	@echo 'Updating to version ${VERSION} in following files:'
	@echo 'src/astropalmerio/__init__.py'
	@./update_version.sh ${VERSION}

lint:
	@echo 'Linting'
	python -m black src/
	python -m pylint src/

test:
	@echo 'Launching tests'
	python -m pytest tests

coverage:
	@echo 'Running coverage'
	coverage run --source src/ -m pytest
	@coverage xml -o reports/coverage.xml
	@coverage report

install:
	@echo 'Installing astropalmerio in editable mode'
	pip install -e .
