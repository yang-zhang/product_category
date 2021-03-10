.PHONY: requirements

PROJECT_NAME=product_category

create_environment:
	conda create --yes --name $(PROJECT_NAME) python=3.7 anaconda

requirements:
	pip install -r requirements.txt
	conda install --yes ipykernel
	python -m ipykernel install --user --name $(PROJECT_NAME) --display-name "$(PROJECT_NAME)"
	pre-commit install
