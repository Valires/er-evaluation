## Development utilities for er_checks
##
## Usage:
## 		make <target> [<arg>=<value> ...]
##
## Targets:
## 		help:		Show this help message.
##		env: 		Create or update conda environment "pv-evaluation"
## 		black:		Format Python files.
ENV?=er-evaluation

.PHONY: help env black

help: makefile
	@sed -n "s/^##//p" $<

env: environment.yml
	@(echo "Creating ${ENV} environment..."; conda env create -f $<) \
	|| (echo "Updating ${ENV} environment...\n"; conda env update -f $<)

black:
	black . --line-length=80
