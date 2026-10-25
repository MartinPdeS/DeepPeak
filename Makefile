.DEFAULT_GOAL := help

PYTHON ?= python3
TAG_VERSION ?= $(or $(VERSION),$(filter v%,$(MAKECMDGOALS)))
RELEASE_KIND := $(filter major minor patch,$(MAKECMDGOALS))

.PHONY: help release-check tag release major minor patch

help:
	@echo "DeepPeak release commands"
	@echo ""
	@echo "  make release-check         Check version metadata consistency"
	@echo "  make tag VERSION=vX.Y.Z    Create a release commit and annotated tag"
	@echo "  make release patch         Create the next patch release and tag"
	@echo "  make release minor         Create the next minor release and tag"
	@echo "  make release major         Create the next major release and tag"

ifneq ($(filter tag,$(MAKECMDGOALS)),)
ifneq ($(strip $(TAG_VERSION)),)
.PHONY: $(TAG_VERSION)
$(TAG_VERSION):
	@:
endif
endif

release-check:
	$(PYTHON) -m martinpdes_maintenance.cli check $(if $(VERSION),--version $(VERSION),)

# Create a release commit and annotated tag. Both forms are supported:
#   make tag v0.2.0
#   make tag VERSION=v0.2.0
tag:
	$(PYTHON) -m martinpdes_maintenance.cli tag "$(TAG_VERSION)"

# Derive and publish the next tag from the highest existing semantic release.
# Examples: make release patch, make release minor, make release major
release:
	@test "$(words $(RELEASE_KIND))" -eq 1 || { echo "usage: make release [patch|minor|major]" >&2; exit 2; }
	@set -eu; release_tag="$$($(PYTHON) -m martinpdes_maintenance.cli next $(RELEASE_KIND))"; \
	$(PYTHON) -m martinpdes_maintenance.cli tag "$$release_tag"; \
	git push origin HEAD "refs/tags/$$release_tag"

major minor patch:
	@:
