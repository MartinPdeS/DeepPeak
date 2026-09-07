PYTHON ?= python3

.PHONY: release tag patch minor major

release:
	@set -eu; release_tag="$$( $(PYTHON) tools/next_release_version.py $(BUMP) )"; \
	$(PYTHON) tools/release_tag.py "$$release_tag"; \
	git push origin HEAD "refs/tags/$$release_tag"

tag:
	@set -eu; release_tag="$$( $(PYTHON) tools/next_release_version.py $(BUMP) )"; \
	$(PYTHON) tools/release_tag.py "$$release_tag"

patch minor major:
	@$(MAKE) release BUMP=$@
