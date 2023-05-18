.PHONY: clean
clean:
	find . -depth -name __pycache__ -exec rm -rf {} \;
	find . -depth -name \*.pyc -exec rm -f {} \;