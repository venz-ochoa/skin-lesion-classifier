pipeline:
	@echo "--- Cleaning and Rebuilding Environment ---"
	rm -rf venv
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	@echo "--- Environment Ready. Running Pipeline ---"
	chmod +x src/data_pipeline.sh
	./src/data_pipeline.sh