cd ../
docker run --rm -v $PWD:/workspace python:3.10 /bin/bash -c "cd /workspace && \
pip install uv==0.11.14 && uv sync --locked --group docs && uv run make -C docs clean html && uv run make -C docs html"
echo "To start web server just run in docs directory:"
echo "python3 -m http.server 8000 --directory ./build/html/"
