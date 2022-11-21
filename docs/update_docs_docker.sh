cd ../
docker run --rm -v $PWD:/workspace python:3.8 /bin/bash -c "cd /workspace && \
pip install -r requirements/requirements_docs.txt && cd docs/ && rm -rf build && make clean && make html && make html"
echo "To start web server just run in docs directory:"
echo "python3 -m http.server 8000 --directory ./build/html/"
