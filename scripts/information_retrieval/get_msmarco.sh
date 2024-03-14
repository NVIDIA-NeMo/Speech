echo "=== Acquiring MSMARCO dataset ==="
echo "---"

mkdir -p msmarco_dataset
cd msmarco_dataset

echo "- Downloading passages"
wget --quiet --continue https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz
tar -xzvf collection.tar.gz
rm collection.tar.gz

echo "- Downloading queries"
wget --quiet --continue https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz
tar -xzvf queries.tar.gz
rm queries.tar.gz
rm queries.eval.tsv

echo "- Downloading relevance labels"
wget --quiet --continue https://msmarco.blob.core.windows.net/msmarcoranking/qrels.train.tsv
wget --quiet --continue https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.tsv

echo "---"