from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

from nemo.collections.nlp.models.rag.custom_bert_embedder import NeMoBertEmbeddings
from nemo.core.config import hydra_runner
from nemo.utils import logging


@hydra_runner(config_path="conf", config_name="rag_indexing")
def main(cfg) -> None:

    # load data
    logging.info("Loading documents.")
    documents = SimpleDirectoryReader(cfg.indexing.data.data_path).load_data()

    # set text transformation
    logging.info("Setting text transformation.")
    Settings.text_splitter = SentenceSplitter()
    Settings.chunk_size = cfg.indexing.data.chunk_size
    Settings.chunk_overlap = cfg.indexing.data.chunk_overlap

    # load embedder
    logging.info("Loading embedding models.")
    model_path = cfg.indexing.embedder.model_path
    embed_batch_size = cfg.indexing.embedder.embed_batch_size
    if cfg.indexing.embedder.model_type == "bert":
        embed_model = NeMoBertEmbeddings(model_path=model_path, cfg=cfg, embed_batch_size=embed_batch_size)
    else:
        assert cfg.indexing.model_type in ["bert"], "Currently RAG pipeline supports 'bert' for embeddings models."
        embed_model = None
    Settings.embed_model = embed_model

    # index data
    logging.info("Indexing data.")
    index = VectorStoreIndex.from_documents(documents, show_progress=True)

    # save index data to disk
    logging.info("Saving index to disk.")
    index_path = cfg.indexing.index_path
    index.storage_context.persist(persist_dir=index_path)


if __name__ == '__main__':
    main()
