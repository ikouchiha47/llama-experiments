import os

from dask.distributed import Client, LocalCluster, wait
from llama_index.core import VectorStoreIndex, Settings
from .dataloader import DataLoaderConfig

from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class Embedder:
    def __init__(self, df, cfg):
        self.df = df
        self.cfg = cfg

    def index_partition(self, partition, cfg: DataLoaderConfig):
        # print("indexing partition...")
        partition = partition.dropna()
        result = partition.apply(cfg.format_row, axis=1)

        db_cfg = cfg.db_config()

        Settings.embed_model = HuggingFaceEmbedding(
            model_name=cfg.model_path,
            cache_folder="./models/",
            device=cfg.torch_device,
            normalize=False
        )
        print("db name ", db_cfg["table_name"])

        store = PGVectorStore.from_params(
            database=db_cfg["database"],
            host=db_cfg["host"],
            password=db_cfg["password"],
            port="5433",
            user=db_cfg["user"],
            table_name=db_cfg["table_name"],
            embed_dim=db_cfg["embed_dim"],
        )

        context = StorageContext.from_defaults(vector_store=store)
        docs = result.tolist()

        VectorStoreIndex(
            docs,
            storage_context=context,
            show_progress=True,
            # use_async=True
        )

        return result

    def index_db(self, use_dask_client=False):
        if self.df is None:
            raise Exception("UninitializedDataframeException")

        client = None
        df = self.df

        if use_dask_client:
            print("using dask distributed client")
            df = self.df.sample(frac=0.01)

            cluster = LocalCluster(
                memory_limit='2GB'
            )
            client = Client(cluster)
            cluster.scale(os.cpu_count() - 1)

            print(client.dashboard_link)

        print("running embedding of documents")

        partitions = df.map_partitions(
            self.index_partition,
            cfg=self.cfg,
            meta=(None, 'object')
        )

        print(df.npartitions)
        partitions.compute()

        if use_dask_client and client is not None:
            client.close()
