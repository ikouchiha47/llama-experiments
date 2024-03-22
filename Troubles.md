## Trouble

```bash
2024-03-22 22:09:21,459 - distributed.worker.memory - WARNING - Unmanaged memory use is high. This may indicate a memory leak or the memory may not be released to the OS; see https://distributed.dask.org/en/latest/worker-memory.html#memory-not-released-back-to-the-os for more information. -- Unmanaged memory: 1.36 GiB -- Worker memory limit: 1.86 GiB
Generating embeddings:   2%|██▌                                                                                                       | 20/848 [00:01<01:21, 10.15it/s]2024-03-22 22:09:23,890 - distributed.worker.memory - WARNING - Unmanaged memory use is high. This may indicate a memory leak or the memory may not be released to the OS; see https://distributed.dask.org/en/latest/worker-memory.html#memory-not-released-back-to-the-os for more information. -- Unmanaged memory: 1.36 GiB -- Worker memory limit: 1.86 GiB

processing and embedding complete
```

Using dask to encode and store using HuggingFaceEmbeddings and Pg
VectorStoreIndex, results in such problems.

Overall need to use some other techniques to process huge files. Probably using
dask to create smaller files and run indexing on them.

But for practical purposes, no one is allowing 1Gib of CSV docs to be uploaded.

