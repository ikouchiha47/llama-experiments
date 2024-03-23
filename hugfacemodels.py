import torch

# from sentence_transformers import SentenceTransformer
#
#
# class HugFaceModels:
#     sentence_downloads = [
#         ("sentence-transformers/all-MiniLM-L6-v2", Path("./models"))]
#
#     @classmethod
#     def download_sentence_transformers(cls):
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         for model_name, path in cls.sentence_downloads:
#             model = SentenceTransformer(
#                 model_name,
#                 device=device,
#             )
#             model.save(str(path / model_name))
#
#
# if __name__ == "__main__":
#     HugFaceModels.download_sentence_transformers()
