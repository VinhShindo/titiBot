import torch
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
from keybert import KeyBERT
from typing import List

EMB_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
model_name_keybert = "vinai/phobert-base"

def load_embedding_model(model_name: str):
    return SentenceTransformer(model_name)

def load_keybert_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    class PhobertEmbeddingModel:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer

        def embed_documents(self, documents):
            encoded_input = self.tokenizer(documents, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            return sentence_embeddings.tolist()

        def _mean_pooling(self, model_output, attention_mask):
            token_embeddings = model_output[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
    return KeyBERT(model=PhobertEmbeddingModel(model, tokenizer))

kw_model = load_keybert_model(model_name_keybert)
print("--> Khởi tạo mô hình KeyBERT thành công.")

def extract_keywords_keybert(user_query: str) -> List[str]:
    keywords = kw_model.extract_keywords(user_query, keyphrase_ngram_range=(1, 2), top_n=5, use_mmr=True, diversity=0.7)
    extracted_keywords = [keyword for keyword, score in keywords]
    print(f"--> Từ khóa được trích xuất: {extracted_keywords}")
    return extracted_keywords