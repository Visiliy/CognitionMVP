# cosine_baseline.py
import os
import glob
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from docx import Document
import PyPDF2


class CosineBaselineSearch:
    def __init__(self, documents_folder_path, device="cpu"):
        self.device = device
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2').to(device)
        self.documents = []
        self.doc_embeddings = None
        self._load_documents(documents_folder_path)
        self._encode_documents()

    def _load_documents(self, folder_path):
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder {folder_path} not found")

        extensions = ['*.txt', '*.pdf', '*.docx']
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(folder_path, ext)))
        files = sorted(files)

        texts = []
        for file in files:
            if file.lower().endswith('.txt'):
                with open(file, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
            elif file.lower().endswith('.pdf'):
                with open(file, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ''.join(page.extract_text() or '' for page in reader.pages)
                    texts.append(text)
            elif file.lower().endswith('.docx'):
                doc = Document(file)
                text = '\n'.join(paragraph.text for paragraph in doc.paragraphs)
                texts.append(text)

        self.documents = texts

    def _encode_documents(self):
        print("Encoding documents with SentenceTransformer...")
        embeddings = self.model.encode(
            self.documents,
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=True
        )
        self.doc_embeddings = F.normalize(embeddings, p=2, dim=1)  # (N, D)

    def search(self, query, top_k=3, threshold=0.6):
        query_emb = self.model.encode([query], convert_to_tensor=True, device=self.device)
        query_emb = F.normalize(query_emb, p=2, dim=1)  # (1, D)

        similarities = torch.mm(self.doc_embeddings, query_emb.T).squeeze(1)  # (N,)
        similarities = (similarities + 1.0) / 2.0  # [0, 1]

        mask = similarities >= threshold
        if not mask.any():
            return [(0, 0.0)]

        scores, indices = torch.sort(similarities, descending=True)
        results = []
        for score, idx in zip(scores, indices):
            if len(results) >= top_k:
                break
            if score >= threshold:
                results.append((idx.item() + 1, score.item()))
        return results

    def get_document_text(self, doc_index):
        if doc_index == 0:
            return "Не найдено ни одного документа."
        return self.documents[doc_index - 1]