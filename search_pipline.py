import os
import pickle
import glob
import torch.nn.functional as F
from docx import Document
import PyPDF2
import warnings
from transformers import AutoTokenizer, AutoModel
import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")


class CognitionSearch:

    def __init__(self, documents_folder_path, device):
        self.tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-multilingual-cased')
        self.bert_model = AutoModel.from_pretrained('google-bert/bert-base-multilingual-cased')
        self.st_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2').to(device)

        self.SIMILARITY_THRESHOLD = 0.6
        self.F_BETA = 0.8
        self.device = device
        self.documents = []

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="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",
        )

        self.bert_embeddings, self.st_embeddings = self.__processing_and_loading_documents(
            documents_folder_path,
            "documents_embeddings_bert",
            "documents_embeddings_st"
        )

    def __get_embeddings_bert(self, text_list):
        try:
            inputs = self.tokenizer(
                text_list,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors='pt'
            )
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            return outputs.last_hidden_state.cpu().detach()
        except Exception as e:
            print(e)
            return None

    def __get_embeddings_st(self, text_list):
        try:
            embeddings = self.st_model.encode(
                text_list,
                convert_to_tensor=True,
                device=self.device
            )
            return embeddings.unsqueeze(1).cpu().detach()
        except Exception as e:
            print(e)
            return None

    def __cross_attention(self, query, context):
        try:
            attention_logits = torch.matmul(query, context.transpose(-1, -2))
            attention_logits = attention_logits / (query.shape[-1] ** 0.5)
            attention_weights = torch.softmax(attention_logits, dim=-1)
            attention_out = torch.matmul(attention_weights, context)
            attention_out = torch.nn.functional.layer_norm(
                attention_out,
                attention_out.shape[-1:],
                weight=None,
                bias=None,
                eps=1e-6
            )
            return attention_out
        except Exception as e:
            print(e)
            return None

    def __canonical_correlation_attention(self, query, context):
        batch_size, seq_len_q, embed_dim = query.shape
        _, seq_len_c, _ = context.shape

        query_norm = F.normalize(query, dim=-1)
        context_norm = F.normalize(context, dim=-1)

        query_features = query_norm.mean(dim=1, keepdim=True)
        context_features = context_norm.mean(dim=1, keepdim=True)

        cov_matrix = torch.matmul(
            query_features.transpose(1, 2),
            context_features
        )

        query_transformed = torch.matmul(query_norm, cov_matrix)

        scores = torch.matmul(query_transformed, context_norm.transpose(1, 2))
        scores = scores / (embed_dim ** 0.5)

        attn_weights = F.softmax(scores, dim=-1)

        output = torch.matmul(attn_weights, context_norm)
        return F.normalize(output, dim=-1)

    def __f_measure_unification(self, attention_out, covariance_out):
        attention_norm = torch.norm(attention_out, dim=-1, keepdim=True)
        covariance_norm = torch.norm(covariance_out, dim=-1, keepdim=True)

        total_norm = attention_norm + covariance_norm + 1e-8
        dynamic_weight = attention_norm / total_norm
        return dynamic_weight * attention_out + (1 - dynamic_weight) * covariance_out

    def __cosine_score(self, vector1, vector2):
        try:
            sim = F.cosine_similarity(vector1, vector2)
            return (sim.item() + 1.0) / 2.0
        except Exception as e:
            print(e)
            return None

    def __processing_and_loading_documents(self, documents_folder_path, embeddings_folder_path_bert, embeddings_folder_path_st):
        try:
            if not os.path.exists(documents_folder_path):
                print(f"folder {documents_folder_path} not found")
                return None, None

            extensions = ['*.txt', '*.pdf', '*.docx']
            files = []

            for ext in extensions:
                pattern = os.path.join(documents_folder_path, ext)
                files.extend(glob.glob(pattern))
            files = sorted(files)

            if len(files) == 0:
                print(f"No files found in {documents_folder_path}")
                return None, None

            texts = []
            for file in files:
                if file.lower().endswith('.txt'):
                    with open(file, 'r', encoding='utf-8') as f:
                        texts.append(f.read())
                elif file.lower().endswith('.pdf'):
                    with open(file, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        text = ''
                        for page in reader.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text
                        texts.append(text)
                elif file.lower().endswith('.docx'):
                    doc = Document(file)
                    text = '\n'.join(paragraph.text for paragraph in doc.paragraphs)
                    texts.append(text)

            self.documents = texts

            os.makedirs(embeddings_folder_path_bert, exist_ok=True)
            os.makedirs(embeddings_folder_path_st, exist_ok=True)

            bert_embeddings = self.__get_embeddings_bert(texts)
            st_embeddings = self.__get_embeddings_st(texts)

            with open(os.path.join(embeddings_folder_path_bert, "bert_embeddings.pkl"), 'wb') as bert_file:
                pickle.dump(bert_embeddings, bert_file)

            with open(os.path.join(embeddings_folder_path_st, "st_embeddings.pkl"), 'wb') as st_file:
                pickle.dump(st_embeddings, st_file)

            return bert_embeddings, st_embeddings

        except Exception as e:
            print(e)
            return None, None

    def __search(self, query, top_k=3):
        try:
            bert_query_embeddings = self.__get_embeddings_bert([query])
            st_query_embeddings = self.__get_embeddings_st([query])

            batch_size = self.bert_embeddings.shape[0]
            relevant_documents = []
            for i in range(batch_size):
                documents_bert_embeddings = self.bert_embeddings[i, :, :]
                documents_st_embeddings = self.st_embeddings[i, :, :]

                attention_bert_out = self.__cross_attention(documents_bert_embeddings.unsqueeze(0), bert_query_embeddings)
                attention_st_out = self.__cross_attention(documents_st_embeddings.unsqueeze(0), st_query_embeddings)

                covariance_bert_out = self.__canonical_correlation_attention(documents_bert_embeddings.unsqueeze(0), bert_query_embeddings)
                covariance_st_out = self.__canonical_correlation_attention(documents_st_embeddings.unsqueeze(0), st_query_embeddings)

                bert_f_measure = self.__f_measure_unification(attention_bert_out, covariance_bert_out)
                st_f_measure = self.__f_measure_unification(attention_st_out, covariance_st_out)

                bert_f_measure_mean = torch.mean(bert_f_measure, dim=1, keepdim=True)
                st_f_measure_mean = torch.mean(st_f_measure, dim=1, keepdim=True)
                similarity1 = self.__cosine_score(bert_f_measure_mean.squeeze(0), torch.mean(documents_bert_embeddings, dim=0, keepdim=False).unsqueeze(0))
                similarity2 = self.__cosine_score(st_f_measure_mean.squeeze(0), torch.mean(documents_st_embeddings, dim=0, keepdim=False).unsqueeze(0))

                bert_weight = 0.6
                st_weight = 0.4

                similarity = (bert_weight * similarity1 + st_weight * similarity2)
                if similarity < self.SIMILARITY_THRESHOLD:
                    continue
                relevant_documents.append((i + 1, similarity))

            if len(relevant_documents) == 0:
                print(f"No documents found for {query}")
                return [(0, 0)]

            relevant_documents.sort(key=lambda x: x[1], reverse=True)
            return relevant_documents[:min(top_k, len(relevant_documents))]

        except Exception as e:
            print(e)
            return None

    def generate_answer(self, query, top_k=3):
        try:
            if len(query) == 0:
                print("Empty query")
                return None

            documents_and_indexes = self.__search(query, top_k)
            if not documents_and_indexes:
                return None

            context = ""
            if documents_and_indexes[0][0] == 0:
                context += f"<document_{0}>Не найдено ни одного документа.</document_{0}>"
            else:
                for i in range(len(documents_and_indexes)):
                    index = documents_and_indexes[i][0]
                    text = self.documents[index - 1]
                    context += f"<document_{index}>{text}</document_{index}>"

            prompt = f"Контекст: {context}. Запрос: {query}. Системный промпт: Ответь на вопрос, исходя из информации предоставленной в контексте, на том же языке, что и вопрос, как можно больше старайся цитировать предоставленные источники. Если документов не найдено, то просто ответь на вопрос без подкрепления информации документами. Текст пиши без Markdown разметки, прсото в виде параграфа."

            messages = [{"role": "user", "content": prompt}]
            response = self.client.chat.completions.create(
                model="mistralai/devstral-2512:free",
                messages=messages,
                stream=True
            )

            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    yield content

        except Exception as e:
            print(e)
            return None

if __name__ == '__main__':
    search = CognitionSearch("files", "mps")
    while True:
        query = input("Введи запрос: ")
        if query == "exit":
            break
        for chunk in search.generate_answer(query):
            print(chunk) if len(chunk) > 0 else None
        print()
