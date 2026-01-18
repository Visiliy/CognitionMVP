# compare_approaches.py
import argparse
from cosine_baseline import CosineBaselineSearch
from search_pipline import CognitionSearch  # ваша сложная модель
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=str, default="files", help="Path to documents folder")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("Загрузка базовой модели (косинусное сходство)...")
    baseline = CosineBaselineSearch(args.docs, device=args.device)

    print("Загрузка продвинутой модели (CognitionSearch)...")
    advanced = CognitionSearch(args.docs, device=args.device)

    while True:
        query = input("\nВведите запрос (или 'exit' для выхода): ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue

        print("\n" + "="*60)
        print("🔍 Запрос:", query)
        print("="*60)

        # --- Базовый подход ---
        print("\n[БАЗОВЫЙ ПОДХОД: Косинусное сходство]")
        baseline_results = baseline.search(query, top_k=3, threshold=0.6)
        for idx, sim in baseline_results:
            print(f"  Документ {idx}: сходство = {sim:.4f}")
            if idx != 0:
                preview = " ".join(baseline.get_document_text(idx).split()[:20]) + "..."
                print(f"    Превью: {preview}")

        # --- Продвинутый подход ---
        print("\n[ПРОДВИНУТЫЙ ПОДХОД: CognitionSearch]")
        try:
            advanced_results = advanced._CognitionSearch__search(query, top_k=3)
            for idx, sim in advanced_results:
                print(f"  Документ {idx}: сходство = {sim:.4f}")
                if idx != 0:
                    preview = " ".join(advanced.documents[idx - 1].split()[:20]) + "..."
                    print(f"    Превью: {preview}")
        except Exception as e:
            print(f"  Ошибка в продвинутом поиске: {e}")

        # --- Генерация ответа (только продвинутая модель) ---
        print("\n[ГЕНЕРАЦИЯ ОТВЕТА (CognitionSearch)]")
        try:
            full_answer = ""
            for chunk in advanced.generate_answer(query, top_k=3):
                full_answer += chunk
                print(chunk, end='', flush=True)
            print()
        except Exception as e:
            print(f"Ошибка генерации: {e}")

if __name__ == "__main__":
    main()