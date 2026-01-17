import requests
# Назначение: библиотека позволяет отправлять HTTP-запросы (GET, POST и другие), обрабатывать ответы сервера, а также получать данные от внешних API-сервисов.
import json
# Назначение: модуль позволяет сериализовать (преобразовать в JSON) объекты Python и десериализовать (преобразовать из JSON) в объекты Python.
from datetime import datetime
# Назначение: модуль содержит функции для работы с датой и временем.

BASE_URL = "http://127.0.0.1:8000"
CORPUS_FILE = "../data/corpus.txt"


def load_texts_from_file(filename):
    """Загрузка корпуса текстов из файла"""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
            # очистка от невидимых символов (\n) и фильтрация пустых строк, чтобы не нагружать сервер бесполезными данными
        return texts
    except FileNotFoundError:
        print(f"Файл не найден: {filename}")
        return None


def check_server():
    """Проверка доступности сервера"""
    try:
        response = requests.get(f"{BASE_URL}/", timeout=2) # отправляет короткий GET-запрос с timeout=2 (ожидание не более 2 секунд)
        return response.status_code == 200 # код ошибки
    except requests.exceptions.RequestException: # исключения 
        return False


def send_request(endpoint, payload): # адрес и посылка
    """Отправка POST-запроса"""
    try:
        response = requests.post(
            f"{BASE_URL}{endpoint}", # превращает части адреса в полную ссылку, например: http://127.0.0.1. 
            json=payload, # превращает текст из словаря в json
            timeout=30 # у него есть 30 сек.
        )
        response.raise_for_status() #проверяет сатутс, если есть ошибка то сразу ведет в эксепт
        return response.json() # если все ок - возвращает ответ
    except requests.exceptions.RequestException as e: # исключения 
        print(f"Ошибка при запросе {endpoint}: {e}")
        return None


def main():
    print("Проверка доступности сервера...")
    if not check_server():
        print(f"Сервер недоступен: {BASE_URL}")
        print(">> Запустите сервер командой:")
        print("   uvicorn app.main:app --reload")
        return

    print("Сервер доступен (хвала богам)")
    print("-" * 50)

    texts = load_texts_from_file(CORPUS_FILE)
    if texts is None or len(texts) == 0:
        print("Корпус пуст или не загружен")
        return

    print(f"Загружено текстов: {len(texts)}")
    print("-" * 50)

    results = {}

    print("Отправка корпуса на сервер...")

    # Bag of Words
    bow_result = send_request("/bag-of-words", {"texts": texts})
    if bow_result is not None:
        results["bag-of-words"] = bow_result

    # TF-IDF
    tfidf_result = send_request("/tf-idf", {"texts": texts})
    if tfidf_result is not None:
        results["tf-idf"] = tfidf_result

    # NLTK (анализ первого текста)
    nltk_result = send_request("/text_nltk", {"text": texts[0]})
    if nltk_result is not None:
        results["text_nltk"] = nltk_result

    # Сохранение результатов (автоматическую генерацию уникального имени для файла с результатами)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # создается «отпечаток времени», чтобы файлы не назывались одинаково
    output_file = f"results_{timestamp}.json"
    # Итог: Переменная output_file будет содержать строку вроде results_20231027_143005.json

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Результаты сохранены в файл: {output_file}")
    except Exception as e:
        print(f"Ошибка при сохранении результатов: {e}")


if __name__ == "__main__":
    main()
# «Если этот файл запустили напрямую (как главную программу), то выполняй функцию main(). 
# Если же этот файл просто кто-то импортировал в другой скрипт — ничего не запускай».
