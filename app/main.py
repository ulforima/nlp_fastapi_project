from fastapi import FastAPI
from app.nlp.bow import bag_of_words
from app.nlp.tfidf import tf_idf
from app.nlp.nltk_tools import process_text #наш файл (nltk_tools.py), содержащий вспомогательные функции, путь к нему
#конкретная функция или класс внутри этого файла
from app.nlp.word2vec import word2vec_analysis
app = FastAPI(title="NLP Microservice") 
# создание экземпляра приложения: создает экземпляр класса фаст апи и присваивает его переменной апп
# это основная точка взаимодействия для настройки всего апи
# параметр title используется для автоматической документации

@app.get("/") #декоратор сообщает FastAPI, что функция root должна обрабатывать HTTP GET-запросы 
# (основной метод протокола HTTP, предназначенный для получения (запроса) данных с сервера) 
# по корневому пути (/)
def root():
    return {
        "message": "NLP microservice is running",
        "endpoints": [
            "/bag-of-words",
            "/tf-idf",
            "/text_nltk"
        ]
    }
# определяет функцию, которая будет вызываться при обращении к этому маршруту. 
# она возвращает словарь Python, 
# который FastAPI автоматически преобразует в JSON-ответ - 
# это текстовые данные в формате JSON, которые сервер отправляет клиенту (браузеру или приложению) 
# в качестве результата обработки HTTP-запроса. 


@app.post("/bag-of-words")
def bow_endpoint(data: dict):
    texts = data["texts"]
    return bag_of_words(texts)

# Декоратор @app.post("/bag-of-words"): определяет маршрут для обработки HTTP POST-запросов по пути /bag-of-words
# /bag-of-words - (эндпоинт (или «ручка») нашего API. Если проводить аналогию, то это конкретный «отдел» в цифровом офисе.)
# эндпоинт (endpoint) — это конкретный адрес, по которому можно вызвать определённую функцию твоего сервера через интернет
# POST-запросы обычно используются для отправки данных на сервер (например, для создания данных или запуска обработки)

# Функция bow_endpoint(data: dict): принимает входящие данные (тело запроса) в виде словаря Python (data: dict)
# Извлекает список текстов и передает его в импортированную функцию bag_of_words()

@app.post("/tf-idf")
def tfidf_endpoint(data: dict):
    texts = data["texts"]
    return tf_idf(texts)

@app.post("/text_nltk")
def nltk_endpoint(data: dict):
    text = data["text"]
    return process_text(text)

@app.post("/word2vec")
def word2vec_endpoint(data: dict):
    texts = data["texts"]
    method = data.get("method", "tfidf")  # По умолчанию используем tfidf
    return word2vec_analysis(texts, method=method)


# Почему это удобно?
# не нужно создавать разные серверы для разных задач. просто создаем разные пути (маршруты (эндпоинты)):
# /bag-of-words — для векторизации.
# /tf-idf — для оценки важности слов.
# /text_nltk — для очистки текста.



