# Media Transcription Service

Сервис для преобразования видеозаписей в структурированный текст с использованием технологий искусственного интеллекта.

Сервис был реализован мной в виде челенджа, который звучал так: можно ли за выходные сделать сервис с нуля, на 
фреймворке с которым я не работал. Из наработок был опыт работы с Whisper, LangChain, Ollama Gemma. Сервис создавался в 
плотном сотрудничестве с Anthropic Claude 3.7. Без помощи ИИ оцениваю написание этого сервиса для себя в 1 неделю рабочую.

## Основной функционал

- Загрузка видеофайлов
- Извлечение аудио из видео
- Транскрибирование аудио в текст с помощью OpenAI Whisper
- Обработка текста с использованием GPT-4o или Ollama Gemma
- Получение различных типов обработки:
  - Краткое содержание встречи
  - Формирование технического задания
  - Пользовательские запросы

## Технический стек

### Бэкенд
- Python 3.10+
- FastAPI
- SQLAlchemy
- Alembic (миграции)
- PostgreSQL
- RabbitMQ
- OpenAI API
- Langchain
- FFmpeg

### Фронтенд
- HTML/CSS/JavaScript
- jQuery
- Bootstrap 5

## Быстрый старт

### Локальная разработка

1. Клонировать репозиторий:
   ```bash
   git clone https://github.com/your-username/media-transcription-service.git
   cd media-transcription-service
   ```

2. Настроить переменные окружения:
   ```bash
   cp .env.example .env
   # Отредактируйте .env, добавив ваш OPENAI_API_KEY и другие настройки
   ```

3. Загрузить модель Gemma для Ollama:
   ```bash
   curl -X POST http://localhost:11434/api/pull -d '{"name": "gemma:7b"}'
   ```

4. Открыть в браузере:
   ```
   http://localhost:8000/
   ```

5Войти с учетными данными по умолчанию:
   - Логин: `admin`
   - Пароль: `admin`

## Структура проекта

```
media-transcription-service/
├── backend/            # FastAPI бэкенд
│   ├── app/            # Код приложения
│   │   ├── api/        # API эндпоинты
│   │   ├── core/       # Конфигурация и утилиты
│   │   ├── db/         # Работа с базой данных
│   │   ├── models/     # SQLAlchemy модели
│   │   ├── schemas/    # Pydantic схемы
│   │   ├── services/   # Сервисы
│   │   ├── templates/  # Jinja2 шаблоны
│   │   ├── worker/     # Логика воркера
│   │   └── main.py     # Точка входа приложения
├── alembic/            # Миграции
├── worker/             # Код фонового обработчика
│   ├── worker.py       # Основной код воркера
└── .env.example        # Пример файла переменных окружения
```

## Расширение функционала

### Добавление нового типа обработки текста

1. Добавить новый тип в модель `TranscriptionType` в файле `backend/app/models/transcription.py`
2. Обновить шаблон выбора типа `backend/app/templates/user/select_type.html`
3. Добавить соответствующий промт в функцию `process_with_llm` в `worker/worker.py`

### Интеграция с другой моделью ИИ

1. Добавить новый клиент в `worker/worker.py` в функции `process_with_llm`
2. Обновить переменные окружения в `.env` и Kubernetes конфигурациях
