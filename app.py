"""Простой сайт. Запуск: кнопка Run в PyCharm или streamlit run app.py."""
import sys
from pathlib import Path

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

# Кнопка Run запускает обычный Python, поэтому сначала включаем сервер Streamlit.
if __name__ == "__main__" and get_script_run_ctx(suppress_warning=True) is None:
    from streamlit.web import cli

    sys.argv = ["streamlit", "run", str(Path(__file__).resolve()),
                "--server.address", "127.0.0.1", "--server.port", "8501",
                "--browser.gatherUsageStats", "false"]
    raise SystemExit(cli.main())

from groq import APIConnectionError, APIStatusError, AuthenticationError, RateLimitError

from rag import build_search, find_sources, generate_answer

st.set_page_config(page_title="RAG по статьям Хабра", page_icon="📚")
st.title("RAG по статьям Хабра")
st.write("Введите вопрос. LangChain найдёт фрагменты статей, а Groq составит ответ.")


@st.cache_resource
def prepare_search():
    # Поиск строится один раз и остаётся в памяти до остановки приложения.
    return build_search()


with st.form("question_form"):
    question = st.text_area("Вопрос", placeholder="Что изменилось в политике SoundCloud относительно ИИ?",
                            max_chars=1500)
    submitted = st.form_submit_button("Получить ответ", type="primary")

if submitted:
    if len(question.strip()) < 2:
        st.warning("Введите вопрос к статьям.")
    else:
        try:
            with st.spinner("Читаем статьи и готовим поиск. Первый запрос займёт больше времени…"):
                retriever, articles_count, chunks_count = prepare_search()
            st.caption(f"В базе: {articles_count} статей, {chunks_count} фрагментов.")
            with st.spinner("Ищем фрагменты и получаем ответ Groq…"):
                sources = find_sources(retriever, question.strip())
            st.subheader("Ответ")
            with st.spinner("Получаем ответ Groq…"):
                st.write_stream(generate_answer(question.strip(), sources))
            for i, doc in enumerate(sources, 1):
                with st.expander(f"[{i}] {doc.metadata['title']}"):
                    st.write(doc.page_content)
                    st.link_button("Открыть статью на Хабре", doc.metadata["url"])
        except AuthenticationError:
            st.error("Groq не принял ключ. Проверьте его в личном кабинете.")
        except RateLimitError:
            st.error("Достигнут лимит Groq. Подождите и повторите запрос.")
        except APIConnectionError:
            st.error("Не удалось соединиться с Groq. Проверьте интернет.")
        except APIStatusError:
            st.error("Groq отклонил запрос. Проверьте доступ к модели, указанной в rag.py.")
        except FileNotFoundError:
            st.error("Не найден data/habr_10000.csv.bz2. Поместите файл из задания в папку data.")
        except ValueError as error:
            st.error(str(error))
