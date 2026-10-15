"""Шаг 1: читаем выданный CSV и превращаем статьи в документы LangChain."""
import bz2
import csv
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_core.documents import Document

DATA_FILE = Path(__file__).parent / "data" / "habr_10000.csv.bz2"


def load_articles(path=DATA_FILE):
    csv.field_size_limit(20_000_000)  # В CSV встречаются длинные статьи.
    articles = []
    with bz2.open(path, "rt", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            html = BeautifulSoup(row["post_html"], "html.parser")
            for tag in html(["script", "style"]):
                tag.decompose()
            text = html.get_text("\n", strip=True)
            articles.append(Document(
                page_content=row["post_title"] + "\n\n" + text,
                metadata={"title": row["post_title"],
                          "url": f"https://habr.com/ru/articles/{row['post_id']}/"},
            ))
    return articles
