import pathlib
from langchain_core.documents import Document


def load_text_file(path: str) -> Document:
    text = pathlib.Path(path).read_text(encoding="utf-8")
    return Document(page_content=text, metadata={"source": path})


def load_directory(dir_path: str, pattern: str = "*.md") -> list[Document]:
    return [load_text_file(str(p)) for p in sorted(pathlib.Path(dir_path).glob(pattern))]
