from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

from langchain.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings()
persist_directory = './chroma/'

loader = TextLoader("./resume.md")
markdown_documents = loader.load()

# with open("resume.md", "r") as f:
#     markdown_text = f.read()
# print(markdown_document)

chunk_size =26
chunk_overlap = 4

# r_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=450,
#     chunk_overlap=0,
#     separators=["\n\n", "\n", "(?<=\. )", " ", ""]
# )

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    return_each_line=True
)
print(len(markdown_documents))
print(markdown_documents[0].page_content)

md_header_splits = markdown_splitter.split_text(markdown_documents[0].page_content)
print(len(md_header_splits))

for split in md_header_splits:
    print(split)
    print()

print(type(md_header_splits[0]))
vectordb = Chroma.from_documents(
    documents=md_header_splits,
    embedding=embedding,
    persist_directory=persist_directory
)
print(vectordb._collection.count())