from langchain.document_loaders import TextLoader


def start():
    loader = TextLoader('./mediumblogs/mediumblog1.txt')
    document = loader.load()
    print(document)


if __name__ == '__main__':
    print('Hello, VectorStore!')
    start()
