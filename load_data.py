import os

DATA_DIR = r"data\20_newsgroups"
documents = []
labels = []

for category in os.listdir(DATA_DIR):

    category_path = os.path.join(DATA_DIR, category)

    if os.path.isdir(category_path):

        for file in os.listdir(category_path):

            file_path = os.path.join(category_path, file)

            with open(file_path, "r", encoding="latin1") as f:
                text = f.read()

                documents.append(text)
                labels.append(category)

print("Total documents:", len(documents))
print("First category:", labels[0])
print("Sample text:\n", documents[0][:300])