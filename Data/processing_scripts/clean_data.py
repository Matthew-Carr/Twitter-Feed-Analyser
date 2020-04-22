import re
import spacy
import pandas as pd

nlp = spacy.load('en_core_web_md')

# def clean_subject(subject: str) -> str:
#     subject = ' '.join(subject.split())
#     return subject.strip().lower()

isascii = lambda s: len(s) == len(s.encode())


# def clean_string(text: str, subject: str) -> str:
#     text += ' '
#     text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # @... until a space is found
#     text = re.sub("[0-9]+", "", text)  # remove numbers
#     text = ' '.join(text.split())  # remove duplicate whitespaces
#     text = text.strip()  # remove leading and trailing whitespaces
#     text = text.lower()  # convert to lower-case
#     text = text.replace(subject, 'subject')
#
#     # special occurrences of strange characters e.g. &nbsp;
#     text = text.replace('&nbsp;', ' ')
#     text = text.replace('&amp;', '&')
#     text = text.replace('&gt;', '>')
#     text = text.replace('&lt;', '<')
#
#     text = ' '.join([token.text for token in nlp(text) if isascii(token.text) and token.text not in {'#'}])
#     text = re.sub(r'([@][\w_-]+)', '<user>', text)
#     text = re.sub(r'http\S+', '<url>', text)  # remove URLs
#
#     return text

def clean_string(text: str) -> str:
    text += ' '
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # @... until a space is found
    text = re.sub("[0-9]+", "", text)  # remove numbers
    text = ' '.join(text.split())  # remove duplicate whitespaces
    text = text.strip()  # remove leading and trailing whitespaces
    text = text.lower()  # convert to lower-case

    # special occurrences of strange characters e.g. &nbsp;
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&gt;', '>')
    text = text.replace('&lt;', '<')

    text = ' '.join([token.text for token in nlp(text) if isascii(token.text) and token.text not in {'#'}])
    text = re.sub(r'([@][\w_-]+)', '<user>', text)
    text = re.sub(r'http\S+', '<url>', text)  # remove URLs

    return text


def clean_dataset(data: pd.DataFrame):

    data['clean_data'] = data['text']
    for index, row in data.iterrows():
        if index % 200 == 0:
            print(index)
        data.loc[index, 'clean_data'] = clean_string(row['text'])
    data['clean_data'].to_csv(path_or_buf='../processed_data/CleanData.csv',
                              index=False, header=['clean_data'])


if __name__ == '__main__':
    data_file = pd.read_csv("../processed_data/RawData.csv", encoding="ISO-8859-1")
    clean_dataset(data_file)
