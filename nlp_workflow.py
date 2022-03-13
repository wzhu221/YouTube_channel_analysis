import pandas as pd
import string
import re
import emoji
import nltk

def nlp_workflow(df:pd.DataFrame, columns:list):
    for col_name in columns:
        if df[col_name].dtype != 'O':
            raise TypeError (f'Wrong data type for column: {col_name}.')
        else: 
            pass
    # function to remove punctuations in the review content, 
    # and to change the review content to all lower case letters
    def kill_punct(input):
        no_punct = [word for word in input if word not in string.punctuation]
        content_no_punct = ''.join(no_punct)
        return content_no_punct
    # function to remove stop words
    def kill_stopwords(input):
        no_stopword = list(word for word in input if word not in nltk.corpus.stopwords.words('english'))
        return no_stopword
    # function to extract word stems
    def stemmer(input):
        stemmed = list(nltk.stem.porter.PorterStemmer().stem(word) for word in input)
        return stemmed
    for col_name in columns:
        new_col_name = col_name + '_nlp'
        # remove punctuations in the review content and make lower case
        df[new_col_name] = df[col_name].apply(lambda x: kill_punct(x.lower()))
        # remove emojis in the review content
        df[new_col_name] = df[new_col_name].apply(lambda x: re.sub(emoji.get_emoji_regexp(), '', x))
        # remove leading and trailing whitespaces of the review content
        df[new_col_name] = df[new_col_name].str.strip()
        # tokenise the review content into a list of words
        df[new_col_name] = df[new_col_name].apply(lambda x: nltk.word_tokenize(x))
        df[new_col_name] = df[new_col_name].apply(lambda x: kill_stopwords(x))
        df[new_col_name] = df[new_col_name].apply(lambda x: stemmer(x))
    return df