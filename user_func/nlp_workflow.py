import pandas as pd
import string
import re
import emoji
import nltk

def nlp_workflow(df:pd.DataFrame, columns:list):
    # check only columns with the 'object' data type is passed to this function
    # if not, throw an error and hault
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
    # function to fetch the word tag of each word in the list
    # reference to this function: https://www.geeksforgeeks.org/python-lemmatization-approaches-with-examples/
    def pos_tagger(nltk_tag):
        if nltk_tag.startswith('J'):
            return nltk.corpus.wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return nltk.corpus.wordnet.VERB
        elif nltk_tag.startswith('N'):
            return nltk.corpus.wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return nltk.corpus.wordnet.ADV
        else:         
            return None   
    # function to lemmatise the words
    # reference to this function: https://www.geeksforgeeks.org/python-lemmatization-approaches-with-examples/
    def lemmatiser(input):
        wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), nltk.pos_tag(input)))
        lemmatised = list()
        for word, tag in wordnet_tagged:
            if tag is None:
                # if there is no available tag, append the token as is
                lemmatised.append(word)
            else:       
                # else use the tag to lemmatize the token
                lemmatised.append(nltk.stem.WordNetLemmatizer().lemmatize(word, tag))
        return lemmatised
    # apply the functions defined above to the columns of interest
    # we make good use of lambda expressions
    # as those functions take care of strings one at a time
    # lambda function helps to traverse through all items in the column
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
        # remove stop words from the list of words
        df[new_col_name] = df[new_col_name].apply(lambda x: kill_stopwords(x))
        # lemmatise all words in the list of words
        df[new_col_name] = df[new_col_name].apply(lambda x: lemmatiser(x))
    return df