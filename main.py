import pandas as pd
import numpy as np
import re


def get_words(filename):
    word_list = []
    temp = []
    filo = open(filename, "r")
    data = filo.read()
    words = data.split()
    # print(words)

    for j in words:
        if j.find('—') == -1:
            temp.append(j)
        else:
            temp.append(j[0:j.find('—')])
            temp.append(j[j.find('—') + 1:])
    for i in temp:
        i = i.lower()
        word_list.append(re.sub(r"[^A-Za-z0-9]", "", i))
    return word_list


def create_dictionary(word_list):
    dictionary = {}
    for i in word_list:
        if i in dictionary:
            dictionary[i] += 1
        else:
            dictionary.update({i: 1})
    return dictionary


def create_dataframe(word_dictionary):
    return pd.DataFrame(word_dictionary, index=[0])
    # https://stackoverflow.com/questions/17839973/constructing-pandas-dataframe-from-values-in-variables-gives-valueerror-if-usi


def update_dataframe(dataframe, merged_dictionary, dictionary, document):
    for i in merged_dictionary:
        if dictionary.get(i) is None:
            dataframe.loc[dataframe['Document'] == document, i] = 0
        else:
            dataframe.loc[dataframe['Document'] == document, i] = dictionary.get(i)
    return dataframe


# tf - number of times word occurs/ number of words
def term_frequency(dataframe):
    dataframe = dataframe.iloc[:, 1:].apply(lambda x: x / dataframe['# of words'])
    return dataframe


# idf - ln(N --> number of rows/1+nt)
def check_word_idf(word, dictionary):
    if word in dictionary.keys():
        return 1
    return 0
    # return 1 for found, 0 for not found


def main():
    # gets list of words in document
    Frank_list = get_words('Frankenstein.txt')
    Gatsby_list = get_words('TheGreatGatsby.txt')
    Wallpaper_list = get_words('TheYellowWallpaper.txt')
    # print(len(Frank_list))
    # print(len(Gatsby_list))

    # created dictionary of number of occurrences of each word in document
    Frank_dictionary = create_dictionary(Frank_list)
    Gatsby_dictionary = create_dictionary(Gatsby_list)
    Wallpaper_dictionary = create_dictionary(Wallpaper_list)
    # print(len(Frank_dictionary))
    # print(len(Gatsby_dictionary))

    # created merged dictionary to have every word in both documents (w/o repeats)
    merged_dictionary = {}
    merged_dictionary.update(Frank_dictionary)
    merged_dictionary.update(Gatsby_dictionary)
    merged_dictionary.update(Wallpaper_dictionary)
    dataframe = create_dataframe(merged_dictionary)

    # initializing 2nd and 3rd row to be all 1
    dataframe.loc[1, :] = 1
    dataframe.loc[2, :] = 1

    # add column Document and formatting column position
    dataframe['Document'] = ['Frankenstein', 'The Great Gatsby', 'The Yellow Wallpaper']
    name_column = dataframe.pop('Document')
    dataframe.insert(0, 'Document', name_column)
    # https://www.geeksforgeeks.org/how-to-move-a-column-to-first-position-in-pandas-dataframe/

    # updating dataframe rows with correct word occurrences
    dataframe = update_dataframe(dataframe, merged_dictionary, Frank_dictionary, 'Frankenstein')
    dataframe = update_dataframe(dataframe, merged_dictionary, Gatsby_dictionary, 'The Great Gatsby')
    dataframe = update_dataframe(dataframe, merged_dictionary, Wallpaper_dictionary, 'The Yellow Wallpaper')
    dataframe['# of words'] = (len(Frank_list), len(Gatsby_list), len(Wallpaper_list))

    # created new dataframe with tf
    print(dataframe)
    tf_dataframe = term_frequency(dataframe)
    tf_dataframe = tf_dataframe.loc[:, tf_dataframe.columns != '# of words']
    print(tf_dataframe)

    # list with idf
    idf_list = []
    N = len(dataframe)
    for i in merged_dictionary.keys():
        idf_list.append(np.log(N / (
                    1 + check_word_idf(i, Frank_dictionary) + check_word_idf(i, Gatsby_dictionary) + check_word_idf(i,
                                                                                                                    Wallpaper_dictionary))))
    # print(idf_list) #0.4054651081081644, 0.0, -0.2876820724517809

    # multiply tf and idf to get tfidf
    tfidf_data = tf_dataframe * idf_list
    print(tfidf_data)

    # words with max tfidf values in each story
    copy_tfidf = tfidf_data.copy()
    # https://stackoverflow.com/questions/14734695/get-column-name-where-value-is-something-in-pandas-dataframe
    Frank_max_values = []
    Frank_max_words = []
    Gatsby_max_values = []
    Gatsby_max_words = []
    Wallpaper_max_values = []
    Wallpaper_max_words = []
    for i in range(5):
        # Frankenstein highest tfidf
        max1 = copy_tfidf.iloc[0].max()
        max_word1 = copy_tfidf.apply(lambda row: row[row == max1].index, axis=1)
        Frank_max_values.append(max1)
        Frank_max_words.append(max_word1)
        # Gatsby highest tfidf
        max2 = copy_tfidf.iloc[1].max()
        max_word2 = copy_tfidf.apply(lambda row: row[row == max2].index, axis=1)
        Gatsby_max_values.append(max2)
        Gatsby_max_words.append(max_word2)
        # Yellow Wallpaper highest tfidf
        max3 = copy_tfidf.iloc[2].max()
        max_word3 = copy_tfidf.apply(lambda row: row[row == max3].index, axis=1)
        Wallpaper_max_values.append(max3)
        Wallpaper_max_words.append(max_word3)
        # replacing tfidf values
        copy_tfidf = copy_tfidf.replace(max1, np.NaN)
        copy_tfidf = copy_tfidf.replace(max2, np.NaN)
        copy_tfidf = copy_tfidf.replace(max3, np.NaN)
    # print(Frank_max_values) # [0.0005001579296136748, 0.0004495801614504941, 0.0003315653690697394, 0.0003090863609972147, 0.00028660735292469]
    # print(Frank_max_words)  # elizabeth, feelings, clerval, misery, justine, cottage
    # print(Gatsby_max_values) # [0.0016368931243913537, 0.0014782147092717835, 0.001252724329891342, 0.0006096591738804531, 0.0005595502006847994]
    # print(Gatsby_max_words)  # gatsby, tom, daisy, car, gastbys
    # print(Wallpaper_max_values) # [0.0007925690336044914, 0.0005944267752033685, 0.0003302370973352047, 0.0002641896778681638, 0.00019814225840112285]
    # print(Wallpaper_max_words)  # wallpaper, jennie, creeping, nursery, smell, arbors, color, daytime, queer, bedstead, shines, fungus

    # words with highest tf values (most common words) in each story
    copy_tf = tf_dataframe.copy()
    Frank_max_tf_values = []
    Frank_max_tf_words = []
    Gatsby_max_tf_values = []
    Gatsby_max_tf_words = []
    Wallpaper_max_tf_values = []
    Wallpaper_max_tf_words = []
    for i in range(10):
        # Frankenstein highest tf
        max_tf1 = copy_tf.iloc[0].max()
        max_term1 = copy_tf.apply(lambda row: row[row == max_tf1].index, axis=1)
        Frank_max_tf_values.append(max_tf1)
        Frank_max_tf_words.append(max_term1)
        # Gatsby highest tf
        max_tf2 = copy_tf.iloc[1].max()
        max_term2 = copy_tf.apply(lambda row: row[row == max_tf2].index, axis=1)
        Gatsby_max_tf_values.append(max_tf2)
        Gatsby_max_tf_words.append(max_term2)
        # Yellow Wallpaper highest tf
        max_tf3 = copy_tf.iloc[2].max()
        max_term3 = copy_tf.apply(lambda row: row[row == max_tf3].index, axis=1)
        Wallpaper_max_tf_values.append(max_tf3)
        Wallpaper_max_tf_words.append(max_term3)
        # replacing tf values
        copy_tf = copy_tf.replace(max_tf1, np.NaN)
        copy_tf = copy_tf.replace(max_tf2, np.NaN)
        copy_tf = copy_tf.replace(max_tf3, np.NaN)
    # print(Frank_max_tf_values) # [0.0552044352044352, 0.03991683991683992, 0.0386001386001386, 0.03426195426195426, 0.027096327096327096, 0.02327096327096327, 0.017948717948717947, 0.01465003465003465, 0.014192654192654192, 0.013568953568953568]
    # print(Frank_max_tf_words)  # the, and, i, of, to, my, a, in, that, was
    # cprint(Gatsby_max_tf_values) # [0.04935118434603501, 0.03227600411946447, 0.02898043254376931, 0.02455200823892894, 0.02325437693099897, 0.022966014418125645, 0.016745623069001028, 0.016601441812564368, 0.015818743563336766, 0.01223480947476828]
    # print(Gatsby_max_tf_words)  # the, and, a, i, to, of, he, in, was, that
    # print(Wallpaper_max_tf_values) #[0.049845251669653035, 0.04691317804202639, 0.03909431503502199, 0.02818048542107835, 0.023293696041700604, 0.02313080306238801, 0.021501873269262096, 0.017918227724385078, 0.017429548786447303, 0.015149047076071022]
    # print(Wallpaper_max_tf_words)  # i, and, the, it, a, to , is, that, of, in
    # print(tfidf_data['the']) #-0.015881, -0.014197, -0.011247
    # print(tfidf_data['and'])  # -0.011483, -0.009285, -0.013496
    # print(tfidf_data['i'])  # -0.011105, -0.007063, -0.014340
    # print(tfidf_data['a'])  # -0.005164, -0.008337, -0.006701
    # print(tfidf_data['of'])  # -0.009857, -0.006607, -0.005014
    # print(tfidf_data['that'])  # -0.004083, -0.003520, -0.005155
    # print(tfidf_data['in'])  # -0.004215, -0.004776, -0.004358



main()
