from __future__ import annotations
import re
from chatgpt_lib.api_log import log
from chatgpt_lib import WORK_DIR
import json
import configparser
from nltk import word_tokenize
import os
import spacy
nlp = spacy.load( "en_core_web_sm" )

config = configparser.ConfigParser()
config_path = os.path.join(WORK_DIR, 'config', 'config.ini')
config.read(config_path)
BREAK_ON_X = int(config.get('main', 'BREAK_ON_X'))
B_Sentence_size_Y = int(config.get('main', 'B_Sentence_size_Y'))


def spacy_parse(text):
    parse = nlp(text)
    return [str(item) for item in parse.sents]


def parse_response(text: str, delete_quoted: bool = False):
    """
    Return list of text paragraphs of response, sentences' and words' numbers

    :param text: str. Uncleaned text.
    :param delete_quoted: bool. To delete quoted sentences?
    """
    # replace 3 spaces or tabs after .!? with new line
    text = re.sub(r'(?<=[.!?])[ \t]{3,}', '\n', text)
    # replace other 3 spaces or tabs with 1 space
    text = re.sub(r'[ \t]{3,}', ' ', text)
    # split lines by any count of new line characters
    par_list = [x for x in re.split(r'[\n\r\f\v]+', text) if x.strip()]
    # get sentences 2d list (divided by paragraph)
    # and count sentences' number (minimum one letter between dots)
    text_parsing = TextParsingV3(par_list, delete_quoted=delete_quoted)
    return {'par_list': par_list,
            'sent_number': text_parsing.sent_number,
            'words_number': text_parsing.words_number,
            'sent_2d_list': text_parsing.sent_2d_list,
            'words_3d_list': text_parsing.words_3d_list,
            'quotes_failed': text_parsing.quotes_failed
            }


class SpecialTokens:
    prefix = 'startsespecialtoken'
    postfix = 'endsespecialtoken'
    regex = re.compile(prefix + r'\d+' + postfix)

    def __init__(self):
        self.counter = 0
        self.token_name__value: dict[str, str] = dict()

    def add(self, value: str) -> str:
        """
        Get new token name and save

        :return: str. Token name, e.g. abcsespecialtokens1
        """
        self.counter += 1
        token_name = f"{self.prefix}{self.counter}{self.postfix}"
        self.token_name__value[token_name] = value
        return token_name

    def cipher(self, regex: re.Pattern, text: str) -> str:
        cur_text = text
        while True:
            result = regex.search(cur_text)
            if result:
                start, end = result.span()
                value = cur_text[start: end]
                token_name = self.add(value)
                cur_text = cur_text[:start] + token_name + cur_text[end:]
            else:
                break
        return cur_text

    def decipher(self, text: str) -> str:
        cur_text = text
        while True:
            result = self.regex.search(cur_text)
            if result:
                start, end = result.span()
                token_name = cur_text[start: end]
                value = self.token_name__value[token_name]
                cur_text = cur_text[:start] + value + cur_text[end:]
            else:
                break
        return cur_text


class TextParsingV3:
    end_of_sentence_marks = {'.', '?', ':', ';', '!', '...', '…', '"'}
    digits_str = {str(x) for x in range(10)}
    regex_decimal_numbers = re.compile(r'\d*\.\d+')

    def __init__(self,
                 par_list: list,
                 delete_quoted: bool = False,
                 ):
        """
        Text Parsing V3 (2023-01-30).
        Convert paragraphs' list to 2d list of sentences.
        Deleting sentences in double quotes.
        Additional dots/commas parsing.

        :param par_list: list List of bulk text paragraphs
        :param delete_quoted: bool. To delete quoted sentences?
        """
        self.par_list = par_list
        self.delete_quoted = delete_quoted
        # self.sql = sql_obj
        # `abbr_ignore` MySQL table rows: list of dictionaries
        abbr_filename = os.path.join(WORK_DIR, 'data', 'abbr.json')
        with open(abbr_filename, 'r', encoding='utf-8') as f:
            abbr = f.read()
        self.abbr_ignore = json.loads(abbr)
        ##### self.abbr_ignore = AbbrIgnoreTable(sql_obj).fetch()
        ##### self.resp_xforms_dict = RespXFormsTable(sql_obj).get_resps_xforms_dict()
        contractions_filename = os.path.join(WORK_DIR, 'data', 'contractions.json')
        with open(contractions_filename, 'r', encoding='utf-8') as f:
            contractions = f.read()
        self.resp_xforms_dict = json.loads(contractions)
        for in_txt in self.resp_xforms_dict:
            self.abbr_ignore.append({'abbr': in_txt, 'ignore_case': 1})
        self.special_tokens = SpecialTokens()
        self.BREAK_ON_X = BREAK_ON_X
        self.B_Sentence_size_Y = B_Sentence_size_Y  # y <= x
        # to be parsed:
        self.sent_2d_list: list[list[str]] = []
        self.sent_number: int = 0
        self.words_3d_list: list[list[list[str]]] = []
        self.words_number: int = 0
        # errors:
        self.quotes_failed: bool = False
        self.no_punctuation: bool = False
        self.text2sentences()
        self.sentences2words()
        self._decipher()

    def _find_convert_special_tokens(self, paragraph: str) -> str:
        """
        Find and convert special tokens:
        1) Tokens from `abbr_ignore` MySQL table
        2) decimal numbers like 2.8

        :param paragraph: str. Initial paragraph text
        :return: str. Paragraph text with converted special tokens
        """
        cur_text = paragraph
        # abbr ignore + resp_xforms
        for ai_dict in self.abbr_ignore:
            for prefix in (r'\W', r'^'):
                regex_template = f"(?<={prefix})" + re.escape(ai_dict['abbr'])
                regex = re.compile(regex_template, re.I if ai_dict['ignore_case'] else 0)
                cur_text = self.special_tokens.cipher(regex, cur_text)
        # decimal numbers like 2.8 or .123
        cur_text = self.special_tokens.cipher(self.regex_decimal_numbers, cur_text)
        return cur_text

    def _decipher(self):
        sent_2d_list = []
        for par_sent_list in self.sent_2d_list:
            sent_2d_list.append([self.special_tokens.decipher(sentence) for sentence in par_sent_list])
        self.sent_2d_list = sent_2d_list
        word_3d_list = []
        for list2d in self.words_3d_list:
            word_3d_list.append([])
            for list1d in list2d:
                word_3d_list[-1].append([self.special_tokens.decipher(word) for word in list1d])
        self.words_3d_list = word_3d_list

    def _sent_tokenize(self, text: str) -> (list[str], str):
        """
        Divide text to sentences by end of sentence marks

        :param text: str
        :return: list of str, str. List of sentences, error message
        """
        div_indexes = []
        div_chars = []
        error_msg = ''
        for i, char in enumerate(text):
            if char not in self.end_of_sentence_marks:
                continue
            # : (except when it has numbers on both sides)
            if (char == ':') and (i > 0) and (i < (len(text)-1)) and \
                    (text[i-1] in self.digits_str) and (text[i+1] in self.digits_str):
                continue
            div_indexes.append(i)
            div_chars.append(char)
        div_indexes.append(len(text))
        sent_list = []
        start_index = -1
        for end_index in div_indexes:
            sent_list.append(text[start_index+1:end_index+1])
            start_index = end_index
        if not div_indexes:
            error_msg = 'no punctuation'
        return [x.strip() for x in sent_list if x.strip()], error_msg

    @staticmethod
    def _check_quotes(text: str) -> (str, bool):
        """
        Check quotes: even number or ' and " quotes.
        Must be no nested quotes.
        Ignore apostrophes: 'll n't X's s'
        Replace single quotes with "

        :return: New text, True if error
        """
        quotes_open = ''
        regex_1 = re.compile(r"[\w]'s", re.I)
        regex_2 = re.compile(r"s'[\s]", re.I)
        new_text = []
        for i, char in enumerate(text):
            new_text.append(char)
            if char == "'":  # ignore apostrophes
                if (text[i - 1:i + 2] == "n't") or (text[i + 1:i + 3] == "ll") or \
                        regex_1.search(text[i - 1:i + 2]) or regex_2.search(text[i - 1:i + 2]):
                    continue
            if char in ('"', "'"):
                if char == "'":
                    new_text[-1] = '"'
                if quotes_open:
                    if quotes_open != char:
                        return '', True
                    else:
                        quotes_open = ''
                else:
                    quotes_open = char
                continue
        return ''.join(new_text), bool(quotes_open)  # must be closed

    def _xy_sentence_breaking(self, sent_list: list[str]) -> list[str]:
        """
        X/Y Sentence Breaking:
          if sentence is > BREAK_ON_X, chop it by B_Sentence_size_Y number of words
        """
        new_sent_list = []
        for sentence in sent_list:
            word_list = sentence.split(' ')
            if len(word_list) <= self.BREAK_ON_X:
                new_sent_list.append(sentence)
            else:
                while word_list:
                    if len(word_list) > self.BREAK_ON_X:
                        new_sent_list.append(' '.join(word_list[:self.B_Sentence_size_Y]))
                        word_list = word_list[self.B_Sentence_size_Y:]
                    else:
                        new_sent_list.append(' '.join(word_list))
                        break
        return new_sent_list

    def text2sentences(self):
        sentences = []
        sent_number = 0
        for paragraph in self.par_list:
            paragraph = self._find_convert_special_tokens(paragraph)
            # transform double quotes to "
            new_par = re.sub(r"['’]{2}", '"', paragraph)
            new_par = re.sub(r'[“”¨]', '"', new_par)
            # transform single quotes to '
            new_par = re.sub(r"[`'’]", "'", new_par)
            # add . after ."
            new_par = re.sub(r'\.[ ]*\"[ ]*', '.". ', new_par)
            new_par = new_par.replace('. .', '.')
            # check for even number of double quotes
            new_par, quotes_failed = self._check_quotes(new_par)
            if quotes_failed:
                self.sent_2d_list = []
                self.sent_number = 0
                self.quotes_failed = True
                return
            # put dot before quoted text
            new_par = re.sub(r'"', '. "', new_par)
            if self.delete_quoted:
                # delete sentences in double quotes
                new_par = re.sub(r'("[^"]+?")', "", new_par)

            # delete dots, commas, spaces remaining at the start of the paragraph
            new_par = re.sub(r'^[ .,]+', '', new_par)
            # delete remaining commas and spaces before the dot
            new_par = re.sub(r'[, ]+\.', '.', new_par)
            # add spaces after the dot if necessary
            new_par = re.sub(r'\.(?=[^ ])', '. ', new_par)
            # fix big spacings
            new_par = re.sub(r'[\s]+', ' ', new_par)
            # fix comma spacings
            new_par = re.sub(r',(?=[^ ])', ', ', new_par)
            new_par = new_par.replace(' ,', ',')
            # also fix double commas
            new_par = re.sub(r',{2,}', ',', new_par)
            sent_list, error_msg = self._sent_tokenize(new_par)
            if error_msg == 'no punctuation':
                self.no_punctuation = True
            sent_list = [x.strip() for x in sent_list if re.findall(r'[a-zA-Z]', x.strip())]
            sent_list = self._xy_sentence_breaking(sent_list)
            for i in range(len(sent_list)):
                sent_list[i] = sent_list[i].strip('"').strip()
            sentences.append(sent_list)
            sent_number += len(sent_list)
        self.sent_2d_list = sentences
        self.sent_number = sent_number

    @staticmethod
    def _sent2words_tokenize_v2(sentence):
        """Tokenize current sentence to words"""
        # transform double quotes to "
        new_sent = re.sub(r"['’]{2}", '"', sentence)
        new_sent = re.sub(r'[“”]', '"', new_sent)
        # transform single quotes to '
        new_sent = re.sub(r"[`'’]", "'", new_sent)
        # get word list and use only words that contain alpha chars and/or numbers
        cur_words_list = [x for x in word_tokenize(new_sent) if re.search(r"[a-zA-Z0-9]", x)]
        return cur_words_list

    def sentences2words(self):
        """Convert 2d list of sentences (divided by paragraphs) to 3d list of words"""
        self.words_3d_list = []
        self.words_number = 0
        for par_index in range(len(self.sent_2d_list)):
            # new list of sentences of this paragraph
            self.words_3d_list.append([])
            for sent_index in range(len(self.sent_2d_list[par_index])):
                sentence = self.sent_2d_list[par_index][sent_index]
                # tokenize sentence to words
                cur_words_list = self._sent2words_tokenize_v2(sentence)
                self.words_3d_list[par_index].append(cur_words_list)
                # count words in current sentence
                self.words_number += len(cur_words_list)


class SETask22SALTClass:
    def __init__(self, se_salt: str, stop_words: set):
        self.salt = se_salt
        self.S = int(se_salt[-4])
        self.A = int(se_salt[-3])
        self.L = int(se_salt[-2])
        self.T = int(se_salt[-1])

        # default stop list, spell checker, lemmatizer, stemmer
        self.stop_words = stop_words
        self.stop_words_info = 'Not used'
        self.spell = lambda x: x
        self.spellchecker_info = 'No spell checking'
        self.lemmatizer = None
        self.lemma = lambda x: x
        self.lemma_info = "Not used"
        self.stemmer = None
        self.stem = lambda x: x
        self.stem_info = "Not used"
        # init stop list, spell checker, lemmatizer, stemmer using 'SALT' settings
        self.init_salt()
        contractions_filename = os.path.join(WORK_DIR, 'data', 'contractions.json')
        with open(contractions_filename, 'r', encoding='utf-8') as f:
            contractions = f.read()
        self.resp_xforms_dict = json.loads(contractions)
        #####  self.resps_xforms_dict = RespXFormsTable(sql_obj).get_resps_xforms_dict()

    def init_salt(self):
        """Initialize stop list, spellchecking, lemmatization, stemming using SALT parameter"""
        # initialize stop list
        if self.S:
            if not self.stop_words:
                log('warning', 'No stop-list words found')
                self.stop_words_info = "Empty"
            else:
                self.stop_words_info = "Yes (" + str(len(self.stop_words)) + ' words)'
        # initialize spellchecking
        if self.A:
            from autocorrect import Speller
            spell = Speller()
            # use spellchecking if the word is bigger than A (in SALT) parameter
            self.spell = lambda x: spell(x) if (len(x) > self.A) else x
            self.spellchecker_info = "autocorrect.spell"
            if self.A > 1:
                self.spellchecker_info += ' (if length>' + str(self.A) + ')'
        # initialize lemmatization
        if self.L:
            if self.L == 1:
                from nltk.stem import WordNetLemmatizer
                self.lemmatizer = WordNetLemmatizer()
                self.lemma = self.lemmatizer.lemmatize
                # additional initializing (on the first lemmatized word)
                self.lemma('mice')
                self.lemma_info = "WordNet lemmatizer"
            else:
                log('warning', 'Wrong L digit: using no lemmatizer.')
        # initialize stemming
        if self.T:
            if self.T == 1:
                from nltk.stem.porter import PorterStemmer
                self.stemmer = PorterStemmer()
                self.stem = self.stemmer.stem
                self.stem_info = "PorterStemmer"
            elif self.T == 2:
                from nltk.stem.lancaster import LancasterStemmer
                self.stemmer = LancasterStemmer()
                self.stem = self.stemmer.stem
                self.stem_info = "LancasterStemmer"
            elif self.T == 3:
                from nltk.stem import SnowballStemmer
                self.stemmer = SnowballStemmer('english')
                self.stem = self.stemmer.stem
                self.stem_info = "SnowballStemmer('english')"
            else:
                log('warning', 'Wrong T digit: using no stemmer')

    def process_word(self, word: str):
        """If the word not in the stop list, then do spell checking, lemmatizing, stemming on this word

        :param word: str
        :return: str or False if the word is in the stop list
        """
        return self.stem(self.lemma(self.spell(word))) if word not in self.stop_words else False

    def apply_resps_xforms(self, sentence):
        """Apply resps_xforms_dict to the sentence.

        :param sentence: str
        :return: str. New sentence.
        """
        new_sentence = sentence
        for in_txt in self.resps_xforms_dict:
            # replace in_txt text with out_txt text
            new_sentence = new_sentence.replace(in_txt, self.resps_xforms_dict[in_txt])
        return new_sentence

    def tokenize_sentence(self, sentence):
        """Tokenize current sentence to words"""
        # transform double quotes to "
        new_sent = re.sub(r"['’]{2}", '"', sentence)
        new_sent = re.sub(r'[“”]', '"', new_sent)
        # transform single quotes to '
        new_sent = re.sub(r"[`'’]", "'", new_sent)
        # apply resps_xforms dictionary
        new_sent = self.apply_resps_xforms(new_sent)
        # get word list and use only words that contain only alpha chars
        cur_words_list = [x for x in word_tokenize(new_sent) if re.match(r"[a-zA-Z]+", x)]
        return cur_words_list

    def process_sentence(self, sentence):
        """Tokenize sentence and return the list of processed word-tokens"""
        result_tokens = []
        for word in self.tokenize_sentence(sentence.lower()):
            token = self.process_word(word)
            if token:
                result_tokens.append(token)
        return result_tokens


class SETask05SALTClass(SETask22SALTClass):
    def process_sentence_pre_expanding(self, sentence):
        """Pre expanding: tokenize sentence, delete stop words, return sentence without stop words"""
        result_tokens = []
        for word in self.tokenize_sentence(sentence.lower()):
            token = word if word not in self.stop_words else False
            # if word not in stop list and consists of only alpha chars, add it to sentence
            if token and re.match(r'^[a-z]+$', token):
                result_tokens.append(token)
        return ' '.join(result_tokens)

