import re
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
import unicodedata

class TweetPreprocessor(object):

    def __init__(self):
        self.FLAGS = re.MULTILINE | re.DOTALL
        self.ALLCAPS = ''#''<allcaps>'
        self.HASHTAG = ''

        self.URL = ''#''<url>'
        self.USER = ''#'<user>'
        self.SMILE = ''#'<smile>'
        self.LOLFACE =''# '<lolface>'
        self.SADFACE = ''#'<sadface>'
        self.NEUTRALFACE =''# '<neutralface>'
        self.HEART = ''#'<heart>'
        self.NUMBER = ''#''<number>'
        self.REPEAT = ''#'<repeat>'
        self.ELONG = ''#'<elong>'

    def remove_accents(self,input_str):
        nfkd_form = unicodedata.normalize('NFKD', input_str)
        only_ascii = nfkd_form.encode('ASCII', 'ignore')
        return only_ascii

    def _hashtag(self, text):
        text = text.group()
        hashtag_body = text[1:]
        if hashtag_body.isupper():
            #print("isnot")
            result = (self.HASHTAG + " {} " + self.ALLCAPS).format(hashtag_body)
        else:
           # print("error")
            #result = " ".join([self.HASHTAG] + re.split(r"(?=[A-Z])", hashtag_body, flags=self.FLAGS))
            result = " ".join([self.HASHTAG] + re.split("#", hashtag_body, flags=self.FLAGS))
        return result

    def _calculate_languages_ratios(self, text):
        """
        Calculate probability of given text to be written in several languages and
        return a dictionary that looks like {'french': 2, 'spanish': 4, 'english': 0}

        @param text: Text whose language want to be detected
        @type text: str

        @return: Dictionary with languages and unique stopwords seen in analyzed text
        @rtype: dict
        """

        languages_ratios = {}

        tokens = wordpunct_tokenize(text)
        words = [word.lower() for word in tokens]

        # Compute per language included in nltk number of unique stopwords appearing in analyzed text
        for language in stopwords.fileids():
            stopwords_set = set(stopwords.words(language))
            words_set = set(words)
            common_elements = words_set.intersection(stopwords_set)

            languages_ratios[language] = len(common_elements)  # language "score"

        return languages_ratios


    # ----------------------------------------------------------------------
    def _detect_language(self, text):
        """
        Calculate probability of given text to be written in several languages and
        return the highest scored.

        It uses a stopwords based approach, counting how many unique stopwords
        are seen in analyzed text.

        @param text: Text whose language want to be detected
        @type text: str

        @return: Most scored language guessed
        @rtype: str
        """

        ratios = self._calculate_languages_ratios(text)

        most_rated_language = max(ratios, key=ratios.get)

        return most_rated_language

    def isfloat(self,value):
        try:
            float(value)
            return False
        except ValueError:
            return True

    def _allcaps(self, text):
        text = text.group()
        return text.lower() + ' ' + self.ALLCAPS

    def preprocess(self, text, type):

       if isinstance(text, str):
            if(len(text)>0) and (self.isfloat(text)):


                eyes, nose = r"[8:=;]", r"['`\-]?"

                re_sub = lambda pattern, repl: re.sub(pattern, repl, text, flags=self.FLAGS)
                text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", self.URL)

                text = re_sub(r"/"," / ")
                text = re_sub(r"@\w+", self.USER)
                text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), self.SMILE)
                text = re_sub(r"{}{}p+".format(eyes, nose), self.LOLFACE)
                text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), self.SADFACE)
                text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), self.NEUTRALFACE)
                text = re_sub(r"<3", self.HEART)
                text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", self.NUMBER)

                text = re_sub(r"([!?.]){2,}", r"\1 " + self.REPEAT)
                text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 " + self.ELONG)

                text = re_sub(r"([A-Z]){2,}", self._allcaps)

                text = re.sub(r"[^\w\d\s]+", '', text)
                text = re.sub(r'[^a-zA-Z0-9]', r' ', text)



                text = ' '.join([word for word in text.split() if not word.startswith('rt')])
                text = text.lower()
                text = ' '.join([word.strip("_") for word in text.split()])




                return text.lower()


            else:

                text = ""
                return ""


