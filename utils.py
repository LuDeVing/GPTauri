import hashlib
import mimetypes
import os
import re
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Optional, Tuple, List, SupportsInt
from typing import Union, Any

import orjson
import pandas as pd
from unidecode import unidecode

ENCODING = {
    'utf': 'utf-8-sig',
    'latin': 'iso-8859-1'
}


class Files:
    @staticmethod
    def df_to_csv(df, path, append=False, col=None, encoding='utf'):
        mode, header = 'w', True
        if append:
            mode, header = 'a', not os.path.exists(path)

        if not isinstance(df, pd.DataFrame):
            if not isinstance(df, list):
                df = [df]

            df = pd.DataFrame(data=df, columns=col)

        df.to_csv(path, mode=mode, header=header, index=False, encoding=ENCODING[encoding])

    @staticmethod
    def read_file(path, add_format=None, encoding='utf', ensure_lines=False):
        if add_format not in ['df', 'soup', 'lxml', None]:
            raise Exception('Format Not Supported')

        if path.endswith('csv'):
            return pd.read_csv(path, encoding=ENCODING[encoding], header=0, low_memory=False, on_bad_lines='skip')

        with open(path, mode='r', encoding=ENCODING[encoding]) as f:
            if path.endswith('.jsonl') or ensure_lines:
                try:
                    data = [orjson.loads(line) for line in f.readlines()]
                except orjson.JSONDecodeError:
                    return []
            elif path.endswith('.json'):
                try:
                    data = orjson.loads(f.read())
                except orjson.JSONDecodeError:
                    return {}
            else:
                data = f.read()

        if add_format == 'df':
            data = pd.DataFrame(data)

        return data

    @staticmethod
    def save_json(dict_data, path, lines=False):
        with open(path, 'wb') as f:
            if not lines:
                f.write(orjson.dumps(dict_data, option=orjson.OPT_INDENT_2))
            else:
                for line in dict_data:
                    f.write(orjson.dumps(line))
                    f.write(b'\n')


class StrComp:
    MATCH_RATIOS = {
        3: 0.31,
        2: 0.51,
        1: 0.71,
    }
    EXCLUDE_WORDS = {'and', 'the', 'with', 'live', 'in', 'at', 'of', 'to', 'on', 'onto', 'for', 'by', 'from', 'a',
                     'as', 'be', 'an', 'it', 'will', 'are', 'this', 'that', 'is', 'but', 'or', 'nor', 'so', 'if',
                     'up', 'out', 'into'}

    def __init__(self, a: str, b: str, exclude_more: Optional[list] = None,
                 match_ratios: Optional[list] = None, prt: Optional[bool] = False) -> None:

        self.prt = prt

        self.EXCLUDE_WORDS.update(exclude_more or {})
        self.MATCH_RATIOS = self.MATCH_RATIOS or match_ratios

        self.a_original = a
        self.b_original = b

        self.a_split, self.b_split = self.process_words(a), self.process_words(b)
        self.a_join, self.b_join = ' '.join(self.a_split), ' '.join(self.b_split)

    def process_words(self, value: str) -> list:
        split_value = value.split(' ')

        sorted_value = sorted(set(split_value).difference(self.EXCLUDE_WORDS))
        sorted_value = [word for word in sorted_value if len(word) > 1 or word.isnumeric()]

        if ''.join(sorted_value).replace(' ', '') == '':
            return split_value
        else:
            return sorted_value

    def ratcliff_obershelp(self) -> float:
        return SequenceMatcher(None, self.a_join, self.b_join).ratio()

    def levenshtein_distance(self) -> float:
        used = {}

        @lru_cache(maxsize=1024)
        def min_dist(s1: int, s2: int) -> int:

            if s1 == len(self.a_join) or s2 == len(self.b_join):
                return len(self.a_join) - s1 + len(self.b_join) - s2

            if self.a_join[s1] == self.b_join[s2]:
                return used.setdefault((s1 + 1, s2 + 1), min_dist(s1 + 1, s2 + 1))

            next_indices = [
                (s1, s2 + 1),
                (s1 + 1, s2),
                (s1 + 1, s2 + 1)
            ]

            next_values = []

            for next_idx in next_indices:
                next_values.append(used.setdefault(next_idx, min_dist(*next_idx)))

            return 1 + min(next_values)

        distance = min_dist(0, 0)
        max_len = max(len(self.a_join), len(self.b_join))
        ratio = (1 - distance / max_len) if max_len != 0 else 1

        return ratio

    def words_comp(self) -> float:
        smaller_words = len(self.a_split) if len(self.a_split) < len(self.b_split) else len(self.b_split)
        similar_words = len(set(self.a_split) & set(self.b_split))

        return min(similar_words / smaller_words, 1) - 0.01

    def only_city_matches(self):

        a_s = self.a_original.split(' - ')
        b_s = self.b_original.split(' - ')

        city_a = a_s[0]
        city_b = b_s[0]

        common_words = set(a_s[1].split(' ')) & set(b_s[1].split(' '))
        return len(common_words) == 0 and city_a == city_b

    def compare_words(self):

        words_count = min(max(self.MATCH_RATIOS), self.a_join.count(' ') + 1, self.b_join.count(' ') + 1)
        base_ratio = self.MATCH_RATIOS[words_count]

        if (' - ' in self.a_original or ' - ' in self.b_original) and self.only_city_matches():
            base_ratio = self.MATCH_RATIOS[1]

        if self.a_join == self.b_join:
            return 1

        match_ratio = max(self.words_comp(), self.ratcliff_obershelp())

        if self.prt:
            print(self.a_join)
            print(self.b_join)

     #   print(self.a_join, self.b_join)

        if self.prt:
            print(match_ratio)

        if match_ratio < base_ratio:
            return None

        return match_ratio

    @staticmethod
    def strip_all_spaces(str_val: str) -> str:
        str_val = str_val.replace('\n', ' ')
        no_mid_spaces = ' '.join(re.split(r'\s+', str_val, flags=re.UNICODE))
        return no_mid_spaces.strip()

    @staticmethod
    def fix_unicode(str_val: str) -> str:
        decoded_str = unidecode(str_val)
        if decoded_str == '':
            return str_val
        else:
            return decoded_str

    @staticmethod
    def convert_tiny_chars(str_val: str) -> str:
        return ''.join(chr(ord(c) - 0x1F1E6 + ord('A')) if '🇦' <= c <= '🇿' else c for c in str_val)

    @staticmethod
    def clean_chars(str_val: str, replace_with: str = ' ') -> str:
        return re.sub(r'[\W_]', replace_with, str_val).replace('_', replace_with)

    @staticmethod
    def clean_control_chars(str_val: str, replace_with: str = ' ') -> str:
        return re.sub(r'[\x00-\x1F\x7F]', replace_with, str_val)

    @staticmethod
    def clean_escape_chars(str_val: str):
        escape_chars = {
            "\\'": "'",
            '\\"': '"',
            '\\\\': '\\',
            '\\/': '/',
            '\\b': '',
            '\\f': ' ',
            '\\n': ' ',
            '\\r': ' ',
            '\\t': ' '
        }

        for key, val in escape_chars.items():
            str_val = str_val.replace(key, val)

        return str_val

    @staticmethod
    def clean_regex_chars(str_val: str) -> str:
        for val in ['+', '*', '?', '^', '$', '(', ')', '[', ']', '{', '}', '|', '\\']:
            str_val = str_val.replace(val, '')

        return str_val

    @staticmethod
    def clean_tags(str_val: str) -> str:
        return re.sub(re.compile(r'<.*?>', re.DOTALL), '', str_val)

    @staticmethod
    def clean_join_commas(str_val: str, join_by: str = ', ') -> str:
        return join_by.join([part.strip() for part in str_val.split(',') if part.strip()])

    @staticmethod
    def fill_number(num: Union[int, str], digits: int = 2) -> str:
        try:
            return f"{int(num):0{digits}d}"
        except ValueError:
            return num

    @staticmethod
    def get_numbers_len(txt: str, get_numbers: bool = False) -> Union[str, int]:
        all_numbers = re.findall(r'[0-9]+', str(txt))
        numbers_str = ''.join(all_numbers)

        if get_numbers:
            return numbers_str

        return len(numbers_str)

    @staticmethod
    def british_to_american(text):

        brit_to_amer = {
            "colour": "color",
            "favourite": "favorite",
            "organise": "organize",
            "realise": "realize",
            "defence": "defense",
            "travelling": "traveling",
            "centre": "center",
            "theatre": "theater",
            "licence": "license",
            "aluminium": "aluminum"
        }

        words = text.split()
        converted_words = [brit_to_amer.get(word, word) for word in words]
        return ' '.join(converted_words)
