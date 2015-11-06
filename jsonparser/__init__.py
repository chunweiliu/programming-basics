"""
JSON Parser

Implement a JSON parser for the following data type.

JSON        Python
----------------------
array       list
object      dict
----------------------
"""
import re
import unittest


class JSONParser(object):
    def __init__(self):
        self.tokens = []

    def parse(self, string):
        self.tokens = re.findall(r'[-]?\d*\.?\d+|[\[\]{}]|\w+', string)
        self.current_token = self.tokens[0] if self.tokens else None

        # Assume the input string is valid.
        if self.current_token == '[':
            return self._parse_array()
        if self.current_token == '{':
            return self._parse_object()

    def _advance(self):
        self.tokens.pop(0)
        self.current_token = self.tokens[0] if self.tokens else None

    def _parse_array(self):
        self._advance()  # advance from [

        array = []
        while self.current_token and self.current_token != ']':
            if self.current_token == '[':
                array.append(self._parse_array())
            elif self.current_token == '{':
                array.append(self._parse_object())
            else:
                array.append(float(self.current_token))
                self._advance()

        self._advance()  # advance from ]
        return array

    def _parse_object(self):
        self._advance()  # advance from {

        json_object = {}
        while self.current_token and self.current_token != '}':
            key, value = self.current_token, self.tokens[1]

            # Need to pop the key first, so the rest follows the pattern.
            self._advance()

            if value == '[':
                json_object[key] = self._parse_array()
            elif value == '{':
                json_object[key] = self._parse_object()
            else:
                json_object[key] = float(value)
                self._advance()

        self._advance()  # advance from }
        return json_object


class TestJSONParser(unittest.TestCase):
    def test_array(self):
        string = '[-1, .1, 1, 2, 2.5, 10]'
        expected = [-1, .1, 1, 2, 2.5, 10]

        json_parser = JSONParser()
        self.assertEqual(json_parser.parse(string), expected)

    def test_array_of_array(self):
        string = '[[-1, .1, 1], [2, 2.5, 10]]'
        expected = [[-1, .1, 1], [2, 2.5, 10]]

        json_parser = JSONParser()
        self.assertEqual(json_parser.parse(string), expected)

    def test_object(self):
        string = '{\'key1\': 10, \'key2\': 20}'
        expected = {'key1': 10, 'key2': 20}

        json_parser = JSONParser()
        self.assertEqual(json_parser.parse(string), expected)

    def test_object_of_object(self):
        string = '{\'key1\': {\'key1key\': 10}, \'key2\': [20, 21, 22]}'
        expected = {'key1': {'key1key': 10}, 'key2': [20, 21, 22]}

        json_parser = JSONParser()
        self.assertEqual(json_parser.parse(string), expected)


if __name__ == '__main__':
    unittest.main()
