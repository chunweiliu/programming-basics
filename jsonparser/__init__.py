import re


class JSONParser(object):
    def __init__(self):
        self.tokens = []  # Use global variable for parsing.

    def parse(self, text):
        self.tokens = re.findall('[\d.\d]+|[\w]+|[\[\]\{\}]', text)
        if self.tokens[0] == '[':
            return self._parse_array(self.tokens)
        if self.tokens[0] == '{':
            return self._parse_object(self.tokens)
        return None

    def _parse_array(self, tokens):
        tokens.pop(0)  # Pop the '['

        array = []
        while tokens[0] and tokens[0] != ']':
            if tokens[0] == '[':
                array.append(self._parse_array(tokens))
            if tokens[0] == '{':
                array.append(self._parse_object(tokens))

            array.append(tokens[0])
            tokens.pop(0)

        tokens.pop(0)  # Pop the ']'
        return array

    def _parse_object(self, tokens):
        tokens.pop(0)  # Pop the '{'

        json_object = {}
        while tokens[0] and tokens[0] != '}':
            key = tokens[0]
            tokens.pop(0)

            if tokens[0] in ['[', '{']:
                json_object[key] = self._parse_array(tokens) \
                    if tokens[0] == '[' else self._parse_object(tokens)
            else:
                json_object[key] = tokens[0]
                tokens.pop(0)

        tokens.pop(0)  # Pop the '}'
        return json_object

if __name__ == '__main__':
    json_parser = JSONParser()

    print "First Step"
    for value in json_parser.parse(" [ 10, 20, 30.1 ] "):
        print value

    print "\nSecond Step"
    for value in json_parser.parse(" [ 10 , 20, \"hello\", 30.1 ] "):
        print value

    print "\nThird Step"
    for key, value in json_parser.parse("""{
            "hello": "world",
            "key1": 20,
            "key2": 20.3,
            "foo": "bar" }""").items():
        print key, value

    print "\nFourth Step"
    for key, value in json_parser.parse("""{
            "hello": "world",
            "key1": 20,
            "key2": 20.3,
            "foo": {
                "hello1": "world1",
                "key3": [200, 300]
            } }""").items():
        if isinstance(value, dict):
            for key2, value2 in value.items():
                print key2, value2
        else:
            print key, value
