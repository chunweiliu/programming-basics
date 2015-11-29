import re
import unittest


class SpeardSheet(object):
    def __init__(self):
        self.cells = {}

    def set_cell(self, location, content):
        self.cells[location] = content

    def get_cell(self, location):
        if location not in self.cells:
            return ''
        expression = self.cells[location]
        if expression[0] == '=':
            return self._eval(expression[1:])
        return float(expression) if expression.isdigit() else expression

    def _eval(self, expression):
        tokens = re.findall('\w*\d+|[+*-/]', expression)
        for i, _ in enumerate(tokens):
            while tokens[i][0].isalpha():
                tokens[i] = self.cells[tokens[i]]
            if tokens[i][0] == '=':
                tokens[i] = str(self._eval(tokens[i][1:]))
        return eval(' '.join(tokens))


class TestSpeardSheet(unittest.TestCase):
    def test_read_empty(self):
        speard_sheet = SpeardSheet()
        self.assertEqual('', speard_sheet.get_cell('A1'))

    def test_read_content(self):
        speard_sheet = SpeardSheet()
        speard_sheet.set_cell('A1', '100')
        self.assertEqual(100, speard_sheet.get_cell('A1'))

    def test_read_remote_content(self):
        speard_sheet = SpeardSheet()
        speard_sheet.set_cell('A1', '100')
        speard_sheet.set_cell('A2', '=A1')
        self.assertEqual(100, speard_sheet.get_cell('A2'))

    def test_read_remote_content2(self):
        speard_sheet = SpeardSheet()
        speard_sheet.set_cell('A1', '100')
        speard_sheet.set_cell('A2', '200')
        speard_sheet.set_cell('A3', '=A1 + A2')
        self.assertEqual(300, speard_sheet.get_cell('A3'))

    def test_read_remote_content3(self):
        speard_sheet = SpeardSheet()
        speard_sheet.set_cell('A1', '100')
        speard_sheet.set_cell('A2', '=A1 + 100')
        speard_sheet.set_cell('A3', '=A1 + A2')
        self.assertEqual(300, speard_sheet.get_cell('A3'))


if __name__ == '__main__':
    unittest.main()
