import re
import unittest


class Cell(object):
    def __init__(self, cell_string):
        self.string = cell_string
        self.dependences = []


class Excel(object):
    def __init__(self):
        self.data = {}

    def set_cell(self, cell_name, cell_string):
        self.data[cell_name] = cell_string

    def get_cell(self, cell_name):
        if cell_name not in self.data:
            return None

        cell_string = self.data[cell_name]

        if cell_string[0] == '=':
            return self._evaluate(cell_string[1:])
        elif cell_string.isdigit():
            return float(cell_string)
        else:
            return cell_string

    def _evaluate(self, cell_string):
        tokens = re.findall('\w*\d+|[+*-/]', cell_string)

        for i, _ in enumerate(tokens):
            while tokens[i][0].isalpha():
                tokens[i] = self.data[tokens[i]]
            if tokens[i][0] == '=':
                tokens[i] = str(self._evaluate(tokens[i][1:]))
        return eval(' '.join(tokens))


class TestExcel(unittest.TestCase):
    def test_empty_cell(self):
        excel = Excel()
        self.assertEqual(excel.get_cell('A1'), None)

    def test_string(self):
        excel = Excel()
        cell_string = 'a1'
        excel.set_cell('A1', cell_string)
        self.assertEqual(excel.get_cell('A1'), cell_string)

    def test_equation(self):
        excel = Excel()
        cell_string = '1 + 2 * 3'
        excel.set_cell('A1', '=' + cell_string)
        self.assertEqual(excel.get_cell('A1'), eval(cell_string))

    def test_evaluation(self):
        excel = Excel()
        excel.set_cell('A1', '1')
        excel.set_cell('A2', '2')

        excel.set_cell('A3', '=A1 + A2')
        self.assertEqual(excel.get_cell('A3'),
                         excel.get_cell('A1') + excel.get_cell('A2'))

        excel.set_cell('A2', '20')
        self.assertEqual(excel.get_cell('A3'),
                         excel.get_cell('A1') + excel.get_cell('A2'))

    def test_chain_evaluation(self):
        excel = Excel()
        excel.set_cell('A1', '1')
        excel.set_cell('A2', '=A1')

        excel.set_cell('A3', '=A1 + A2')
        self.assertEqual(excel.get_cell('A3'),
                         excel.get_cell('A1') + excel.get_cell('A2'))

    def test_chain_evaluation2(self):
        excel = Excel()
        excel.set_cell('A1', '1')
        excel.set_cell('A2', '=A1')

        excel.set_cell('A3', '=A1 + A2')
        self.assertEqual(excel.get_cell('A3'),
                         excel.get_cell('A1') + excel.get_cell('A2'))

        excel.set_cell('A4', '=A3')
        self.assertEqual(excel.get_cell('A4'), excel.get_cell('A3'))

        excel.set_cell('A3', '0')
        self.assertEqual(excel.get_cell('A4'), excel.get_cell('A3'))

        excel.set_cell('A4', '1')
        self.assertNotEqual(excel.get_cell('A4'), excel.get_cell('A3'))

if __name__ == '__main__':
    unittest.main()
