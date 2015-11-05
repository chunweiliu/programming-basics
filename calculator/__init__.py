"""
Calculator

<expr> ::= <expr> + <term>
         | <expr> - <term>
         | <term>

<term> ::= <term> * <factor>
         | <term> / <factor>
         | <factor>

<factor> ::= [\d]+
           | (<expr>)

"""
import re
import unittest


class Calculator(object):
    def __init__(self):
        self.tokens = []
        self.current_token = None

    def evaluate(self, expression):
        # Regular expression is a key. To handle negative and multiple digits,
        # use '[-]?\d*\.*\d+'.
        self.tokens = re.findall('[-]?\d*\.*\d+|[()+*-/]', expression)
        self.current_token = self.tokens[0] if self.tokens else None
        return self._expr()

    def _advance(self):
        self.tokens = self.tokens[1:]
        self.current_token = self.tokens[0] if self.tokens else None

    def _expr(self):
        result = self._term()  # Process * and / first.
        while self.current_token in ['+', '-']:
            if self.current_token == '+':
                self._advance()  # Advance the operator.
                result += self._term()
            else:
                self._advance()
                result -= self._term()
        return result

    def _term(self):
        result = self._factor()
        while self.current_token in ['*', '/']:
            if self.current_token == '*':
                self._advance()
                result *= self._factor()
            else:
                self._advance()
                result /= self._factor()
        return result

    def _factor(self):
        if self.current_token == '(':
            self._advance()
            result = self._expr()
            self._advance()
            return result

        result = float(self.current_token)
        self._advance()  # Advance the number.
        return result


class TestCalculator(unittest.TestCase):
    def test_zero(self):
        expression = '0'
        calculator = Calculator()
        self.assertEqual(eval(expression), calculator.evaluate(expression))

    def test_plus(self):
        expression = '1 + 1'
        calculator = Calculator()
        self.assertEqual(eval(expression), calculator.evaluate(expression))

    def test_order(self):
        expression = '1 + 2 * 3'
        calculator = Calculator()
        self.assertEqual(eval(expression), calculator.evaluate(expression))

    def test_parenthesis(self):
        expression = '2 * (2 + 3)'
        calculator = Calculator()
        self.assertEqual(eval(expression), calculator.evaluate(expression))

    def test_float(self):
        expression = '2.2 * (2 + 3)'
        calculator = Calculator()
        self.assertEqual(eval(expression), calculator.evaluate(expression))

    def test_two_digits(self):
        expression = '22 * (2 + 3)'
        calculator = Calculator()
        self.assertEqual(eval(expression), calculator.evaluate(expression))

    def test_negative_digit(self):
        expression = '-2 * (2 + 3)'
        calculator = Calculator()
        self.assertEqual(eval(expression), calculator.evaluate(expression))


if __name__ == '__main__':
    unittest.main()
