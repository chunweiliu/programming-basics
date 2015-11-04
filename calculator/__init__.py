"""
An implementation a simple calculator using a BNF parser:

<expr> ::= <expr> + <term>
       |   <expr> - <term>
       |   <term>

<term> ::= <factor> * <term>
       |   <factor> / <term>
       |   <factor>

<factor> ::= 0|1|2|3|4|5|6|7|8|9
         |   (<expr>)

"""
import re


class Calculator():
    def __init__(self):
        self.tokens = []
        self.current_token = None

    def evaluate(self, expression):
        # Regular expression for matching numbers (-2, 10, 1.5) and operator.
        self.tokens = re.findall('[-]?[\d.]+|[()+*-/]', expression)
        self.current_token = self.tokens[0] if self.tokens else None
        return self.expr()

    def expr(self):
        result = self.term()
        # Re-evaluate the current token utill not hitting the '+' or '-'.
        # E.g 1 + 2 + 3
        #           ^ current_token is here after parsing the first two.
        while self.current_token in ['+', '-']:
            if self.current_token == '+':
                self.advance()
                result += self.term()
            if self.current_token == '-':
                self.advance()
                result -= self.term()
        return result

    def term(self):
        result = self.factor()
        while self.current_token in ['*', '/']:
            if self.current_token == '*':
                self.advance()
                result *= self.factor()
            if self.current_token == '/':
                self.advance()
                result /= self.factor()
        return result

    def factor(self):
        if self.current_token[0] == '(':
            self.advance()
            result = self.expr()
            self.advance()
            return result

        result = float(self.current_token)
        self.advance()
        return result

    def advance(self):
        self.tokens = self.tokens[1:]
        self.current_token = self.tokens[0] if self.tokens else None


if __name__ == '__main__':
    calculator = Calculator()
    expression = '(1 + 2) + 3'
    print calculator.evaluate(expression)
