#IMPORTS

from errorArrows import *
import string
from rply import LexerGenerator

#CONSTANTS

DIGITS = '0123456789'
LETTERS = string.ascii_letters
LETTERS_DIGITS = LETTERS + DIGITS

#ERRORS

class Error:
    def __init__(self, start_pos, end_pos, name, details):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.name = name
        self.details = details
    
    def toString(self):
        result = f'{self.name}: {self.details}'
        result += f' File {self.start_pos.fn}, line {self.start_pos.ln + 1}'
        result += '\n\n' + errorArrows(self.start_pos.ftxt, self.start_pos, self.end_pos)
        return result

class IllegalCharError(Error):
    def __init__(self, start_pos, end_pos, details):
        super().__init__(start_pos, end_pos, 'Illegal Character', details)

class InvalidSyntaxError(Error):
    def __init__(self, start_pos, end_pos, details):
        super().__init__(start_pos, end_pos, 'Invalid Syntax', details)

class RuntimeError(Error):
    def __init__(self, start_pos, end_pos, details):
        super().__init__(start_pos, end_pos, 'Runtime error', details)

#POSITION

class Position:
    def __init__(self, index, ln, col, fn, ftxt):
        self.index = index
        self.ln = ln
        self.col = col
        self.fn = fn
        self.ftxt = ftxt
    
    def advance(self, currentChar = None):
        self.index += 1
        self.col += 1
        if currentChar == '\n':
            self.ln += 1
            self.col = 0
        return self
    
    def copy(self):
        return Position(self.index, self.ln, self.col, self.fn, self.ftxt)


#TOKENS

TT_NUM          = 'NUM'         #NUMBER TOKEN TYPE
TT_IDENTIFIER   = 'IDENTIFIER'  #IDENTIFIER TOKEN TYPE
TT_KEYWORD      = 'KEYWORD'     #VARIABLE NAME TOKEN TYPE
TT_PLUS         = 'PLUS'        #ADDITION TOKEN TYPE
TT_MINUS        = 'MINUS'       #SUBTRACTION TOKEN TYPE
TT_MUL          = 'MUL'         #MULTIPLICATION TOKEN TYPE
TT_DIV          = 'DIV'         #DIVISION TOKEN TYPE
TT_POW          = 'POW'         #POWER TOKEN TYPE
TT_EQ           = 'EQ'          #EQUALS TOKEN TYPE
TT_EE           = 'EE'          #DOUBLE EQUALS TOKEN TYPE
TT_NE           = 'NE'          #NOT EQUALS TOKEN TYPE
TT_GT           = 'GT'          #GREATER THAN TOKEN TYPE
TT_LT           = 'LT'          #LESS THAN TOKEN TYPE
TT_GTE          = 'GTE'         #GREATER THAN OR EQUALS TOKEN TYPE
TT_LTE          = 'LTE'         #LESS THAN OR EQUALS TOKEN TYPE
TT_LPAREN       = 'LPAREN'      #LEFT PARENTHESIS TOKEN TYPE
TT_RPAREN       = 'RPAREN'      #RIGHT PARENTHESIS TOKEN TYPE
TT_EOF          = 'EOF'         #END OF FILE TOKEN TYPE
TT_ERROR        = 'ERROR'       #ILLEGAL TEXT ERROR

KEYWORDS        = [
    'num',
    'and',
    'not',
    'or',
    'if',
    'then',
    'elif',
    'else'
]

#LEXER

class Lexer:
    def __init__(self, fn, text):
        self.fn = fn
        self.text = text
        self.lexer_generator = LexerGenerator()
        self.define_lexer_rules()
        self.builtLexer = self.lexer_generator.build()

    def define_lexer_rules(self):
        self.lexer_generator.ignore(r"\s+")
        self.lexer_generator.add(TT_KEYWORD, r"num|and|or|not|if|then|elif|else")
        self.lexer_generator.add(TT_NUM, r"(\d+(\.\d+)?)")
        self.lexer_generator.add(TT_PLUS, r"\+")
        self.lexer_generator.add(TT_MINUS, r"-")
        self.lexer_generator.add(TT_MUL, r"\*")
        self.lexer_generator.add(TT_DIV, r"/")
        self.lexer_generator.add(TT_POW, r"\^")
        self.lexer_generator.add(TT_EE, r"==")
        self.lexer_generator.add(TT_NE, r"!=")
        self.lexer_generator.add(TT_GTE, r">=")
        self.lexer_generator.add(TT_LTE, r"<=")
        self.lexer_generator.add(TT_GT, r">")
        self.lexer_generator.add(TT_LT, r"<")
        self.lexer_generator.add(TT_EQ, r"=")
        self.lexer_generator.add(TT_LPAREN, r"\(")
        self.lexer_generator.add(TT_RPAREN, r"\)")
        self.lexer_generator.add(TT_IDENTIFIER, r"[a-zA-Z_]+")
        self.lexer_generator.add(TT_ERROR, r".*\s")

    def get_token_list(self):
        initialTokens = self.builtLexer.lex(self.text)
        token = initialTokens.next()
        token_count = 1
        token_list = [token]
        while True:
            if token.gettokentype() == TT_ERROR:
                self.pos = Position(token.getsourcepos().idx, token.getsourcepos().lineno - 1, token.getsourcepos().colno - 1 , self.fn, self.text)
                self.end_pos = Position(token.getsourcepos().idx, token.getsourcepos().lineno - 1, token.getsourcepos().colno, self.fn, self.text)
                return [], None, IllegalCharError(self.pos, self.end_pos, "'" + token.getstr() + "'")
            try:
                token_list.append(initialTokens.next())
                token_count += 1
            except StopIteration:
                break
        return token_list, token_count, None

#NODES

class VarAccessNode:
    def __init__(self, varNameToken, fn, txt):
        self.varNameToken = varNameToken
        self.start_pos = Position(self.varNameToken.getsourcepos().idx, self.varNameToken.getsourcepos().lineno - 1, self.varNameToken.getsourcepos().colno - 1, fn, txt)
        self.end_pos = Position(self.varNameToken.getsourcepos().idx, self.varNameToken.getsourcepos().lineno - 1, self.varNameToken.getsourcepos().colno, fn, txt)

class VarAssignNode:
    def __init__(self, varNameToken, valueNode, fn, txt):
        self.varNameToken = varNameToken
        self.valueNode = valueNode
        self.start_pos = Position(self.varNameToken.getsourcepos().idx, self.varNameToken.getsourcepos().lineno - 1, self.varNameToken.getsourcepos().colno - 1, fn, txt)
        self.end_pos = valueNode.end_pos

class NumberNode:
    def __init__(self, token, fn, txt):
        self.token = token
        self.start_pos = Position(self.token.getsourcepos().idx, self.token.getsourcepos().lineno - 1, self.token.getsourcepos().colno - 1, fn, txt)
        self.end_pos = Position(self.token.getsourcepos().idx, self.token.getsourcepos().lineno - 1, self.token.getsourcepos().colno, fn, txt)
    def __repr__(self):
        return f'{self.token}'

class BinaryOperationNode:
    def __init__(self, leftNode, opToken, rightNode, fn, txt):
        self.leftNode = leftNode
        self.opToken = opToken
        self.rightNode = rightNode
        self.start_pos = leftNode.start_pos
        self.end_pos = rightNode.end_pos

    def __repr__(self):
        return f'({self.leftNode}, {self.opToken}, {self.rightNode})'

class UnaryOperationNode:
    def __init__(self, opToken, node, fn, txt):
        self.opToken = opToken
        self.node = node
        self.start_pos = Position(self.opToken.getsourcepos().idx, self.opToken.getsourcepos().lineno - 1, self.opToken.getsourcepos().colno - 1, fn, txt)
        self.end_pos = node.end_pos
    
    def __repr__(self):
        return f'({self.opToken}, {self.node})'

class IfNode:
    def __init__(self, cases, else_case):
        self.cases = cases
        self.else_case = else_case
        self.start_pos = self.cases[0][0].start_pos
        self.end_pos = (self.else_case or self.cases[-1][0]).end_pos

#PARSE RESULT

class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None
        self.advanceCount = 0
    
    def registerAdvance(self):
        self.advanceCount += 1

    def register(self, res):
        self.advanceCount += res.advanceCount
        if res.error: self.error = res.error
        return res.node

    def success(self, node):
        self.node = node
        return self

    def failure(self, error):
        if not self.error or self.advanceCount == 0:
            self.error = error
        return self

#PARSER

class Parser:
    def __init__(self, token_list, token_count, fn, txt):
        self.token_list = token_list
        self.token_count = token_count
        self.token_index = -1
        self.fn = fn
        self.txt = txt
        self.advance()

    def advance(self):
        self.token_index += 1
        if self.token_index < self.token_count:
            self.current_token = self.token_list[self.token_index]
            return
        return self.current_token

    def parse(self):
        res = self.expr()
        if not res.error and self.token_index != self.token_count:
            start_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno - 1, self.fn, self.txt)
            end_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno, self.fn, self.txt)
            return res.failure(InvalidSyntaxError(start_pos, end_pos, "Expected '+', '-', '*' or '/'"))
        return res

    def atom(self):
        res = ParseResult()
        token = self.current_token

        if token.gettokentype() == TT_NUM:
            res.registerAdvance()
            self.advance()
            return res.success(NumberNode(token, self.fn, self.txt))

        elif token.gettokentype() == TT_IDENTIFIER:
            res.registerAdvance()
            self.advance()
            return res.success(VarAccessNode(token, self.fn, self.txt))

        elif token.gettokentype() == TT_LPAREN:
            res.registerAdvance()
            self.advance()
            expr = res.register(self.expr())
            if res.error: return res
            if self.current_token.gettokentype() == TT_RPAREN:
                res.registerAdvance()
                self.advance()
                return res.success(expr)
            else:
                return res.failure(InvalidSyntaxError(
                    self.current_token.getsourcepos().colno - 1, self.current_token.getsourcepos().colno,
                    "Expected ')'"
                ))

        elif token.gettokentype() == TT_KEYWORD and token.getstr() == "if":
            if_expr = res.register(self.if_expr())
            if res.error: return res
            return res.success(if_expr)
        
        start_pos = Position(token.getsourcepos().idx, token.getsourcepos().lineno - 1, token.getsourcepos().colno - 1, self.fn, self.txt)
        end_pos = Position(token.getsourcepos().idx, token.getsourcepos().lineno - 1, token.getsourcepos().colno, self.fn, self.txt)
        return res.failure(InvalidSyntaxError(
            start_pos, end_pos, "Expected number, identifier, '+', '-' or '('"
        ))

    def power(self):
        return self.binaryOperation(self.atom, (TT_POW, ), self.factor)


    def factor(self):
        res = ParseResult()
        token = self.current_token

        if token.gettokentype() in (TT_PLUS, TT_MINUS):
            res.registerAdvance()
            self.advance()
            factor = res.register(self.factor())
            if res.error: return res
            return res.success(UnaryOperationNode(token, factor, self.fn, self.txt))

        return self.power()

    def term(self):
        return self.binaryOperation(self.factor, (TT_MUL, TT_DIV))

    def comp_expr(self):
        res = ParseResult()
        if self.current_token.gettokentype() == TT_KEYWORD and self.current_token.getstr() == 'not':
            op_token = self.current_token
            res.registerAdvance()
            self.advance()
            node = res.register(self.comp_expr())
            if res.error: return res
            return res.success(UnaryOperationNode(op_token, node, self.fn, self.txt))
        
        node = res.register(self.binaryOperation(self.arith_expr, (TT_EE, TT_NE, TT_LT, TT_GT, TT_LTE, TT_GTE)))
        
        if res.error:
            start_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno - 1, self.fn, self.txt)
            end_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno, self.fn, self.txt)
            return res.failure(InvalidSyntaxError(start_pos, end_pos, "Expected number, identifier, '+', '-', '(' or 'not'"))
        return res.success(node)

    def arith_expr(self):
        return self.binaryOperation(self.term, (TT_PLUS, TT_MINUS))
            
    def expr(self):
        res = ParseResult()
        if self.current_token.gettokentype() == TT_KEYWORD and self.current_token.getstr() == 'num':
            res.registerAdvance()
            self.advance()
            if self.current_token.gettokentype() != TT_IDENTIFIER:
                start_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno - 1, self.fn, self.txt)
                end_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno, self.fn, self.txt)
                return res.failure(InvalidSyntaxError(start_pos, end_pos, "Expected an identifier"))
            var_name = self.current_token
            res.registerAdvance()
            self.advance()
            if self.current_token.gettokentype() != TT_EQ:
                start_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno - 1, self.fn, self.txt)
                end_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno, self.fn, self.txt)
                return res.failure(InvalidSyntaxError(start_pos, end_pos, "Expected '='"))
            res.registerAdvance()
            self.advance()
            expr = res.register(self.expr())
            if res.error: return res
            return res.success(VarAssignNode(var_name, expr, self.fn, self.txt))

        node = res.register(self.binaryOperation(self.comp_expr, ((TT_KEYWORD, "and"), (TT_KEYWORD, "or"))))
        if res.error: 
            start_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno - 1, self.fn, self.txt)
            end_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno, self.fn, self.txt)
            return res.failure(InvalidSyntaxError(start_pos, end_pos, "Expected 'num', number, indentifier, '+', '-' or '('"))
        return res.success(node)
    
    def if_expr(self):
        res = ParseResult()
        cases = []
        else_case = None
        start_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno - 1, self.fn, self.txt)
        end_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno, self.fn, self.txt)
        if not (self.current_token.gettokentype() == TT_KEYWORD and self.current_token.getstr() == 'if'):
            return res.failure(InvalidSyntaxError(start_pos, end_pos,f"Expected 'if'"))
        res.registerAdvance()
        self.advance()

        condition = res.register(self.expr())
        if res.error: return res

        start_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno - 1, self.fn, self.txt)
        end_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno, self.fn, self.txt)
        if not (self.current_token.gettokentype() == TT_KEYWORD and self.current_token.getstr() == 'then'):
            return res.failure(InvalidSyntaxError(start_pos, end_pos, f"Expected 'then'"))
        
        res.registerAdvance()
        self.advance()

        expr = res.register(self.expr())
        if res.error: return res
        cases.append((condition, expr))

        while self.current_token.gettokentype() == TT_KEYWORD and self.current_token.getstr() == 'elif':
            res.registerAdvance()
            self.advance()

            condition = res.register(self.expr())
            if res.error: return res

            start_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno - 1, self.fn, self.txt)
            end_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno, self.fn, self.txt)
            if not (self.current_token.gettokentype() == TT_KEYWORD and self.current_token.getstr() == 'then'):
                return res.failure(InvalidSyntaxError(start_pos, end_pos, f"Expected 'then'"))
            
            res.registerAdvance()
            self.advance()

            expr = res.register(self.expr())
            if res.error: return res
            cases.append((condition, expr))

        if self.current_token.gettokentype() == TT_KEYWORD and self.current_token.getstr() == 'else':
            res.registerAdvance()
            self.advance()
            else_case = res.register(self.expr())
            if res.error: return res
        
        return res.success(IfNode(cases, else_case))
    
    def binaryOperation(self, funcA, operationTokens, funcB = None):
        if funcB == None:
            funcB = funcA
        res = ParseResult()
        left = res.register(funcA())
        if res.error: return res
        while self.current_token.gettokentype() in operationTokens or (self.current_token.gettokentype(), self.current_token.getstr()) in operationTokens:
            operationToken = self.current_token
            res.registerAdvance()
            self.advance()
            right = res.register(funcB())
            if res.error: return res
            left = BinaryOperationNode(left, operationToken, right, self.fn, self.txt)
        return res.success(left)

#RUNTIME RESULT

class RuntimeResult:
    def __init__(self):
        self.value = None
        self.error = None

    def register(self, res):
        if res.error: self.error = res.error
        return res.value
    
    def success(self, value):
        self.value = value
        return self

    def failure(self, error):
        self.error = error
        return self

#VALUES

class Number:
    def __init__(self, value):
        self.value = float(value)
        self.setPosition()
    
    def setPosition(self, start_pos = None, end_pos = None):
        self.start_pos = start_pos
        self.end_pos = end_pos
        return self
    
    def add(self, number):
        if isinstance(number, Number):
            return Number(self.value + number.value), None

    def subtract(self, number):
        if isinstance(number, Number):
            return Number(self.value - number.value), None
    
    def multiply(self, number):
        if isinstance(number, Number):
            return Number(self.value * number.value), None
    
    def divide(self, number):
        if isinstance(number, Number):
            if number.value == 0:
                return None, RuntimeError(number.start_pos, number.end_pos, 'Division by 0')
            return Number(self.value / number.value), None

    def pow(self, number):
        if isinstance(number, Number):
            return Number(self.value ** number.value), None

    def compare_eq(self, number):
        if isinstance(number, Number):
            return Number(int(self.value == number.value)), None

    def compare_ne(self, number):
        if isinstance(number, Number):
            return Number(int(self.value != number.value)), None
    
    def compare_gt(self, number):
        if isinstance(number, Number):
            return Number(int(self.value > number.value)), None

    def compare_lt(self, number):
        if isinstance(number, Number):
            return Number(int(self.value < number.value)), None

    def compare_gte(self, number):
        if isinstance(number, Number):
            return Number(int(self.value >= number.value)), None

    def compare_lte(self, number):
        if isinstance(number, Number):
            return Number(int(self.value <= number.value)), None

    def and_by(self, number):
        if isinstance(number, Number):
            return Number(int(self.value and number.value)), None

    def or_by(self, number):
        if isinstance(number, Number):
            return Number(int(self.value or number.value)), None

    def not_num(self):
        return Number(1 if self.value == 0 else 0), None

    def is_true(self):
        return self.value != 0

    def copy(self):
        copy = Number(self.value)
        copy.setPosition(self.start_pos, self.end_pos)
        return copy

    def __repr__(self):
        return str(self.value)

#CONTEXT

class Context:
    def __init__(self):
        self.symbolTable = None

#SYMBOL TABLE

class SymbolTable:
    def __init__(self):
        self.symbols = {}
        self.parent = None
    
    def get(self, name):
        value = self.symbols.get(name, None)
        if not value and self.parent:
            return self.parent.get(name)
        return value
    
    def set(self, name, value):
        self.symbols[name] = value
    
    def remove(self, name):
        del self.symbols[name]

#INTERPRETER

class Interpreter:
    def visit(self, node, context):
        methodName = f'visit_{type(node).__name__}'
        method = getattr(self, methodName, self.noVisit)
        return method(node, context)
    
    def noVisit(self, node, context):
        raise Exception(f'No visit_{type(node).__name__} method defined')

    def visit_VarAccessNode(self, node, context):
        res = RuntimeResult()
        varName = node.varNameToken.value
        value = context.symbolTable.get(varName)

        if not value:
            res.failure(RuntimeError(node.start_pos, node.end_pos, f"'{varName}' is undefined"))
        value = value.copy().setPosition(node.start_pos, node.end_pos)
        return res.success(value)

    def visit_VarAssignNode(self, node, context):
        res = RuntimeResult()
        varName = node.varNameToken.value
        value = res.register(self.visit(node.valueNode, context))
        if res.error: return res

        context.symbolTable.set(varName, value)
        return res.success(value)

    def visit_NumberNode(self, node, context):
        return RuntimeResult().success(Number(node.token.getstr()).setPosition(node.start_pos, node.end_pos))

    def visit_BinaryOperationNode(self, node, context):
        res = RuntimeResult()
        left = res.register(self.visit(node.leftNode, context))
        if res.error: return res
        right = res.register(self.visit(node.rightNode, context))
        if res.error: return res

        error = None
        if node.opToken.gettokentype() == TT_PLUS:
            result, error = left.add(right)
        elif node.opToken.gettokentype() == TT_MINUS:
            result, error = left.subtract(right)
        elif node.opToken.gettokentype() == TT_MUL:
            result, error = left.multiply(right)
        elif node.opToken.gettokentype() == TT_DIV:
            result, error = left.divide(right)
        elif node.opToken.gettokentype() == TT_POW:
            result, error = left.pow(right)
        elif node.opToken.gettokentype() == TT_EE:
            result, error = left.compare_eq(right)
        elif node.opToken.gettokentype() == TT_NE:
            result, error = left.compare_ne(right)
        elif node.opToken.gettokentype() == TT_LT:
            result, error = left.compare_lt(right)
        elif node.opToken.gettokentype() == TT_GT:
            result, error = left.compare_gt(right)
        elif node.opToken.gettokentype() == TT_LTE:
            result, error = left.compare_lte(right)
        elif node.opToken.gettokentype() == TT_GTE:
            result, error = left.compare_gte(right)
        elif node.opToken.gettokentype() == TT_KEYWORD and node.opToken.getstr() == "and":
            result, error = left.and_by(right)
        elif node.opToken.gettokentype() == TT_KEYWORD and node.opToken.getstr() == "or":
            result, error = left.or_by(right)

        if error: 
            return res.failure(error)
        else: 
            return res.success(result.setPosition(node.start_pos, node.end_pos))

    def visit_UnaryOperationNode(self, node, context):
        res = RuntimeResult()
        number = res.register(self.visit(node.node, context))
        if res.error: return res
        error = None
        if node.opToken.gettokentype() == TT_MINUS:
            number, error = number.multiply(Number(-1))
        elif node.opToken.gettokentype() == TT_KEYWORD and node.opToken.getstr() == "not":
            number, error = number.not_num()
        if error: 
            return res.failure(error)
        else:
            return res.success(number.setPosition(node.start_pos, node.end_pos))

    def visit_IfNode(self, node, context):
        res = RuntimeResult()

        for condition, expr in node.cases:
            condition_value = res.register(self.visit(condition, context))
            if res.error: return res

            if condition_value.is_true():
                expr_value = res.register(self.visit(expr, context))
                if res.error: return res
                return res.success(expr_value)
        
        if node.else_case:
            else_value = res.register(self.visit(node.else_case, context))
            if res.error: return res
            return res.success(else_value)
        return res.success(None)

#RUN

globalSymbolTable = SymbolTable()

def run(fn, text):
    #Generate tokens
    lexer = Lexer(fn, text)
    token_list, token_count, error = lexer.get_token_list()
    if error: return None, error

    #Generate abstract syntax tree
    parser = Parser(token_list, token_count, fn, text)
    ast = parser.parse()
    if ast.error: return None, ast.error
    
    #Run the program
    interpreter = Interpreter()
    context = Context()
    context.symbolTable = globalSymbolTable
    result = interpreter.visit(ast.node, context)

    return result.value, result.error