#IMPORTS

from errorArrows import *
import string

#CONSTANTS

DIGITS = '0123456789'
LETTERS = string.ascii_letters
LETTERS_DIGITS = LETTERS + DIGITS

#ERRORS

class Error:
    def __init__(self, startPos, endPos, name, details):
        self.startPos = startPos
        self.endPos = endPos
        self.name = name
        self.details = details
    
    def toString(self):
        result = f'{self.name}: {self.details}'
        result += f' File {self.startPos.fn}, line {self.startPos.ln + 1}'
        result += '\n\n' + errorArrows(self.startPos.ftxt, self.startPos, self.endPos)
        return result

class IllegalCharError(Error):
    def __init__(self, startPos, endPos, details):
        super().__init__(startPos, endPos, 'Illegal Character', details)

class InvalidSyntaxError(Error):
    def __init__(self, startPos, endPos, details):
        super().__init__(startPos, endPos, 'Invalid Syntax', details)

class RuntimeError(Error):
    def __init__(self, startPos, endPos, details):
        super().__init__(startPos, endPos, 'Runtime error', details)

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
TT_LPAREN       = 'LPAREN'      #LEFT PARENTHESIS TOKEN TYPE
TT_RPAREN       = 'RPAREN'      #RIGHT PARENTHESIS TOKEN TYPE
TT_EOF          = 'EOF'         #END OF FILE TOKEN TYPE

KEYWORDS        = [
    'num'
]

class Token:
    def __init__(self, type, value = None, startPos = None, endPos = None):
        self.type = type
        self.value = value
        if startPos: 
            self.startPos = startPos.copy()
            self.endPos = startPos.copy()
            self.endPos.advance()
        if endPos: self.endPos = endPos

    def __repr__(self):
        if self.value: return f'{self.type}:{self.value}'
        return f'{self.type}' 
    
    def matches(self, type, value):
        return self.type == type and self.value == value

#LEXER

class Lexer:
    def __init__(self, fn, text):
        self.fn = fn
        self.text = text
        self.pos = Position(-1, 0, -1, fn, text)
        self.currentChar = None
        self.advance()
    
    def advance(self):
        self.pos.advance(self.currentChar)
        self.currentChar = self.text[self.pos.index] if self.pos.index < len(self.text) else None

    def makeTokens(self):
        tokens = []
        while self.currentChar != None:
            if self.currentChar in ' \t':
                self.advance()
            elif self.currentChar in DIGITS:
                tokens.append(self.makeNumber())
            elif self.currentChar in LETTERS:
                tokens.append(self.makeIdentifier())
            elif self.currentChar == '+':
                tokens.append(Token(TT_PLUS, startPos=self.pos))
                self.advance()
            elif self.currentChar == '-':
                tokens.append(Token(TT_MINUS, startPos=self.pos))
                self.advance()
            elif self.currentChar == '*':
                tokens.append(Token(TT_MUL, startPos=self.pos))
                self.advance()
            elif self.currentChar == '/':
                tokens.append(Token(TT_DIV, startPos=self.pos))
                self.advance()
            elif self.currentChar == '^':
                tokens.append(Token(TT_POW, startPos=self.pos))
                self.advance()
            elif self.currentChar == '=':
                tokens.append(Token(TT_EQ, startPos=self.pos))
                self.advance()
            elif self.currentChar == '(':
                tokens.append(Token(TT_LPAREN, startPos=self.pos))
                self.advance()
            elif self.currentChar == ')':
                tokens.append(Token(TT_RPAREN, startPos=self.pos))
                self.advance()
            else:
                startPos = self.pos.copy()
                char = self.currentChar
                self.advance()
                return [], IllegalCharError(startPos, self.pos, "'" + char + "'")
        tokens.append(Token(TT_EOF, startPos = self.pos))
        return tokens, None

    def makeNumber(self):
        numStr = ''
        dotCount = 0
        startPos = self.pos.copy()

        while self.currentChar != None and self.currentChar in DIGITS + '.':
            if self.currentChar == '.':
                if dotCount == 1: break
                dotCount += 1
            numStr += self.currentChar
            self.advance()
        return Token(TT_NUM, float(numStr), startPos, self.pos)

    def makeIdentifier(self):
        identifierString = ''
        startPos = self.pos.copy()

        while self.currentChar != None and self.currentChar in LETTERS_DIGITS + '_':
            identifierString += self.currentChar
            self.advance()
        
        tokenType = TT_KEYWORD if identifierString in KEYWORDS else TT_IDENTIFIER
        return Token(tokenType, identifierString, startPos, self.pos)

#NODES

class VarAccessNode:
    def __init__(self, varNameToken):
        self.varNameToken = varNameToken
        self.startPos = self.varNameToken.startPos
        self.endPos = self.varNameToken.endPos

class VarAssignNode:
    def __init__(self, varNameToken, valueNode):
        self.varNameToken = varNameToken
        self.valueNode = valueNode
        self.startPos = self.varNameToken.startPos
        self.endPos = self.valueNode.endPos

class NumberNode:
    def __init__(self, token):
        self.token = token
        self.startPos = self.token.startPos
        self.endPos = self.token.endPos
    def __repr__(self):
        return f'{self.token}'

class BinaryOperationNode:
    def __init__(self, leftNode, opToken, rightNode):
        self.leftNode = leftNode
        self.opToken = opToken
        self.rightNode = rightNode
        self.startPos = self.leftNode.startPos
        self.endPos = self.rightNode.endPos

    def __repr__(self):
        return f'({self.leftNode}, {self.opToken}, {self.rightNode})'

class UnaryOperationNode:
    def __init__(self, opToken, node):
        self.opToken = opToken
        self.node = node
        self.startPos = self.opToken.startPos
        self.endPos = self.node.endPos
    
    def __repr__(self):
        return f'({self.opToken}, {self.node})'

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
    def __init__(self, tokens):
        self.tokens = tokens
        self.tokenIndex = -1
        self.advance()

    def advance(self):
        self.tokenIndex += 1
        if self.tokenIndex < len(self.tokens):
            self.currentToken = self.tokens[self.tokenIndex]
        return self.currentToken

    def parse(self):
        res = self.expr()
        if not res.error and self.currentToken.type != TT_EOF:
            return res.failure(InvalidSyntaxError(self.currentToken.startPos, self.currentToken.endPos, "Expected '+', '-', '*' or '/'"))
        return res

    def atom(self):
        res = ParseResult()
        token = self.currentToken

        if token.type == TT_NUM:
            res.registerAdvance()
            self.advance()
            return res.success(NumberNode(token))

        elif token.type == TT_IDENTIFIER:
            res.registerAdvance()
            self.advance()
            return res.success(VarAccessNode(token))

        elif token.type == TT_LPAREN:
            res.registerAdvance()
            self.advance()
            expr = res.register(self.expr())
            if res.error: return res
            if self.currentToken.type == TT_RPAREN:
                res.registerAdvance()
                self.advance()
                return res.success(expr)
            else:
                return res.failure(InvalidSyntaxError(
                    self.currentToken.startPos, self.currentToken.endPos,
                    "Expected ')'"
                ))

        return res.failure(InvalidSyntaxError(
            token.startPos, token.endPos,
            "Expected number, identifier, '+', '-' or '('"
        ))

    def power(self):
        return self.binaryOperation(self.atom, (TT_POW, ), self.factor)


    def factor(self):
        res = ParseResult()
        token = self.currentToken

        if token.type in (TT_PLUS, TT_MINUS):
            res.registerAdvance()
            self.advance()
            factor = res.register(self.factor())
            if res.error: return res
            return res.success(UnaryOperationNode(token, factor))

        return self.power()

    def term(self):
        return self.binaryOperation(self.factor, (TT_MUL, TT_DIV))


    def expr(self):
        res = ParseResult()
        if self.currentToken.matches(TT_KEYWORD, 'num'):
            res.registerAdvance()
            self.advance()
            if self.currentToken.type != TT_IDENTIFIER:
                return res.failure(InvalidSyntaxError(self.currentToken.startPos, self.currentToken.endPos, "Expected an identifier"))
            varName = self.currentToken
            res.registerAdvance()
            self.advance()
            if self.currentToken.type != TT_EQ:
                return res.failure(InvalidSyntaxError(self.currentToken.startPos, self.currentToken.endPos, "Expected '='"))
            res.registerAdvance()
            self.advance()
            expr = res.register(self.expr())
            if res.error: return res
            return res.success(VarAssignNode(varName, expr))

        node = res.register(self.binaryOperation(self.term, (TT_PLUS, TT_MINUS)))
        if res.error: return res.failure(InvalidSyntaxError(self.currentToken.startPos, self.currentToken.endPos, "Expected 'num', number, indentifier, '+', '-' or '('"))
        return res.success(node)
    
    def binaryOperation(self, funcA, operationTokens, funcB = None):
        if funcB == None:
            funcB = funcA
        res = ParseResult()
        left = res.register(funcA())
        if res.error: return res
        while self.currentToken.type in operationTokens:
            operationToken = self.currentToken
            res.registerAdvance()
            self.advance()
            right = res.register(funcB())
            if res.error: return res
            left = BinaryOperationNode(left, operationToken, right)
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
        self.value = value
        self.setPosition()
    
    def setPosition(self, startPos = None, endPos = None):
        self.startPos = startPos
        self.endPos = endPos
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
                return None, RuntimeError(number.startPos, number.endPos, 'Division by 0')
            return Number(self.value / number.value), None

    def pow(self, number):
        if isinstance(number, Number):
            return Number(self.value ** number.value), None

    def copy(self):
        copy = Number(self.value)
        copy.setPosition(self.startPos, self.endPos)
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
            res.failure(RuntimeError(node.startPos, node.endPos, f"'{varName}' is undefined"))
        value = value.copy().setPosition(node.startPos, node.endPos)
        return res.success(value)

    def visit_VarAssignNode(self, node, context):
        res = RuntimeResult()
        varName = node.varNameToken.value
        value = res.register(self.visit(node.valueNode, context))
        if res.error: return res

        context.symbolTable.set(varName, value)
        return res.success(value)

    def visit_NumberNode(self, node, context):
        return RuntimeResult().success(Number(node.token.value).setPosition(node.startPos, node.endPos))

    def visit_BinaryOperationNode(self, node, context):
        res = RuntimeResult()
        left = res.register(self.visit(node.leftNode, context))
        if res.error: return res
        right = res.register(self.visit(node.rightNode, context))
        if res.error: return res

        error = None
        if node.opToken.type == TT_PLUS:
            result, error = left.add(right)
        elif node.opToken.type == TT_MINUS:
            result, error = left.subtract(right)
        elif node.opToken.type == TT_MUL:
            result, error = left.multiply(right)
        elif node.opToken.type == TT_DIV:
            result, error = left.divide(right)
        elif node.opToken.type == TT_POW:
            result, error = left.pow(right)

        if error: 
            return res.failure(error)
        else: 
            return res.success(result.setPosition(node.startPos, node.endPos))

    def visit_UnaryOperationNode(self, node, context):
        res = RuntimeResult()
        number = res.register(self.visit(node.node, context))
        if res.error: return res
        error = None
        if node.opToken.value == TT_MINUS:
            number, error = number.multiply(Number(-1))
        if error: 
            return res.failure(error)
        else:
            return res.success(number.setPosition(node.startPos, node.endPos))

#RUN

globalSymbolTable = SymbolTable()

def run(fn, text):
    #Generate tokens
    lexer = Lexer(fn, text)
    tokens, error = lexer.makeTokens()
    if error: return None, error

    #Generate abstract syntax tree
    parser = Parser(tokens)
    ast = parser.parse()
    if ast.error: return None, ast.error

    #Run the program
    interpreter = Interpreter()
    context = Context()
    context.symbolTable = globalSymbolTable
    result = interpreter.visit(ast.node, context)

    return result.value, result.error