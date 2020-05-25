#IMPORTS

from errorArrows import *
import string
import os
from rply import LexerGenerator

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
    def __init__(self, start_pos, end_pos, details, context):
        super().__init__(start_pos, end_pos, 'Runtime error', details)
        self.context = context

class TypeError(Error):
    def __init__(self, start_pos, end_pos, details):
        super().__init__(start_pos, end_pos, 'Type error', details)

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
TT_STR          = 'STR'         #STRING TOKEN TYPE
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
TT_LSQUARE      = 'LSQUARE'     #LEFT SQUARE BRACKET
TT_RSQUARE      = 'RSQUARE'     #RIGHT SQUARE BRACKET
TT_EOF          = 'EOF'         #END OF FILE TOKEN TYPE
TT_COMMA        = 'COMMA'       #COMMA TOKEN TYPE
TT_ARROW        = 'ARROW'       #ARROW TOKEN TYPE
TT_ERROR        = 'ERROR'       #ILLEGAL TEXT ERROR
TT_NL           = 'NEWLINE'     #NEW LINE

KEYWORDS        = [
    'num',
    'str',
    'and',
    'not',
    'or',
    'if',
    'then',
    'elif',
    'else',
    'for',
    'to',
    'step',
    'function',
    'while',
    'end',
    'return',
    'break'
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
        self.lexer_generator.add(TT_STR, r"""(?=["'])(?:"[^"\\]*(?:\\[\s\S][^"\\]*)*"|'[^'\\]*(?:\\[\s\S][^'\\]*)*')""")
        self.lexer_generator.add(TT_KEYWORD, r"\b(function|num|and|or|not|end|if|then|elif|else|for|to|step|while|str)\b")
        self.lexer_generator.add(TT_NUM, r"(\d+(\.\d+)?)")
        self.lexer_generator.add(TT_PLUS, r"\+")
        self.lexer_generator.add(TT_ARROW, r"->")
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
        self.lexer_generator.add(TT_COMMA, r",")
        self.lexer_generator.add(TT_LPAREN, r"\(")
        self.lexer_generator.add(TT_RPAREN, r"\)")
        self.lexer_generator.add(TT_LSQUARE, r"\[")
        self.lexer_generator.add(TT_RSQUARE, r"\]")
        self.lexer_generator.add(TT_IDENTIFIER, r"[a-zA-Z][a-zA-Z0-9]*")
        self.lexer_generator.add(TT_NL, r";")
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
    def __init__(self, varType, varNameToken, valueNode, fn, txt):
        self.varType = varType
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

class StringNode:
    def __init__(self, token, fn, txt):
        self.token = token
        self.start_pos = Position(self.token.getsourcepos().idx, self.token.getsourcepos().lineno - 1, self.token.getsourcepos().colno - 1, fn, txt)
        self.end_pos = Position(self.token.getsourcepos().idx, self.token.getsourcepos().lineno - 1, self.token.getsourcepos().colno, fn, txt)
    def __repr__(self):
        return f'{self.token}'

class ListNode:
    def __init__(self, elementNodes, start_pos, end_pos):
        self.elementNodes = elementNodes
        self.start_pos = start_pos
        self.end_pos = end_pos

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

class ForNode:
    def __init__(self, variable_name_token, start_value_node, end_value_node, step_value_node, body_node, should_return_null):
        self.variable_name_token = variable_name_token
        self.start_value_node = start_value_node
        self.end_value_node = end_value_node
        self.step_value_node = step_value_node
        self.body_node = body_node
        self.should_return_null = should_return_null

        self.start_pos = self.body_node.start_pos
        self.end_pos = self.body_node.end_pos

class WhileNode:
    def __init__(self, condition_node, body_node):
        self.condition_node = condition_node
        self.body_node = body_node

        self.start_pos = self.condition_node.start_pos
        self.end_pos = self.body_node.end_pos

class FunctionDefinitionNode:
    def __init__(self, var_name_token, arg_name_tokens, body_node, fn, txt, should_return_null):
        self.var_name_token = var_name_token
        self.arg_name_tokens = arg_name_tokens
        self.body_node = body_node
        self.should_return_null = should_return_null

        if self.var_name_token:
            self.start_pos = Position(self.var_name_token.getsourcepos().idx, self.var_name_token.getsourcepos().lineno - 1, self.var_name_token.getsourcepos().colno - 1, fn, txt)
        elif len(self.arg_name_tokens) > 0:
            self.start_pos = Position(self.arg_name_tokens[0].getsourcepos().idx, self.arg_name_tokens[0].getsourcepos().lineno - 1, self.arg_name_tokens[0].getsourcepos().colno - 1, fn, txt)
        else:
            self.start_pos = self.body_node.start_pos
        
        self.end_pos = self.body_node.end_pos

class CallNode:
    def __init__(self, node_to_call, arg_nodes):
        self.node_to_call = node_to_call
        self.arg_nodes = arg_nodes

        self.start_pos = node_to_call.start_pos
        if len(self.arg_nodes) > 0:
            self.end_pos = self.arg_nodes[-1].end_pos
        else:
            self.end_pos = node_to_call.end_pos

class ReturnNode:
  def __init__(self, node_to_return, start_pos, end_pos):
    self.node_to_return = node_to_return

    self.start_pos = start_pos
    self.end_pos = end_pos

class BreakNode:
  def __init__(self, start_pos, end_pos):
    self.start_pos = start_pos
    self.end_pos = end_pos

#PARSE RESULT

class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None
        self.advanceCount = 0
        self.reverseCount = 0
    
    def registerAdvance(self):
        self.advanceCount += 1

    def register(self, res):
        self.advanceCount += res.advanceCount
        if res.error: self.error = res.error
        return res.node

    def tryRegister(self, res):
        if res.error:
          self.reverseCount = res.advanceCount
          return None
        return self.register(res)

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
        self.update_current_token()
        return self.current_token

    def reverse(self, amount = 1):
        self.token_index -= amount
        self.update_current_token()
        return self.current_token

    def update_current_token(self):
         if self.token_index >= 0 and self.token_index < self.token_count:
            self.current_token = self.token_list[self.token_index]

    def parse(self):
        res = self.statements()
        if not res.error and self.token_index != self.token_count:
            start_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno - 1, self.fn, self.txt)
            end_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno, self.fn, self.txt)
            return res.failure(InvalidSyntaxError(start_pos, end_pos, "Expected '+', '-', '*' or '/'"))
        return res

    def call(self):
        res = ParseResult()

        atom = res.register(self.atom())
        if res.error: return res

        if self.current_token.gettokentype() == TT_LPAREN:
            res.registerAdvance()
            self.advance()
            arg_nodes = []

            if self.current_token.gettokentype() == TT_RPAREN:
                res.registerAdvance()
                self.advance()
            else:
                arg_nodes.append(res.register(self.expr()))
                if res.error:
                    return res.failure(InvalidSyntaxError(
                    self.current_token.getsourcepos().colno - 1, self.current_token.getsourcepos().colno,
                    "Expected ']', ')', 'num', 'if', 'for', 'while', 'function', identifier, number"
                ))

                while self.current_token.gettokentype() == TT_COMMA:
                    res.registerAdvance()
                    self.advance()

                    arg_nodes.append(res.register(self.expr()))
                    if res.error: return res
                
                if self.current_token.gettokentype() != TT_RPAREN:
                    return res.failure(InvalidSyntaxError(
                    self.current_token.getsourcepos().colno - 1, self.current_token.getsourcepos().colno,
                    "Expected ')' or ','"
                ))

                res.registerAdvance()
                self.advance()
            return res.success(CallNode(atom, arg_nodes))
        return res.success(atom)

    def atom(self):
        res = ParseResult()
        token = self.current_token

        if token.gettokentype() == TT_NUM:
            res.registerAdvance()
            self.advance()
            return res.success(NumberNode(token, self.fn, self.txt))

        if token.gettokentype() == TT_STR:
            res.registerAdvance()
            self.advance()
            return res.success(StringNode(token, self.fn, self.txt))

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

        elif token.gettokentype() == TT_LSQUARE:
            list_expr = res.register(self.list_expr())
            if res.error: return res
            return res.success(list_expr)

        elif token.gettokentype() == TT_KEYWORD and token.getstr() == "if":
            if_expr = res.register(self.if_expr())
            if res.error: return res
            return res.success(if_expr)

        elif token.gettokentype() == TT_KEYWORD and token.getstr() == "for":
            for_expr = res.register(self.for_expr())
            if res.error: return res
            return res.success(for_expr)

        elif token.gettokentype() == TT_KEYWORD and token.getstr() == "while":
            while_expr = res.register(self.while_expr())
            if res.error: return res
            return res.success(while_expr)

        elif token.gettokentype() == TT_KEYWORD and token.getstr() == "function":
            func_def = res.register(self.func_def())
            if res.error: return res
            return res.success(func_def)
        
        start_pos = Position(token.getsourcepos().idx, token.getsourcepos().lineno - 1, token.getsourcepos().colno - 1, self.fn, self.txt)
        end_pos = Position(token.getsourcepos().idx, token.getsourcepos().lineno - 1, token.getsourcepos().colno, self.fn, self.txt)
        return res.failure(InvalidSyntaxError(
            start_pos, end_pos, "Expected number, identifier, '+', '-' or '(', '[', 'if', 'for', 'while', 'function'"
        ))

    def power(self):
        return self.binaryOperation(self.call, (TT_POW, ), self.factor)


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

    def statements(self):
        res = ParseResult()
        statements = []
        start_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno - 1, self.fn, self.txt).copy()
        end_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno, self.fn, self.txt)

        while self.current_token.gettokentype() == TT_NL:
            res.registerAdvance()
            self.advance()

        statement = res.register(self.statement())
        if res.error: return res
        statements.append(statement)

        moreStatements = True

        while True:
            nlCount = 0
            while self.current_token.gettokentype() == TT_NL and self.token_index != self.token_count:
                res.registerAdvance()
                self.advance()
                nlCount += 1
            if nlCount == 0:
                moreStatements = False

            if not moreStatements: break
            statement = res.tryRegister(self.statement())
            if not statement:
                self.reverse(res.reverseCount)
                moreStatements = False
                continue
            statements.append(statement)

        return res.success(ListNode(statements, start_pos, end_pos.copy()))
            
    def statement(self):
        res = ParseResult()
        start_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno - 1, self.fn, self.txt).copy()
        if self.current_token.getstr() == 'return':
          res.registerAdvance()
          self.advance()

          expr = res.tryRegister(self.expr())
          if not expr:
            self.reverse(res.reverseCount)
          return res.success(ReturnNode(expr, start_pos, self.current_token.start_pos.copy()))
    
        if self.current_token.getstr() == 'break':
          res.registerAdvance()
          self.advance()
          return res.success(BreakNode(start_pos, self.current_token.start_pos.copy()))

        expr = res.register(self.expr())
        start_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno - 1, self.fn, self.txt).copy()
        end_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno, self.current_token.getsourcepos().colno, self.fn, self.txt).copy()
        if res.error:
          return res.failure(InvalidSyntaxError(
            start_pos, end_pos, "Expected 'return', 'break', 'num', 'if', 'for', 'while', 'function', identifier, '+', '-', '(' or '['"
          ))
        return res.success(expr)

    def expr(self):
        res = ParseResult()
        if self.current_token.gettokentype() == TT_KEYWORD and (self.current_token.getstr() == 'num' or self.current_token.getstr() == 'str'):
            varType = self.current_token.getstr()
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
            return res.success(VarAssignNode(varType, var_name, expr, self.fn, self.txt))

        node = res.register(self.binaryOperation(self.comp_expr, ((TT_KEYWORD, "and"), (TT_KEYWORD, "or"))))
        if res.error: 
            start_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno - 1, self.fn, self.txt)
            end_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno, self.fn, self.txt)
            return res.failure(InvalidSyntaxError(start_pos, end_pos, "Expected 'num', number, indentifier, '+', '-', '[', '\"' or '('"))
        return res.success(node)
    
    def list_expr(self):
        res = ParseResult()
        elementNodes = []
        start_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno - 1, self.fn, self.txt)

        if self.current_token.gettokentype() != TT_LSQUARE:
          return res.failure(InvalidSyntaxError(
            self.current_token.start_pos, self.current_token.end_pos,
            f"Expected '['"
          ))

        res.registerAdvance()
        self.advance()

        if self.current_token.gettokentype() == TT_RSQUARE:
          res.registerAdvance()
          self.advance()
        else:
          elementNodes.append(res.register(self.expr()))
          if res.error:
            return res.failure(InvalidSyntaxError(
              self.current_token.start_pos, self.current_token.end_pos,
              "Expected ']', 'VAR', 'IF', 'FOR', 'WHILE', 'FUN', int, float, identifier, '+', '-', '(', '[' or 'NOT'"
            ))

          while self.current_token.gettokentype() == TT_COMMA:
            res.registerAdvance()
            self.advance()

            elementNodes.append(res.register(self.expr()))
            if res.error: return res

          if self.current_token.gettokentype() != TT_RSQUARE:
            return res.failure(InvalidSyntaxError(
              self.current_token.start_pos, self.current_token.end_pos,
              f"Expected ',' or ']'"
            ))

          res.registerAdvance()
          self.advance()

        end_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno, self.fn, self.txt)
        return res.success(ListNode(elementNodes, start_pos, end_pos))

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

    def for_expr(self):
        res = ParseResult()

        if not (self.current_token.gettokentype() == TT_KEYWORD and self.current_token.getstr() == 'for'):
            start_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno - 1, self.fn, self.txt)
            end_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno, self.fn, self.txt)
            return res.failure(InvalidSyntaxError(start_pos, end_pos,f"Expected 'for'"))
        
        res.registerAdvance()
        self.advance()

        if self.current_token.gettokentype() != TT_IDENTIFIER:
            start_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno - 1, self.fn, self.txt)
            end_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno, self.fn, self.txt)
            return res.failure(InvalidSyntaxError(start_pos, end_pos, f"Expected identifier"))

        variable_name = self.current_token
        res.registerAdvance()
        self.advance()

        if self.current_token.gettokentype() != TT_EQ:
            start_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno - 1, self.fn, self.txt)
            end_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno, self.fn, self.txt)
            return res.failure(InvalidSyntaxError(start_pos, end_pos,f"Expected '='"))
        
        res.registerAdvance()
        self.advance()

        start_value = res.register(self.expr())
        if res.error: return res

        if not (self.current_token.gettokentype() == TT_KEYWORD and self.current_token.getstr() == 'to'):
            start_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno - 1, self.fn, self.txt)
            end_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno, self.fn, self.txt)
            return res.failure(InvalidSyntaxError(start_pos, end_pos,f"Expected 'to'"))
        
        res.registerAdvance()
        self.advance()

        end_value = res.register(self.expr())
        if res.error: return res

        if self.current_token.gettokentype() == TT_KEYWORD and self.current_token.getstr() == 'step':
            res.registerAdvance()
            self.advance()

            step_value = res.register(self.expr())
            if res.error: return res
        else:
            step_value = None
        
        if not (self.current_token.gettokentype() == TT_KEYWORD and self.current_token.getstr() == 'then'):
            start_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno - 1, self.fn, self.txt)
            end_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno, self.fn, self.txt)
            return res.failure(InvalidSyntaxError(start_pos, end_pos,f"Expected 'then'"))
        
        res.registerAdvance()
        self.advance()

        if self.current_token.gettokentype() == TT_NL:
          res.registerAdvance()
          self.advance()

          body = res.register(self.statements())
          if res.error: return res

          if not (self.current_token.gettokentype() == TT_KEYWORD and self.current_token.getstr() == 'end'):
                start_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno - 1, self.fn, self.txt)
                end_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno, self.fn, self.txt)
                return res.failure(InvalidSyntaxError(start_pos, end_pos,f"Expected 'end'"))
          
          res.registerAdvance()
          self.advance()

          return res.success(ForNode(variable_name, start_value, end_value, step_value, body, True))
    
        body = res.register(self.statement())
        if res.error: return res

        return res.success(ForNode(variable_name, start_value, end_value, step_value, body, False))

    def while_expr(self):
        res = ParseResult()

        if not (self.current_token.gettokentype() == TT_KEYWORD and self.current_token.getstr() == 'while'):
            start_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno - 1, self.fn, self.txt)
            end_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno, self.fn, self.txt)
            return res.failure(InvalidSyntaxError(start_pos, end_pos,f"Expected 'while'"))
        
        res.registerAdvance()
        self.advance()

        condition = res.register(self.expr())
        if res.error: return res

        if not (self.current_token.gettokentype() == TT_KEYWORD and self.current_token.getstr() == 'then'):
            start_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno - 1, self.fn, self.txt)
            end_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno, self.fn, self.txt)
            return res.failure(InvalidSyntaxError(start_pos, end_pos,f"Expected 'then'"))
        
        res.registerAdvance()
        self.advance()

        body = res.register(self.statements())
        if res.error: return res

        return res.success(WhileNode(condition, body))

    def func_def(self):
        res = ParseResult()

        if not (self.current_token.gettokentype() == TT_KEYWORD and self.current_token.getstr() == 'function'):
            start_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno - 1, self.fn, self.txt)
            end_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno, self.fn, self.txt)
            return res.failure(InvalidSyntaxError(start_pos, end_pos,f"Expected 'function'"))
        
        res.registerAdvance()
        self.advance()

        if self.current_token.gettokentype() == TT_IDENTIFIER:
            var_name_token = self.current_token
            res.registerAdvance()
            self.advance()
            if self.current_token.gettokentype() != TT_LPAREN:
                start_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno - 1, self.fn, self.txt)
                end_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno, self.fn, self.txt)
                return res.failure(InvalidSyntaxError(start_pos, end_pos,f"Expected '('"))
        else:
            var_name_token = None
            if self.current_token.gettokentype() != TT_LPAREN:
                start_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno - 1, self.fn, self.txt)
                end_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno, self.fn, self.txt)
                return res.failure(InvalidSyntaxError(start_pos, end_pos,f"Expected identifier or '('"))

        res.registerAdvance()
        self.advance()

        arg_name_tokens = []
        if self.current_token.gettokentype() == TT_IDENTIFIER:
            arg_name_tokens.append(self.current_token)
            res.registerAdvance()
            self.advance()

            while self.current_token.gettokentype() == TT_COMMA:
                res.registerAdvance()
                self.advance()
                if self.current_token.gettokentype() != TT_IDENTIFIER:
                    start_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno - 1, self.fn, self.txt)
                    end_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno, self.fn, self.txt)
                    return res.failure(InvalidSyntaxError(start_pos, end_pos,f"Expected identifier"))
                arg_name_tokens.append(self.current_token)
                res.registerAdvance()
                self.advance()
            
            if self.current_token.gettokentype() != TT_RPAREN:
                start_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno - 1, self.fn, self.txt)
                end_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno, self.fn, self.txt)
                return res.failure(InvalidSyntaxError(start_pos, end_pos,f"Expected ')'"))
        else:
            if self.current_token.gettokentype() != TT_RPAREN:
                start_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno - 1, self.fn, self.txt)
                end_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno, self.fn, self.txt)
                return res.failure(InvalidSyntaxError(start_pos, end_pos,f"Expected identifier or ')'"))
        
        res.registerAdvance()
        self.advance()

        if self.current_token.gettokentype() == TT_ARROW:
            res.registerAdvance()
            self.advance()

            node_to_return = res.register(self.expr())
            if res.error: return res

            return res.success(FunctionDefinitionNode(var_name_token, arg_name_tokens, node_to_return, self.fn, self.txt, True))
        
        if self.current_token.gettokentype()  != TT_NL:
            start_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno - 1, self.fn, self.txt)
            end_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno, self.fn, self.txt)
            return res.failure(InvalidSyntaxError(start_pos, self.end_pos, f"Expected '->' or NEWLINE"))

        res.registerAdvance()
        self.advance()

        body = res.register(self.statements())
        if res.error: return res

        if not (self.current_token.gettokentype() == TT_KEYWORD and self.current_token.getstr() == 'end'):
            start_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno - 1, self.fn, self.txt)
            end_pos = Position(self.current_token.getsourcepos().idx, self.current_token.getsourcepos().lineno - 1, self.current_token.getsourcepos().colno, self.fn, self.txt)
            return res.failure(InvalidSyntaxError(start_pos, end_pos, f"Expected 'end'"))

        res.registerAdvance()
        self.advance()
    
        return res.success(FunctionDefinitionNode(var_name_token, arg_name_tokens, node_to_return, self.fn, self.txt, False))

        
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

class Value:
    def _init__(self):
        self.setPosition()
        self.setContext()
    
    def setPosition(self, start_pos = None, end_pos = None):
        self.start_pos = start_pos
        self.end_pos = end_pos
        return self
    
    def setContext(self, context=None):
        self.context = context
        return self

    def add(self, number):
        return None, self.illegal_operation(number)

    def subtract(self, number):
        return None, self.illegal_operation(number)
    
    def multiply(self, number):
        return None, self.illegal_operation(number)
    
    def divide(self, number):
        return None, self.illegal_operation(number)

    def pow(self, number):
        return None, self.illegal_operation(number)

    def compare_eq(self, number):
        return None, self.illegal_operation(number)

    def compare_ne(self, number):
        return None, self.illegal_operation(number)
    
    def compare_gt(self, number):
        return None, self.illegal_operation(number)

    def compare_lt(self, number):
        return None, self.illegal_operation(number)

    def compare_gte(self, number):
        return None, self.illegal_operation(number)

    def compare_lte(self, number):
        return None, self.illegal_operation(number)

    def and_by(self, number):
        return None, self.illegal_operation(number)

    def or_by(self, number):
        return None, self.illegal_operation(number)

    def not_num(self):
        return None, self.illegal_operation()

    def is_true(self):
        return False

    def execute(self):
        return RuntimeResult().failure(self.illegal_operation())

    def copy(self):
        raise Exception('No copy method defined')
    
    def illegal_operation(self, other = None):
        if not other: other = self
        return RuntimeError(self.start_pos, other.end_pos, 'Illegal operation', self.context)

class Number(Value):
    def __init__(self, value):
        super().__init__()
        self.value = float(value)
    
    def add(self, number):
        if isinstance(number, Number):
            return Number(self.value + number.value), None
        else:
            return None, Value.illegal_operation(self, number)

    def subtract(self, number):
        if isinstance(number, Number):
            return Number(self.value - number.value), None
        else:
            return None, Value.illegal_operation(self, number)
    
    def multiply(self, number):
        if isinstance(number, Number):
            return Number(self.value * number.value), None
        else:
            return None, Value.illegal_operation(self, number)
    
    def divide(self, number):
        if isinstance(number, Number):
            if number.value == 0:
                return None, RuntimeError(number.start_pos, number.end_pos, 'Division by 0', self.context)
            return Number(self.value / number.value), None
        else:
            return None, Value.illegal_operation(self, number)

    def pow(self, number):
        if isinstance(number, Number):
            return Number(self.value ** number.value), None
        else:
            return None, Value.illegal_operation(self, number)

    def compare_eq(self, number):
        if isinstance(number, Number):
            return Number(int(self.value == number.value)), None
        else:
            return None, Value.illegal_operation(self, number)

    def compare_ne(self, number):
        if isinstance(number, Number):
            return Number(int(self.value != number.value)), None
        else:
            return None, Value.illegal_operation(self, number)
    
    def compare_gt(self, number):
        if isinstance(number, Number):
            return Number(int(self.value > number.value)), None
        else:
            return None, Value.illegal_operation(self, number)

    def compare_lt(self, number):
        if isinstance(number, Number):
            return Number(int(self.value < number.value)), None
        else:
            return None, Value.illegal_operation(self, number)

    def compare_gte(self, number):
        if isinstance(number, Number):
            return Number(int(self.value >= number.value)), None
        else:
            return None, Value.illegal_operation(self, number)

    def compare_lte(self, number):
        if isinstance(number, Number):
            return Number(int(self.value <= number.value)), None
        else:
            return None, Value.illegal_operation(self, number)

    def and_by(self, number):
        if isinstance(number, Number):
            return Number(int(self.value and number.value)), None
        else:
            return None, Value.illegal_operation(self, number)

    def or_by(self, number):
        if isinstance(number, Number):
            return Number(int(self.value or number.value)), None
        else:
            return None, Value.illegal_operation(self, number)

    def not_num(self):
        return Number(1 if self.value == 0 else 0), None

    def is_true(self):
        return self.value != 0

    def copy(self):
        copy = Number(self.value)
        copy.setPosition(self.start_pos, self.end_pos)
        return copy

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)

class String(Value):
    def __init__(self, value):
        super().__init__()
        self.value = value
    
    def add(self, other):
        if isinstance(other, String):
            return String(self.value[:-1] + other.value[1:]).setContext(self.context), None
        else:
            return None, Value.illegal_operation(self, other)
    
    def is_true(self):
        return len(self.value) > 0
    
    def copy(self):
        copy = String(self.value)
        copy.setPosition(self.start_pos, self.end_pos)
        copy.setContext(self.context)
        return copy

    def __str__(self):
        return self.value

    def __repr__(self):
        return f'{self.value}'

class BaseFunction(Value):
    def __init__(self, name):
        super().__init__()
        self.name = name or "<anonymous>"
    
    def generateNewContext(self):
        new_context = Context(self.context)
        new_context.symbolTable = SymbolTable(new_context.parent.symbolTable)
        return new_context

    def check_args(self, arg_names, args):
        res = RuntimeResult()

        if len(args) > len(arg_names):
            return res.failure(RuntimeError(self.start_pos, self.end_pos, f"Too many arguments passed into '{self}'", self.context))
        
        if len(args) < len(arg_names):
            return res.failure(RuntimeError(self.start_pos, self.end_pos, f"Not enough arguments passed into '{self}'", self.context))
        
        return res.success(None)
    
    def populateArgs(self, arg_names, args, exec_ctx):
        for i in range(len(args)):
            arg_name = arg_names[i]
            arg_value = args[i]
            arg_value.setContext(exec_ctx)
            exec_ctx.symbolTable.set(arg_name, arg_value)

    def checkAndPopulateArgs(self, arg_names, args, exec_ctx):
        res = RuntimeResult()
        res.register(self.check_args(arg_names, args))
        if res.error: return res
        self.populateArgs(arg_names, args, exec_ctx)
        return res.success(None)

class List(Value):
  def __init__(self, elements):
    super().__init__()
    self.elements = elements

  def add(self, other):
    new_list = self.copy()
    new_list.elements.append(other)
    return new_list, None
  
  def copy(self):
    copy = List(self.elements)
    copy.setPosition(self.start_pos, self.end_pos)
    copy.setContext(self.context)
    return copy

  def __str__(self):
    return ", ".join([str(x) for x in self.elements])

  def __repr__(self):
    return f'{", ".join(repr(x) for x in self.elements)}'

class Function(BaseFunction):
    def __init__(self, name, body_node, arg_names, should_return_null):
        super().__init__(name)
        self.body_node = body_node
        self.arg_names = arg_names
        self.should_return_null = should_return_null

    def execute(self, args):
        res = RuntimeResult()
        interpreter = Interpreter()
        exec_ctx = self.generateNewContext()
        res.register(self.checkAndPopulateArgs(self.arg_names, args, exec_ctx))
        if res.error: return res
        
        value = res.register(interpreter.visit(self.body_node, exec_ctx))
        if res.error: return res
        return res.success(Number(0) if self.should_return_null else value)
    
    def copy(self):
        copy = Function(self.name, self.body_node, self.arg_names, self.should_return_null)
        copy.setContext(self.context)
        copy.setPosition(self.start_pos, self.end_pos)
        return copy
    
    def __repr__(self):
        return f"<function '{self.name}'>"

class BuiltInFunction(BaseFunction):
    def __init__(self, name):
        super().__init__(name)

    def execute(self, args):
        res =  RuntimeResult()
        exec_ctx = self.generateNewContext()

        methodName = f'execute_{self.name}'
        method = getattr(self, methodName, self.noVisit)

        res.register(self.checkAndPopulateArgs(method.arg_names, args, exec_ctx))
        if res.error: return res

        returnVal = res.register(method(exec_ctx))
        if res.error: return res
        return res.success(returnVal)

    def noVisit(self, node, context):
        raise Exception(f'No execute_{self.name} method defined')

    def copy(self):
        copy = BuiltInFunction(self.name)
        copy.setContext(self.context)
        copy.setPosition(self.start_pos, self.end_pos)
        return copy

    def __repr__(self):
        return f'{self.name}'

    def execute_print(self, exec_ctx):
        print(str(exec_ctx.symbolTable.get('value')).strip('"\''))
        return RuntimeResult().success(Number(0))
    execute_print.arg_names = ['value']

    def execute_input(self, exec_ctx):
        text = input()
        return RuntimeResult().success(String(text))
    execute_input.arg_names = []

    def execute_inputNum(self, exec_ctx):
        while True:
            text = input()
            try: 
                number = int(text)
                break
            except ValueError:
                print(f"'{text}' must be an integer")
            return RuntimeResult().success(Number(number))
    execute_inputNum.arg_names = []

    def execute_len(self, exec_ctx):
        list_ = exec_ctx.symbolTable.get("list")

        if not isinstance(list_, List):
            return RuntimeResult().failure(RuntimeError(
                self.start_pos, self.end_pos, "Argument must be list", exec_ctx
            ))
        
        return RuntimeResult().success(Number(len(list_.elements)))
    execute_len.arg_names = ["list"]

    def execute_run(self, exec_ctx):
        fn = exec_ctx.symbolTable.get("fn")

        if not isinstance(fn, String):
            return RuntimeResult().failure(RuntimeError(
                self.start_pos, self.end_pos, "Argument must be string", exec_ctx
            ))
        
        fn = fn.value.replace("\"", "")
        try:
            with open(fn, "r") as f:
                script = f.read()
        except Exception as e:
            return RuntimeResult().failure(RuntimeError(
                self.start_pos, self.end_pos, f"Failed to load script \"{fn}\"\n" + str(e), exec_ctx
            ))
        
        _, error = run(fn, script)

        if error:
            return RuntimeResult().failure(RuntimeError(
                self.start_pos, self.end_pos, f"Failed to finish executing script \"{fn}\"\n" + error.toString(), exec_ctx
            ))

        return RuntimeResult().success(Number(0))
    execute_run.arg_names = ["fn"]

BuiltInFunction.print = BuiltInFunction("print")
BuiltInFunction.input = BuiltInFunction("input")
BuiltInFunction.inputNum = BuiltInFunction("inputNum")
BuiltInFunction.len = BuiltInFunction("len")
BuiltInFunction.run = BuiltInFunction("run")

#CONTEXT

class Context:
    def __init__(self, parent=None):
        self.symbolTable = None
        self.parent = parent

#SYMBOL TABLE

class SymbolTable:
    def __init__(self, parent=None):
        self.symbols = {}
        self.parent = parent
    
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
            return res.failure(RuntimeError(node.start_pos, node.end_pos, f"'{varName}' is undefined", context))
        value.setPosition(node.start_pos, node.end_pos).setContext(context)
        value = value.copy()
        return res.success(value)

    def visit_VarAssignNode(self, node, context):
        res = RuntimeResult()
        varName = node.varNameToken.value
        value = res.register(self.visit(node.valueNode, context))
        if res.error: return res

        if node.varType == 'num' and not isinstance(value, Number):
            return res.failure(TypeError(node.start_pos, node.end_pos, 'Expected a number'))

        if node.varType == 'str' and not isinstance(value, String):
            return res.failure(TypeError(node.start_pos, node.end_pos, 'Expected a string'))

        context.symbolTable.set(varName, value)
        return res.success(value)

    def visit_NumberNode(self, node, context):
        return RuntimeResult().success(Number(node.token.getstr()).setPosition(node.start_pos, node.end_pos))

    def visit_StringNode(self, node, context):
        return RuntimeResult().success(String(node.token.value).setPosition(node.start_pos, node.end_pos).setContext(context))

    def visit_ListNode(self, node, context):
        res = RuntimeResult()
        elements = []

        for elementNode in node.elementNodes:
          elements.append(res.register(self.visit(elementNode, context)))
          if res.error: return res

        return res.success(List(elements).setContext(context).setPosition(node.start_pos, node.end_pos))

    def visit_FunctionDefinitionNode(self, node, context):
        res = RuntimeResult()
        func_name = node.var_name_token.value if node.var_name_token else None
        body_node = node.body_node
        arg_names = [arg_name.value for arg_name in node.arg_name_tokens]
        func_value = Function(func_name, body_node, arg_names, node.should_return_null).setPosition(node.start_pos, node.end_pos).setContext(context)

        if node.var_name_token:
            context.symbolTable.set(func_name, func_value)
        return res.success(func_value)
    
    def visit_CallNode(self, node, context):
        res = RuntimeResult()
        args = []

        value_to_call = res.register(self.visit(node.node_to_call, context))
        if res.error: return res
        value_to_call = value_to_call.copy().setPosition(node.start_pos, node.end_pos)
        for arg_node in node.arg_nodes:
            args.append(res.register(self.visit(arg_node, context)))
            if res.error: return res
        return_value = res.register(value_to_call.execute(args))
        if res.error: return res
        return_value.setPosition(node.start_pos, node.end_pos).setContext(context)
        return_value = return_value.copy().setPosition(node.start_pos, node.end_pos).setContext(context)
        return res.success(return_value)

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

    def visit_ForNode(self, node, context):
        res = RuntimeResult()
        elements = []

        start_value = res.register(self.visit(node.start_value_node, context))
        if res.error: return res

        end_value = res.register(self.visit(node.end_value_node, context))
        if res.error: return res

        if node.step_value_node:
            step_value = res.register(self.visit(node.step_value_node, context))
            if res.error: return res
        else:
            step_value = Number(1)
        
        i = start_value.value

        if step_value.value >= 0:
            condition = lambda: i < end_value.value
        else:
            condition = lambda: i > end_value.value
        
        while condition():
            context.symbolTable.set(node.variable_name_token.value, Number(i))
            i += step_value.value

            elements.append(res.register(self.visit(node.body_node, context)))
            if res.error: return res
        
        return res.success(Number(0) if node.should_return_null else List(elements).setContext().setPosition())
    
    def visit_WhileNode(self, node, context):
        res = RuntimeResult()
        elements = []

        while True:
            condition = res.register(self.visit(node.condition_node, context))
            if res.error: return res

            if not condition.is_true(): break

            elements.append(res.register(self.visit(node.body_node, context)))
            if res.error: return res
        return res.success(List(elements).setContext().setPosition())

#RUN

globalSymbolTable = SymbolTable()
globalSymbolTable.set("print", BuiltInFunction.print)
globalSymbolTable.set("input", BuiltInFunction.input)
globalSymbolTable.set("inputNum", BuiltInFunction.inputNum)
globalSymbolTable.set("LEN", BuiltInFunction.len)
globalSymbolTable.set("RUN", BuiltInFunction.run)

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