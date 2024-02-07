# Copyright 2024 Leo (padril) Peckham

from __future__ import annotations
from typing import Tuple, List, Union, Optional, Set, Callable
from subprocess import Popen, PIPE
from fcntl import fcntl, F_GETFL, F_SETFL
from os import O_NONBLOCK
from time import sleep

class Value:
    value: Union[int, str, float]
    def __init__(self, value: Union[int, str, float]):
        self.value = value
    def __repr__(self) -> str:
        return f'Value: {self.value}'

class Operator:
    value: str
    def __init__(self, value: str):
        self.value = value
    def __repr__(self) -> str:
        return f'Operator: {self.value}'

class Ident:
    value: str
    def __init__(self, value: str):
        self.value = value

Token = Value | Operator | Ident

CombData = Optional[Tuple[str, List[str]]]
CombFn = Callable[[CombData], CombData]

class Combinator:
    fn: CombFn
    def __init__(self, fn: CombFn):
        self.fn = fn

    def __call__(self, cd: CombData) -> CombData:
        return self.fn(cd)

    def then(self, other: Combinator) -> Combinator:
        def then_fn(cd: CombData) -> CombData:
            return other(self(cd))
        return Combinator(then_fn)
    
    def then_cat(self, other: Combinator) -> Combinator:
        def then_cat_fn(cd: CombData) -> CombData:
            cd = other(self(cd))
            if not cd: return None
            return (cd[0], cd[1][0:-2] + [cd[1][-2] + cd[1][-1]])
        return Combinator(then_cat_fn)

    def then_opt(self, other: Combinator) -> Combinator:
        return self.then(other).alt(self)

    def then_opt_cat(self, other: Combinator) -> Combinator:
        return self.then_cat(other).alt(self)
        
    def alt(self, other: Combinator) -> Combinator:
        def alt_fn(cd: CombData) -> CombData:
            res = self(cd)
            if res:
                return res
            else:
                return other(cd)
        return Combinator(alt_fn)

    def any_cat(self) -> Combinator:
        def any_cat_fn(cd: CombData) -> CombData:
            last = cd
            cd = self(cd)
            if not cd: return None
            while cd and cd[0] != '':
                last = cd
                cd = self(cd)
                if cd and len(cd[1]) >= 2:
                    cd = (cd[0], cd[1][0:-2] + [cd[1][-2] + cd[1][-1]])
            return last if not cd else cd
        return Combinator(any_cat_fn)

    def any(self) -> Combinator:
        def any_fn(cd: CombData) -> CombData:
            last = cd
            cd = self(cd)
            if not cd: return None
            while cd and cd[0] != '':
                last = cd
                cd = self(cd)
            return last if not cd else cd
        return Combinator(any_fn)

    def discard(self) -> Combinator:
        def discard_fn(cd: CombData) -> CombData:
            res = self(cd)
            if res: return (res[0], res[1][0:-1])
            return None
        return Combinator(discard_fn)

def cterm(s: str) -> Combinator:
    def cterm_fn(cd: CombData) -> CombData:
        if not cd: return None
        if len(s) > len(cd[0]): return None
        for i in range(len(s)):
            if s[i] != cd[0][i]: return None
        return (cd[0][len(s):], cd[1] + [s])
    return Combinator(cterm_fn)

def cset(ss: List[str]) -> Combinator:
    def cset_fn(cd: CombData) -> CombData:
        for s in ss:
            res = cterm(s)(cd)
            if res: return res
        return None
    return Combinator(cset_fn)

def cdigit() -> Combinator:
    return cset(list(map(str, range(0, 10))))

def cwhite() -> Combinator:
    return cset([' ', '\t', '\n'])

def calpha() -> Combinator:
    return cset(list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'))

def calphanum() -> Combinator:
    return cdigit().alt(calpha())

def lex(s: str) -> List[str]:
    res = (
            cdigit().any_cat()                  # Integers
            .then_opt_cat(cterm('.').then_cat(cdigit().any_cat()))  # & Float
            .alt(cwhite().any_cat().discard())  # Whitespace
            .alt(cset(['+', '-', '*', '/', '=']))    # Operators
            .alt(calpha().then_opt_cat(calphanum().alt(cterm('_').any_cat())))
            .any()
           )((s, []))
    if res: return res[1]
    else: return []

def tokenize(command: str) -> List[Token]:
    tokens: List[Token] = []
    lexemes: List[str] = lex(command)
    for lexeme in lexemes:
        token: Optional[Token] = None
        if lexeme == '': continue
        elif lexeme.isdigit(): token = Value(int(lexeme))
        elif lexeme.isidentifier(): token = Ident(lexeme)
        elif lexeme == '+': token = Operator('+')
        elif lexeme == '=': token = Operator('=')
        if token: tokens.append(token)
    return tokens

class OpExpr:
    op: Operator
    x: 'Expr'
    y: 'Expr'
    def __init__(self, op: Operator, x: 'Expr', y: 'Expr'):
        self.op = op
        self.x = x
        self.y = y

Rvalue = Value | Ident
Expr = OpExpr | Rvalue
AST = Expr | List['AST']

def parse_expr(tokens: List[Token]) -> Tuple[List[Token], Expr]:
    if len(tokens) == 1 and \
            isinstance(tokens[0], Rvalue):
        return ([], tokens[0])
    if len(tokens) >= 3 and \
            isinstance(tokens[0], Rvalue) and \
            isinstance(tokens[1], Operator):
        (rem, right) = parse_expr(tokens[2:])
        return (rem, OpExpr(tokens[1], tokens[0], right))
    raise Exception('Parse error')

def parse(tokens: List[Token]) -> AST:
    ast: AST = []
    while tokens:
        (tokens, expr) = parse_expr(tokens)
        ast.append(expr)
    return ast

Env = Tuple[Optional['Env'], Set[str]]

IR = List[str | Tuple[str,...]]

def generate_ir(ast: AST, env: Env, ret: int) -> Tuple[IR, Env]:
    ir: IR = []
    local: Env = env
    if isinstance(ast, Expr):
        if isinstance(ast, Value):
            return ([('set', f'r{ret}', str(ast.value))], env)
        elif isinstance(ast, Ident):
            return ([('set', f'r{ret}', f'$v{ast.value}')], env)
        elif isinstance(ast, OpExpr):
            if ast.op.value == '=' and isinstance(ast.x, Ident):
                (rir, renv) = generate_ir(ast.y, env, ret + 1)
                return (rir + [('set', f'v{ast.x.value}', f'$r{ret+1}'),
                               ('set', f'r{ret}', f'$r{ret+1}')],
                        (env[0], env[1] | renv[1] | {ast.x.value}))
            (lir, lenv) = generate_ir(ast.x, env, ret)
            (rir, renv) = generate_ir(ast.y, env, ret + 1)
            return (lir + rir +\
                    [(str(ast.op.value), f'r{ret}', f'$r{ret+1}')],
                    (env[0], env[1] | lenv[1] | renv[1]))

    else:
        for node in ast:
            (rir, renv) = generate_ir(node, env, ret) 
            ir += rir;
            local = (local[0], local[1] | renv[1])

    return (ir, local)

def generate_bash(ir: IR) -> str:
    bash: str = ''
    for i in ir:
        match i[0]:
            case 'set':
                bash += f'{i[1]}={i[2]}\n'
            case '+':
                bash += f'{i[1]}=$((${i[1]} {i[0]} {i[2]}))\n'
    bash += 'echo $r0\n\0'
    return bash

def zeusc(command: str, env: Env) -> Tuple[str, Env]:
    tokens: List[Token] = tokenize(command)
    ast: AST = parse(tokens)
    (ir, env) = generate_ir(ast, env, 0)
    # ir = optimize_ir(ir)
    bash = generate_bash(ir)
    return (str(bash), env)

def zeus(command: str, env: Env, process) -> Tuple[str, str, Env]:
    (bash, env) = zeusc(command, env)
    # debug # return (bash + str(env), '', 0, env)
    process.stdin.write(bash)
    process.stdin.flush()
    sleep(0.01)
    output = ''
    while True:
        line = process.stdout.readline()
        if not line:
            break
        output += line
    return (output, '', env)

def main() -> int:
    env: Env = (None, set())
    process = Popen(['/bin/bash'], stdin=PIPE, stdout=PIPE, stderr=PIPE,
                    encoding='utf-8', universal_newlines=True)
    
    # Necessary boilerplate so .readline() doesn't shit itself
    if process.stdout:
        flags = fcntl(process.stdout, F_GETFL)
        fcntl(process.stdout, F_SETFL, flags | O_NONBLOCK)
    else:
        exit(1)


    while True:
        try:
            command = input('>> ') 
            (output, error, env) = zeus(command, env, process)
            if output: print(output, end='')
            if error: print(f'Error: {error}')
        except KeyboardInterrupt:
            process.terminate()
            print()
            break
    return 0

if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)

