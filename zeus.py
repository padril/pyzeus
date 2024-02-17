# Copyright 2024 Leo (padril) Peckham

from __future__ import annotations
from typing import Tuple, List, Union, Optional, Set, Callable, TypeVar, Any
from subprocess import Popen, PIPE
from fcntl import fcntl, F_GETFL, F_SETFL
from os import O_NONBLOCK
from time import sleep

SHOW_OPTIONS = {
        'ast': False,
        'tokens': False,
        'ir': False,
        'sh': False,
        }

InBase = TypeVar('InBase')
OutBase = TypeVar('OutBase')
InType = List[InBase]
OutType = List[OutBase]
CombData = Optional[Tuple[InType, OutType]]
CombFn = Callable[[CombData], CombData]

# This is some criminally undercommented shit,,, this will bite me in
# the ass later, but I don't really care right now.

# TODO: The consistency with boxing is fucking stupid here, change that shit

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
        return self.then(other).cat()

    def then_opt(self, other: Combinator) -> Combinator:
        return self.then(other).alt(self)

    def then_opt_cat(self, other: Combinator) -> Combinator:
        return self.then_cat(other).alt(self)

    def cat(self) -> Combinator:
        def cat_fn(cd: CombData) -> CombData:
            cd = self(cd)
            if cd and len(cd[1]) >= 2:
                lcomb = cd[1][-2] if isinstance(cd[1][-2], List) else [cd[1][-2]]
                rcomb = cd[1][-1] if isinstance(cd[1][-1], List) else [cd[1][-1]]
                cd = (cd[0], cd[1][0:-2] + [lcomb + rcomb])
            return cd
        return Combinator(cat_fn)

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
            while cd and cd[0]:
                last = cd
                cd = self(cd)
                if cd and len(cd[1]) >= 2:
                    lcomb = cd[1][-2] if isinstance(cd[1][-2], List) else [cd[1][-2]]
                    rcomb = cd[1][-1] if isinstance(cd[1][-1], List) else [cd[1][-1]]
                    cd = (cd[0], cd[1][0:-2] + [lcomb + rcomb])
            return last if not cd else cd
        return Combinator(any_cat_fn)

    def any(self) -> Combinator:
        def any_fn(cd: CombData) -> CombData:
            last = cd
            cd = self(cd)
            if not cd: return None
            while cd and cd[0]:
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

    def convert(self, fn: Callable[[OutType], OutBase]) -> Combinator:
        def convert_fn(cd: CombData) -> CombData:
            res = self(cd)
            if not res: return None
            appl = fn(res[1][-1])
            if not appl: return None
            return (res[0], res[1][0:-1] + [appl])
        return Combinator(convert_fn)

    def cond(self, fn: Callable[[OutType], bool]) -> Combinator:
        return self.convert(lambda x: x if fn(x) else None)

    def box(self) -> Combinator:
        return self.convert(lambda x: [x])

    def apply(self, fn: Callable[[], Combinator]) -> Combinator:
        def apply_fn(cd: CombData) -> CombData:
            if self(cd):
                ret = fn()(self(cd))
                return ret
            return None
        return Combinator(apply_fn)

    def recurse_or(self, other: Combinator) -> Combinator:
        def recurse_or_fn(cd: CombData) -> CombData:
            res = self(cd)
            return recurse_or_fn(res) if res else other(cd)
        return Combinator(recurse_or_fn)

    def recurse_or_cat(self, other: Combinator) -> Combinator:
        def recurse_or_cat_fn(cd: CombData) -> CombData:
            res = self.cat()(cd)
            return recurse_or_cat_fn(res) if res else other.cat()(cd)
        return Combinator(recurse_or_cat_fn)

    def display(self) -> Combinator:
        def display_fn(cd: CombData) -> CombData:
            print(cd)
            return self(cd)
        return Combinator(display_fn)

def cid() -> Combinator:
    def cid_fn(cd: CombData) -> CombData:
        return cd
    return Combinator(cid_fn)

def cterm[T](xs: List[T]) -> Combinator:
    def cterm_fn(cd: CombData) -> CombData:
        if not cd: return None
        if len(xs) > len(cd[0]): return None
        for i in range(len(xs)):
            if xs[i] != cd[0][i]: return None
        return (cd[0][len(xs):], cd[1] + [xs])
    return Combinator(cterm_fn)

def ctype(xs: List[str]) -> Combinator:
    def ctype_fn(cd: CombData) -> CombData:
        if not cd: return None
        if not cd[0]: return None
        if cd[0][0].t in xs:
            return (cd[0][1:], cd[1] + [[cd[0][0]]])
        else: return None
    return Combinator(ctype_fn)

def cstr(s: str) -> Combinator:
    return cterm(list(s))

def cset[T](xs: List[List[T]]) -> Combinator:
    def cset_fn(cd: CombData) -> CombData:
        for x in xs:
            res = cterm(x)(cd)
            if res: return res
        return None
    return Combinator(cset_fn)

def cdigit() -> Combinator:
    return cset(list(map(lambda x: list(str(x)), range(0, 10))))

def cwhite() -> Combinator:
    return cset([[' '], ['\t'], ['\n']])

def calpha() -> Combinator:
    return cset(list(map(
        lambda x: [str(x)],
        'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')))

def calphanum() -> Combinator:
    return cdigit().alt(calpha())

class Token[ValueType]:
    t: str
    v: ValueType
    def __init__(self, t: str, v: ValueType):
        self.t = t
        self.v = v
    def __repr__(self):
        return f'{self.t}: {self.v}' if self.v else f'{self.t}'
    def __eq__(self, other):
        return self.t == other.t and self.v == other.v

def stringify(xs: List[str]) -> str:
    if xs == []: return ''
    return ''.join(x for x in xs[0])

def tokenize(s: str) -> List[Token]:
    res: Optional[Tuple[List[str], List[List[Token]]]] = (
            # Integers
            cdigit().any_cat().then_cat(cstr('.').then_cat(cdigit().any_cat()))
            .box()
            .convert(stringify)
            .convert(lambda xs: Token('val', float(str(xs))))
            .alt(cdigit().any_cat()
                 .box()
                 .convert(stringify)
                 .convert(lambda xs: Token('val', int(str(xs)))))
            .alt(cwhite().any_cat().discard())  # Whitespace
            .alt(cset(list(map(list, sorted(['+', '-', '*', '/', '=', '->'],
                                            key=len, reverse=True))))
                 .box()
                 .convert(stringify)
                 .convert(lambda xs: Token('op', str(xs))))    # Operators
            .alt(calpha().then_opt_cat(calphanum().alt(cstr('_')).any_cat())
                 .box()
                 .convert(stringify)
                 .convert(lambda xs: Token('ident', str(xs))))
            .any()
            )((list(s), []))
    return res[1] if res else []

OP_PREC = {
        '=' :  50,
        '+' : 100,
        '*' : 200,
        }

def precedes(x, y) -> bool:
    if x[0].t != 'op' or y[0].t != 'op':
        return True
    if x[0].v not in OP_PREC:
        return False
    if y[0].v not in OP_PREC:
        return True
    return OP_PREC[x[0].v] <= OP_PREC[y[0].v]

def pol_order(xs: List) -> List:
    return [xs[1], xs[0], xs[2]]

def parse_opexpr() -> Combinator:
    return (
            ctype(['val', 'ident']).box()
            .then_cat(ctype(['op']))
            .apply(parse_expr)
            .convert(pol_order)
            .cond(lambda xs: precedes(xs, xs[2]))
            .alt(ctype(['val', 'ident']).box()
                 .then_cat(ctype(['op']))
                 .then_cat(ctype(['val', 'ident']).box())
                 .convert(pol_order)
                 .box()
                 .then_cat(ctype(['op']))
                 .apply(parse_expr)
                 .convert(pol_order))
            .alt(ctype(['val', 'ident']))
            .box()
            .cat())

def parse_fnexpr() -> Combinator:
    return (
            ctype(['ident']).any_cat().box()
            .then_cat(cterm([Token('op', '->')]))
            .apply(parse_expr)
            .convert(pol_order)
            .box()
            .cat())

def parse_callexpr() -> Combinator:
    return (
            ctype(['ident']).box()
            .then_cat(ctype(['ident', 'val']).any_cat().box())
            .convert(lambda xs: [Token('call', None), *xs]))

def parse_expr() -> Combinator:
    return (parse_fnexpr()
            .alt(parse_callexpr())
            .alt(parse_opexpr()))

AST = Token | List['AST']

def parse(tokens: List[Token]) -> AST:
    res = parse_expr().any()((tokens, []))
    if not res:
        return []
    elif isinstance(res[1], List):
        return res[1]
    else:
        return []

Env = Tuple[Optional['Env'], Set[str]]

IR = List[str | Tuple[str,...]]

FUNCTION_COUNTER = 0

def generate_ir(ast: AST, env: Env, ret: int) -> Tuple[IR, Env]:
    ir: IR = []
    local: Env = env
    if isinstance(ast, Token):
        if ast.t == 'val':
            return ([('set', f'r{ret}', str(ast.v))], env)
        elif ast.t == 'ident':
            return ([('set', f'r{ret}', f'$v{ast.v}')], env)
    elif isinstance(ast[0], Token) and ast[0].t == 'call':
        assert(isinstance(ast[1], List))
        assert(isinstance(ast[1][0], Token))
        assert(isinstance(ast[2], List))
        args = [(f'v{x.v}' if x.t == 'ident' else str(x.v))
                for x in ast[2] if isinstance(x, Token)]
        return ([('call', ast[1][0].v, ' '.join(args), f'r{ret}')],
                env)
    elif isinstance(ast[0], Token) and ast[0].t == 'op':
        if ast[0].v == '=':
            assert(isinstance(ast[1], List))
            if len(ast[1]) != 1:
                raise NotImplementedError('Cannot assign value to list')
            if isinstance(ast[1][0], List):
                raise Exception('Parse error: cannot assign to expression')
            if ast[1][0].t != 'ident':
                raise Exception('Parse error: cannot assign to rvalue')
            (rir, renv) = generate_ir(ast[2], env, ret + 1)
            return (rir + [('set', f'v{ast[1][0].v}', f'$r{ret+1}'),
                           ('set', f'r{ret}', f'$r{ret+1}')],
                    (env[0], env[1] | renv[1] | {ast[1][0].v}))
        elif ast[0].v == '->':
            assert(isinstance(ast[1], List))
            idents = [ident.v for ident in ast[1] if isinstance(ident, Token)]
            (rir, _) = generate_ir(ast[2], (env, set(idents)), 0)
            global FUNCTION_COUNTER
            FUNCTION_COUNTER += 1
            def replace_vs(x: str) -> str:
                if len(x) < 2: return x
                if x[0] == '$':
                    if x[1] == 'v':
                        if x[2:] not in idents: return x
                        return f'$a{idents.index(x[2:]) + 1}'
                elif x[0] == 'v':
                    if x[1:] not in idents: return x
                    return f'a{idents.index(x[1:]) + 1}'
                return x
            return ([('label', f'f{FUNCTION_COUNTER}', str(len(idents)))] + \
                    list(map(
                        lambda xs: tuple(map(replace_vs, xs)),
                        rir)) + \
                    [('endlabel',), ('set', f'r{ret}', f'f{FUNCTION_COUNTER}')],
                    env)
        (lir, lenv) = generate_ir(ast[1], env, ret)
        (rir, renv) = generate_ir(ast[2], env, ret + 1)
        return (lir + rir +\
                [(str(ast[0].v), f'r{ret}', f'$r{ret+1}')],
                (env[0], env[1] | lenv[1] | renv[1]))

    else:
        for node in ast:
            (rir, renv) = generate_ir(node, env, ret) 
            ir += rir;
            local = (local[0], local[1] | renv[1])

    return (ir, local)

def generate_bash(ir: IR) -> str:
    bash: str = ''
    prefix = ''
    for i in ir:
        match i[0]:
            case 'set':
                bash += f'{prefix}{i[1]}={i[2]}\n'
            case '+' | '-' | '*' | '/':
                # TODO: Implement `bc -l` for floating point values using
                # a more advanced ast mechanism/dictionary (divf vs divi etc)
                bash += f'{prefix}{i[1]}=$(bc <<< "${i[1]} {i[0]} {i[2]}")\n'
            case 'label':
                bash += f'{prefix}{i[1]} () {{\n'
                prefix += '    '
                for ident in range(1,int(i[2]) + 1):
                    bash += f'{prefix}a{ident}=${ident}\n'
            case 'endlabel':
                bash += f'{prefix}echo $r0\n'
                prefix = prefix[:-4]
                bash += '}\n'
            case 'call':
                bash += f'{i[3]}="$($v{i[1]} {i[2]})"\n'

    bash += 'echo $r0\n\0'
    return bash

def zeusc(command: str, env: Env) -> Tuple[str, Env]:
    tokens: List[Token] = tokenize(command)
    if SHOW_OPTIONS['tokens']:
        print(tokens)
    ast: AST = parse(tokens)
    if SHOW_OPTIONS['ast']:
        print(ast)
    (ir, env) = generate_ir(ast, env, 0)
    if SHOW_OPTIONS['ir']:
        print(ir)
    # ir = optimize_ir(ir)
    bash = generate_bash(ir)
    if SHOW_OPTIONS['sh']:
        print(bash)
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
            options = command.strip().split()
            if len(options) == 2 and options[0] == 'show':
                global SHOW_OPTIONS
                if options[1] not in SHOW_OPTIONS:
                    print(f'Option \'{options[1]}\' does not exist')
                    continue
                SHOW_OPTIONS[options[1]] = not SHOW_OPTIONS[options[1]]
                print(f'Set \'{options[1]}\' to {SHOW_OPTIONS[options[1]]}')
                continue
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

