from __future__ import annotations

from dataclasses import dataclass

from typing import Tuple, Dict, Generator, cast, Callable, \
        Sequence, Optional

from string import ascii_letters as ASCII_LETTERS, digits as ASCII_DIGITS

import re

@dataclass(frozen=True)
class Cut: ...

@dataclass(frozen=True)
class Empty: ...

@dataclass(frozen=True)
class Anything: ...

@dataclass(frozen=True)
class Terminal[T]:
    data: T

@dataclass(frozen=True)
class Alternation[T]:
    data: Tuple[T, ...]

@dataclass(frozen=True)
class Not[T]:
    data: T

type Symbol[T]  = Cut | Empty | Anything | Terminal[T] | str  # str is a key for Context
type Order[T]   = Symbol[T] | Tuple[Order[T], ...] | Alternation[Order[T]] \
        | Not[Order[T]]
type Context[T] = Dict[str, NonTerminal[T]]
type AST[T]     = Tuple[T | AST[T], ...]

@dataclass(frozen=True)
class NonTerminal[T]:
    data: Order[T]
    convert: Callable[[AST[T]], AST[T]] = lambda xs: xs

@dataclass
class Success[T]:
    ast: AST[T]
    remainder: Tuple[T, ...]
    cut: bool = False

def match[T](ctx: Context[T], xs: Tuple[T, ...], symbol: Order[T]
             ) -> Generator[Success[T], None, None]:
    """
    Uses a str to non-terminal map to match a symbol in the map to a tuple.

    expr = addexpr
    addexpr = multexpr ( '+' multexpr ) *
    multexpr = number ( '*' number ) *

    >>> number = NonTerminal(Alternation({Terminal(str(i)) for i in range(10)}))
    >>> mestar = NonTerminal(Alternation({Empty(), (Terminal('*'), 'number', 'mestar')}))
    >>> multexpr = NonTerminal(('number', 'mestar'), lambda xs: (xs,))
    >>> aestar = NonTerminal(Alternation({Empty(), (Terminal('+'), 'multexpr', 'aestar')}))
    >>> addexpr = NonTerminal(('multexpr', 'aestar'), lambda xs: (xs,))
    >>> ctx = {'number': number, 'mestar': mestar, 'multexpr': multexpr, 'aestar': aestar, 'addexpr': addexpr}
    >>> [m for m in match(ctx, tuple('1+2*3'), 'addexpr') if not m.remainder]
    [Success(ast=((('1',), '+', ('2', '*', '3')),), remainder=(), cut=False)]
    >>> [m for m in match(ctx, tuple('1*2+3'), 'addexpr') if not m.remainder]
    [Success(ast=((('1', '*', '2'), '+', ('3',)),), remainder=(), cut=False)]
    """
    match symbol:
        case Cut():
            yield Success(tuple(), xs, True)
            return
        case Empty():
            yield Success(tuple(), xs)
        case Anything():
            yield Success((xs[0],), tuple(r) if (r := xs[1:]) else tuple())
        case Terminal():
            if xs and xs[0] == symbol.data:
                yield Success((xs[0],), tuple(r) if (r := xs[1:]) else tuple())
        case tuple():
            if not symbol: yield Success(tuple(), xs)  # Empty
            car, *cdr = symbol
            cdr = tuple(cdr)
            if not cdr: cdr = Empty()
            # try to match car
            for car_match in match(ctx, xs, car):
                # try to match cdr
                for cdr_match in match(ctx, car_match.remainder, cdr):
                    # we need the cast here, because the concatenation leads to
                    # some weird pyright stuff
                    ast = cast(AST[T], car_match.ast + cdr_match.ast)
                    # if there's a cut, don't try the other matches
                    yield Success(ast, cdr_match.remainder,
                                  car_match.cut or cdr_match.cut)
                    if car_match.cut or cdr_match.cut: return
        case Alternation():
            for s in symbol.data:  # cast to iterate over Alternation
                for m in match(ctx, xs, s):
                    yield m
        case Not():
            for m in match(ctx, xs, symbol.data):
                return
            yield Success(tuple(), xs)
        case str():
            nt = ctx[symbol]
            for m in match(ctx, xs, nt.data):
                m.ast = nt.convert(m.ast)
                yield m
                if m.cut: return

def find[T](ctx: Context[T], xs: Sequence[T], symbol: Order[T]
            ) -> Optional[AST[T]]:
    for m in match(ctx, tuple(xs), symbol):
        if not m.remainder: return m.ast
    return None

def wrap(xs: AST[str]) -> Tuple[AST[str]]: return (xs,)
def discard(_) -> Tuple: return tuple()
def concat(xs: str | AST[str]) -> Tuple[str]:
    if isinstance(xs, str):
        return (xs,)
    elif isinstance(xs, Tuple):
        return (''.join(concat(x)[0] for x in xs),)
def convert_escaped(s: str) -> str:
        ESCAPED = {
                '\\n': '\n',
                '\\r': '\r',
                '\\t': '\t',
                '\\f': '\f',
                '\\b': '\b',
                }
        for key, val in ESCAPED.items():
            s = s.replace(key, val)
        from re import sub as resub
        return resub(r'\\(.)', r'\1', s)

def parse_daphne(s: str, compiled: str) -> Optional[AST[str]]:
    """
    This uses Daphne, a personalized form of EBNF, with:
     - ellipses for ranges
     - ! to cut (stop backtracing)
     - and additional field to describe the convert function
        - several default functions exist, like `wrap`, which wraps the AST in
          a tuple, and `discard` which returns an empty tuple
     - removal of "alternates" used by standard EBNF
     - removal of {} and [] for repetition and optionality, respectively
    
    Definition in `cf.daphne`
    ```

    >>> parse_daphne('expr = ( "(", expr, ")" | "a" ... "z" ) * -> lambda m: concat(m) ;', 'arrows.do')
    (('RULE', ('NAME', ('IDENT', 'expr')), ('PATTERN', ('ANY', ('ALT', ('SEQ', ('TERM', '('), ('IDENT', 'expr'), ('TERM', ')')), ('RANGE', ('TERM', 'a'), ('TERM', 'z'))))), ('CONV', 'lambda m: concat(m) ')),)
    """

    ctx = {}
    with open(compiled, 'r') as f:
        ctx = compile_ast(eval(f.read()))

    grammar = find(ctx, s, 'grammar')
    
    return grammar

def compile_ast[T](grammar: AST[str]) -> Context[T]:
    """
    >>> grammar = parse_daphne('expr = ( "(", expr, ")" | "a" ... "z" ) * -> lambda m: concat(m);', 'arrows.do')
    >>> ctx = compile_ast(grammar)
    >>> [m for m in match(ctx, '((a)b)(cd)()', 'expr') if not m.remainder]
    [Success(ast=('((a)b)(cd)()',), remainder=(), cut=False)]
    >>> [m for m in match(ctx, '(()aaaa', 'expr') if not m.remainder]
    []
    """
    context = {}
    def compile_expr(ast: AST[str], prefix: str, suffix: int
                     ) -> Tuple[Context[T], int]:  # next usable suffix
        start_suffix = suffix
        suffix += 1
        ctx = {}
        t, *e = ast
        match t:
            case 'CUT':
                ctx[f'{prefix}:{start_suffix}'] = NonTerminal(Cut())
            case 'ANY':
                assert(not isinstance(e[0], str))
                nt = NonTerminal(Alternation(((f'{prefix}:{suffix}',
                                             f'{prefix}:{start_suffix}'),
                                            Empty())))
                ctx, suffix = compile_expr(e[0], prefix, suffix)
                ctx[f'{prefix}:{start_suffix}'] = nt
            case 'MANY':
                assert(not isinstance(e[0], str))
                nt = NonTerminal(Alternation(((f'{prefix}:{suffix}',
                                              f'{prefix}:{start_suffix}'),
                                             f'{prefix}:{suffix}')))
                ctx, suffix = compile_expr(e[0], prefix, suffix)
                ctx[f'{prefix}:{start_suffix}'] = nt
            case 'OPT':
                assert(not isinstance(e[0], str))
                nt = NonTerminal(Alternation((f'{prefix}:{suffix}', Empty())))
                ctx, suffix = compile_expr(e[0], prefix, suffix)
                ctx[f'{prefix}:{start_suffix}'] = nt
            case 'DIFF':
                assert(not isinstance(e[0], str) and
                       not isinstance(e[1], str))
                success_suffix = suffix
                success, suffix = compile_expr(e[0], prefix, suffix)
                failure_suffix = suffix
                failure, suffix = compile_expr(e[1], prefix, suffix)
                nt = NonTerminal((Not(f'{prefix}:{failure_suffix}'),
                                  f'{prefix}:{success_suffix}'))
                ctx = success | failure | ctx
                ctx[f'{prefix}:{start_suffix}'] = nt
            case 'ALT':
                alt = []
                for option in e:
                    assert(not isinstance(option, str))
                    alt = alt + [f'{prefix}:{suffix}']
                    c, suffix = compile_expr(option, prefix, suffix)
                    ctx = c | ctx
                ctx[f'{prefix}:{start_suffix}'] = NonTerminal(Alternation(tuple(alt)))
            case 'SEQ':
                seq = []
                for option in e:
                    assert(not isinstance(option, str))
                    seq = seq + [f'{prefix}:{suffix}']
                    c, suffix = compile_expr(option, prefix, suffix)
                    ctx = c | ctx
                ctx[f'{prefix}:{start_suffix}'] = NonTerminal(tuple(seq))
            case 'RANGE':
                (_, start), (_, end) = e
                assert(isinstance(start, str) and isinstance(end, str))
                nt = NonTerminal(
                        Alternation(tuple(Terminal(chr(c))
                                          for c in range(ord(start), ord(end) + 1))))
                ctx[f'{prefix}:{start_suffix}'] = nt
            case 'TERM':
                seq = tuple(Terminal(i) for i in e)
                seq = seq[0] if len(seq) == 1 else seq
                ctx[f'{prefix}:{start_suffix}'] = NonTerminal(seq)
            case 'IDENT':
                ctx[f'{prefix}:{start_suffix}'] = NonTerminal(e[0])
        return ctx, suffix


    for rule in grammar: 
        RULE, (NAME, (IDENT, name)), (PATTERN, expr), (CONV, func) = rule
        assert(RULE == 'RULE')

        assert(NAME == 'NAME' and IDENT == 'IDENT')
        assert(isinstance(name, str))

        assert(CONV == 'CONV')
        if func:
            assert(isinstance(func, str))
            func = eval(func)  # TODO : an eval is dangerous! there should be
                               # more checking here, but this is fine since
                               # it's just a prototype
        else:
            func = lambda x: x

        assert(PATTERN == 'PATTERN')
        assert(not isinstance(expr, str))
        c, _ = compile_expr(expr, name, 0)
        context = context | c
        context[name] = NonTerminal(f'{name}:0', func)

    return context

def selfhost(compiled: str, inpath: str, outpath: str) -> None:
    with open(inpath, 'r') as infile, open(outpath, 'w') as outfile:
        s = infile.read()
        daphne = parse_daphne(s, compiled)
        print(daphne, file=outfile)

def make_argparser():
    argparser = argparse.ArgumentParser('pydaphne')

    argparser.add_argument(
            'compiled',
            help='A path to the compiled *.do file to be used for the parse.',
            type=str)
    argparser.add_argument(
            'input',
            help='A path to a *.d file describing a grammar to be compiled.',
            type=str)
    argparser.add_argument(
            'output',
            help='A path to the compiled *.do file to be created.',
            type=str)

    return argparser

if __name__ == '__main__':
    import doctest
    doctest.testmod()

    import argparse
    argparser = make_argparser()
    args = argparser.parse_args()

    selfhost(args.compiled, args.input, args.output)

