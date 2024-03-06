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
        return s.replace('\\', '')

def parse_daphne(s: str) -> Optional[AST[str]]:
    """
    This uses Daphne, a personalized form of EBNF, with:
     - ellipses for ranges
     - ! to cut (stop backtracing)
     - and additional field to describe the convert function
        - several default functions exist, like `wrap`, which wraps the AST in
          a tuple, and `discard` which returns an empty tuple
     - removal of "alternates" used by standard EBNF
     - removal of {} and [] for repetition and optionality, respectively

    Definition (minus all the correct functions):

    ```
    visible_ascii = "!" ... "~" ;
    whitespace    = " " | "\n" | "\t" | "\r" | "\f" | "\b" ;
    character     = visible_ascii | whitespace ;
    letter        = "A" ... "Z" | "a" ... "z" ;
    digit         = "0" ... "9" ;

    S { discard } = whitespace * ;

    quantifier = "+" | "*" | "?" | "!" ;
    identifier = letter 
               | letter , ( letter | digit | "_" | " " ) * ,
                 ( letter | digit | "_" ) ;

    single_terminal = "'" , ( character - "'" ) + , "'"
                    | '"' , ( character - '"' ) + , '"' ;
    range_terminal  = single_terminal , S , "..." , S , single_terminal ;
    terminal        = single_terminal | range_terminal

    term { wrap } = "(" , S , rhs , S , ")" | terminal | identifier ;

    factor_expression { wrap } = term , S , quantifier
                               | term , S , "-" , S , term
    factor = factor_expression | term

    concatenation { wrap } = factor , ( S, "," , S , factor ) * ;
    alternation = concatenation , ( S, "|" , S , concatenation ) * ;

    block { wrap } = "{" , ( character - ( "{" | "}" ) | block ) * , "}" ;

    rhs { wrap } = alternation ;
    lhs { wrap } = identifier , ( S , block ) ?;

    rule { wrap } = lhs , S , "=" , S , rhs , S , ";" ;

    grammar = ( S , rule , S ) * ;
    ```

    >>> parse_daphne('expr {concat} = ( "(", expr, ")" | "a" ... "z" ) * ;')
    (('RULE', ('NAME', ('IDENT', 'expr')), ('FUNC', ('BLOCK', 'concat')), ('ANY', ('ALT', ('SEQ', ('TERM', '('), ('IDENT', 'expr'), ('TERM', ')')), ('RANGE', ('TERM', 'a'), ('TERM', 'z'))))),)
    """

    visible_ascii_fs = tuple(Terminal(c) for c in map(chr, range(33, 127)))
    visible_ascii = NonTerminal(Alternation(visible_ascii_fs))

    whitespace_fs = tuple(Terminal(c) for c in ' \n\t\r\f\b')
    whitespace = NonTerminal(Alternation(whitespace_fs))

    character_fs = visible_ascii_fs + whitespace_fs
    character = NonTerminal(Alternation(character_fs))

    letter_fs = tuple(Terminal(c) for c in ASCII_LETTERS)
    letter = NonTerminal(Alternation(letter_fs))

    digit_fs = tuple(Terminal(c) for c in ASCII_DIGITS)
    digit = NonTerminal(Alternation(digit_fs))

    S_fs = ((Alternation(whitespace_fs), 'S'), Empty())
    S = NonTerminal(Alternation(S_fs), discard)

    quantifier_fs = tuple(Terminal(c) for c in '*+?')
    quantifier = NonTerminal(Alternation(quantifier_fs))

    _ident_star = NonTerminal(Alternation((
        (Alternation((letter_fs + digit_fs + (Terminal('_'), Terminal(' ')))),
         '_ident_star'), Empty())))
    identifier = NonTerminal(
            Alternation((
                Alternation(letter_fs), (
                    Alternation(letter_fs), '_ident_star',
                    Alternation(letter_fs + digit_fs + (Terminal('_'),))))),
                lambda xs: (('IDENT', *concat(xs)),))

    _non_single_quote_star = NonTerminal(
            Alternation((
                (Alternation(
                    tuple(el for el in character_fs
                          if el != Terminal("'"))),
                 '_non_single_quote_star'),
                (Terminal('\\'), Anything(), '_non_single_quote_star'),
                Empty())))
    _non_double_quote_star = NonTerminal(
            Alternation((
                (Alternation(
                    tuple(el for el in character_fs
                          if el != Terminal('"'))),
                 '_non_double_quote_star'),
                (Terminal('\\'), Anything(), '_non_double_quote_star'),
                Empty())))
    
    single_terminal = NonTerminal(
            Alternation(
                ((Terminal("'"), '_non_single_quote_star', Terminal("'")),
                 (Terminal('"'), '_non_double_quote_star', Terminal('"')))),
                lambda xs: (('TERM', *convert_escaped(concat(xs[1:-1])[0])),))

    range_terminal = NonTerminal(('single_terminal', 'S',
                                  tuple(Terminal(c) for c in '...'), 'S',
                                  'single_terminal'),
                                 lambda xs: (('RANGE', xs[0], xs[-1]),))

    cut = NonTerminal(Terminal('!'), lambda _: (('CUT',),))
    
    terminal = NonTerminal(Alternation(('single_terminal', 'range_terminal',
                                        'cut')))

    _paren_term = NonTerminal((Terminal('('), 'S', 'rhs', 'S', Terminal(')')),
                              lambda xs: xs[1:-1])
    term_fs = Alternation(('_paren_term', 'terminal', 'identifier'))
    term = NonTerminal(term_fs)

    QUANT_MAP = {
            '*': 'ANY',
            '+': 'MANY',
            '?': 'OPT',
            }
    _quantified_expression = NonTerminal(
            ('term', 'S', 'quantifier'),
            lambda xs: (QUANT_MAP[xs[1]], xs[0]) if isinstance(xs[1],
                                                               str) else xs)

    _difference_expression = NonTerminal(
            ('term', 'S', Terminal('-'), 'S', 'term'),
            lambda xs: ('DIFF', xs[0], xs[2]))

    factor_expression_fs = Alternation(
            ('_quantified_expression',
             '_difference_expression'))
    factor_expression = NonTerminal(factor_expression_fs, wrap)

    factor = NonTerminal(Alternation(('factor_expression', 'term')))

    _concatenation_star = NonTerminal(
            Alternation((('S', Terminal(','), 'S', 'factor',
                        '_concatenation_star'), Empty())),
            lambda xs: (xs[1], *(xs[2:])) if len(xs) > 1 else tuple())
    concatenation = NonTerminal(('factor', '_concatenation_star'),
                                lambda xs: (('SEQ', *xs),)
                                if len(xs) > 1 else xs)

    _alternation_star = NonTerminal(
            Alternation((('S', Terminal('|'), 'S', 'concatenation',
                        '_alternation_star'), Empty())),
            lambda xs: (xs[1], *(xs[2:])) if len(xs) > 1 else tuple())
    alternation = NonTerminal(('concatenation', '_alternation_star'),
                              lambda xs: (('ALT', *xs),)
                              if len(xs) > 1 else xs)

    _block_star = NonTerminal(
            Alternation((
                (Alternation(
                    tuple(el for el in character_fs
                          if el not in {Terminal('{'), Terminal('}')}) + \
                                  ('block',)),
                 '_block_star'),
                Empty())))
    block = NonTerminal((Terminal('{'), '_block_star', Terminal('}')))

    outer_block = NonTerminal('block',
                        lambda xs: (('BLOCK', concat(xs[1:-1])[0]),))

    rhs = NonTerminal('alternation')

    lhs = NonTerminal(Alternation(('identifier', ('identifier', 'S', 'outer_block'))),
                      lambda xs: (('NAME', xs[0]), ('FUNC', *(xs[1:]))))

    rule = NonTerminal(('S', 'lhs', 'S', Terminal('='), 'S', 'rhs', 'S',
                        Terminal(';'), Cut(), 'S'),
                       lambda xs: (('RULE', xs[0], xs[1], xs[3]),))

    grammar = NonTerminal(Alternation((('rule', 'grammar'), Empty())))

    ctx = {
            'visible_ascii_fs': visible_ascii_fs,
            'visible_ascii': visible_ascii,
            'whitespace_fs': whitespace_fs,
            'whitespace': whitespace,
            'character_fs': character_fs,
            'character': character,
            'letter_fs': letter_fs,
            'letter': letter,
            'digit_fs': digit_fs,
            'digit': digit,
            'S_fs': S_fs,
            'S': S,
            'quantifier_fs': quantifier_fs,
            'quantifier': quantifier,
            '_ident_star': _ident_star,
            'identifier': identifier,
            '_non_single_quote_star': _non_single_quote_star,
            '_non_double_quote_star': _non_double_quote_star,
            'single_terminal': single_terminal,
            'range_terminal': range_terminal,
            'cut': cut,
            'terminal': terminal,
            '_paren_term': _paren_term,
            'term_fs': term_fs,
            'term': term,
            '_quantified_expression': _quantified_expression,
            '_difference_expression': _difference_expression,
            'factor_expression_fs': factor_expression_fs,
            'factor_expression': factor_expression,
            'factor': factor,
            '_concatenation_star': _concatenation_star,
            'concatenation': concatenation,
            '_alternation_star': _alternation_star,
            'alternation': alternation,
            '_block_star': _block_star,
            'block': block,
            'outer_block': outer_block,
            'rhs': rhs,
            'lhs': lhs,
            'rule': rule,
            'grammar': grammar,
            }

    grammar = find(ctx, s, 'grammar')
    
    return grammar

def compile_ast[T](grammar: AST[str]) -> Context[T]:
    """
    >>> grammar = parse_daphne('expr { concat } = ( "(", expr, ")" | "a" ... "z" ) * ;')
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
        RULE, (NAME, (IDENT, name)), (FUNC, *block), expr = rule
        assert(RULE == 'RULE')

        assert(NAME == 'NAME' and IDENT == 'IDENT')
        assert(isinstance(name, str))

        assert(FUNC == 'FUNC')
        if block:
            BLOCK, func = block[0]
            assert(BLOCK == 'BLOCK' and isinstance(func, str))
            func = eval(func)  # TODO : an eval is dangerous! there should be
                               # more checking here, but this is fine since
                               # it's just a prototype
        else:
            func = lambda x: x

        assert(not isinstance(expr, str))
        c, _ = compile_expr(expr, name, 0)
        context = context | c
        context[name] = NonTerminal(f'{name}:0', func)

    return context

def selfhost(filename: str) -> None:
    with open(filename, 'r') as f, open('compiled.txt', 'w') as compiled:
        s = f.read()
        daphne = parse_daphne(s)
        print(daphne, file=compiled)

if __name__ == '__main__':
    import doctest
    doctest.testmod()

    selfhost('cf.daphne')

