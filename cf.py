from __future__ import annotations

from dataclasses import dataclass

from typing import Tuple, FrozenSet, Dict, Generator, cast, Callable, \
        Sequence, Optional

from string import ascii_letters as ASCII_LETTERS, digits as ASCII_DIGITS

@dataclass(frozen=True)
class Empty: ...

@dataclass(frozen=True)
class Terminal[T]:
    data: T

type Symbol[T]  = Empty | Terminal[T] | str  # str is a key for Context
type Order[T]   = Symbol[T] | Tuple[Order[T], ...] | FrozenSet[Order[T]]
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

def match[T](ctx: Context[T], xs: Tuple[T, ...], symbol: Order[T]
             ) -> Generator[Success[T], None, None]:
    """
    Uses a str to non-terminal map to match a symbol in the map to a tuple.

    expr = addexpr
    addexpr = multexpr ( '+' multexpr ) *
    multexpr = number ( '*' number ) *

    >>> number = NonTerminal(frozenset({Terminal(str(i)) for i in range(10)}))
    >>> mestar = NonTerminal(frozenset({Empty(), (Terminal('*'), 'number', 'mestar')}))
    >>> multexpr = NonTerminal(('number', 'mestar'), lambda xs: (xs,))
    >>> aestar = NonTerminal(frozenset({Empty(), (Terminal('+'), 'multexpr', 'aestar')}))
    >>> addexpr = NonTerminal(('multexpr', 'aestar'), lambda xs: (xs,))
    >>> ctx = {'number': number, 'mestar': mestar, 'multexpr': multexpr, 'aestar': aestar, 'addexpr': addexpr}
    >>> [m for m in match(ctx, tuple('1+2*3'), 'addexpr') if not m.remainder]
    [Success(ast=((('1',), '+', ('2', '*', '3')),), remainder=())]
    >>> [m for m in match(ctx, tuple('1*2+3'), 'addexpr') if not m.remainder]
    [Success(ast=((('1', '*', '2'), '+', ('3',)),), remainder=())]
    """
    match symbol:
        case Empty():
            yield Success(tuple(), xs)
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
                    yield Success(ast, cdr_match.remainder)
        case frozenset():
            for s in tuple(symbol):  # cast to iterate over frozenset
                for m in match(ctx, xs, s):
                    yield m
        case str():
            nt = ctx[symbol]
            for m in match(ctx, xs, nt.data):
                m.ast = nt.convert(m.ast)
                yield m

def wrap(xs: AST[str]) -> Tuple[AST[str]]: return (xs,)
def discard(_) -> Tuple: return tuple()
def concat(xs: str | AST[str]) -> Tuple[str]:
    if isinstance(xs, str):
        return (xs,)
    elif isinstance(xs, Tuple):
        return (''.join(concat(x)[0] for x in xs),)

def parse_ebnf(s: str) -> Optional[AST[str]]:
    """
    This uses a personalized form of EBNF, with:
     - ellipses for ranges
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
    range_terminal  = single_terminal , "..." , single_terminal ;
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

    >>> parse_ebnf('expr {concat} = ( "(", expr, ")" | "a" ... "z" ) * ;')
    ('GRAMMAR', ('RULE', ('IDENT', 'expr'), ('BLOCK', 'concat'), ('ANY', ('ALT', ('SEQ', ('TERM', '('), ('IDENT', 'expr'), ('TERM', ')')), ('RANGE', ('TERM', 'a'), ('TERM', 'z'))))))
    """

    visible_ascii_fs = frozenset(Terminal(c) for c in map(chr, range(33, 127)))
    visible_ascii = NonTerminal(visible_ascii_fs)

    whitespace_fs = frozenset(Terminal(c) for c in ' \n\t\r\f\b')
    whitespace = NonTerminal(whitespace_fs)

    character_fs = visible_ascii_fs | whitespace_fs
    character = NonTerminal(character_fs)

    letter_fs = frozenset({Terminal(c) for c in ASCII_LETTERS})
    letter = NonTerminal(letter_fs)

    digit_fs = frozenset({Terminal(c) for c in ASCII_DIGITS})
    digit = NonTerminal(digit_fs)

    S_fs = frozenset({(whitespace_fs, 'S'), Empty()})
    S = NonTerminal(S_fs, discard)

    quantifier_fs = frozenset(Terminal(c) for c in '*+?')
    quantifier = NonTerminal(quantifier_fs)

    _ident_star = NonTerminal(frozenset(
        {(letter_fs | digit_fs | frozenset({Terminal('_'), Terminal(' ')}),
          '_ident_star'), Empty()}))
    identifier = NonTerminal(
            frozenset({
                letter_fs, (
                    letter_fs, '_ident_star',
                    letter_fs | digit_fs | frozenset({Terminal('_')}))}),
                lambda xs: (('IDENT', *concat(xs)),))

    _non_single_quote_star = NonTerminal(
            frozenset({(character_fs - frozenset({Terminal("'")}),
                                                 '_non_single_quote_star'),
                       Empty()}))
    _non_double_quote_star = NonTerminal(
            frozenset({(character_fs - frozenset({Terminal('"')}),
                                                 '_non_double_quote_star'),
                       Empty()}))
    single_terminal = NonTerminal(
            frozenset(
                {(Terminal("'"), '_non_single_quote_star', Terminal("'")),
                 (Terminal('"'), '_non_double_quote_star', Terminal('"'))}),
                lambda xs: (('TERM', *(xs[1:-1])),))

    range_terminal = NonTerminal(('single_terminal', 'S',
                                  tuple(Terminal(c) for c in '...'), 'S',
                                  'single_terminal'),
                                 lambda xs: (('RANGE', xs[0], xs[-1]),))
    
    terminal = NonTerminal(frozenset({'single_terminal', 'range_terminal'}))

    _paren_term = NonTerminal((Terminal('('), 'S', 'rhs', 'S', Terminal(')')),
                              lambda xs: (xs[1:-1],))
    term_fs = frozenset({'_paren_term', 'terminal', 'identifier'})
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

    factor_expression_fs = frozenset(
            {'_quantified_expression',
             ('term', 'S', Terminal('-'), 'S', 'term')})
    factor_expression = NonTerminal(factor_expression_fs, wrap)

    factor = NonTerminal(frozenset({'factor_expression', 'term'}))

    _concatenation_star = NonTerminal(
            frozenset({('S', Terminal(','), 'S', 'factor',
                        '_concatenation_star'), Empty()}),
            lambda xs: (xs[1], *(xs[2:])) if len(xs) > 1 else tuple())
    concatenation = NonTerminal(('factor', '_concatenation_star'),
                                lambda xs: (('SEQ', *xs),) if len(xs) > 1 else xs)

    _alternation_star = NonTerminal(
            frozenset({('S', Terminal('|'), 'S', 'concatenation',
                        '_alternation_star'), Empty()}),
            lambda xs: (xs[1], *(xs[2:])) if len(xs) > 1 else tuple())
    alternation = NonTerminal(('concatenation', '_alternation_star'),
                              lambda xs: ('ALT', *xs) if len(xs) > 1 else xs)

    _block_star = NonTerminal(
            frozenset({
                (character_fs - frozenset({Terminal('{'), Terminal('}')}),
                 '_block_star'),
                'block',
                Empty()}))
    block = NonTerminal((Terminal('{'), '_block_star', Terminal('}')),
                        lambda xs: (('BLOCK', concat(xs[1:-1])[0]),))

    rhs = NonTerminal('alternation')

    lhs = NonTerminal(frozenset({'identifier', ('identifier', 'S', 'block')}),
                      lambda xs: (xs[0:2], *(xs[2:])))

    rule = NonTerminal(('S', 'lhs', 'S', Terminal('='), 'S', 'rhs', 'S',
                        Terminal(';'), 'S'),
                       lambda xs: (('RULE', *(xs[0]), xs[2]),))

    grammar = NonTerminal(frozenset({('rule', 'grammar'), Empty()}))

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
            'terminal': terminal,
            '_paren_term': _paren_term,
            'term_fs': term_fs,
            'term': term,
            '_quantified_expression': _quantified_expression,
            'factor_expression_fs': factor_expression_fs,
            'factor_expression': factor_expression,
            'factor': factor,
            '_concatenation_star': _concatenation_star,
            'concatenation': concatenation,
            '_alternation_star': _alternation_star,
            'alternation': alternation,
            '_block_star': _block_star,
            'block': block,
            'rhs': rhs,
            'lhs': lhs,
            'rule': rule,
            'grammar': grammar,
            }

    result = find(ctx, s, 'grammar')
    
    return ('GRAMMAR', *result) if result else result

def txt_to_context[T](filename: str) -> Context[T]:
    ...
    """
    ctx = {}
    with open(filename) as f:
        for line in f:
            name, nt = str_to_non_terminal(line)
            ctx[name] = nt
    return ctx
    """


def find[T](ctx: Context[T], xs: Sequence[T], symbol: Order[T]
            ) -> Optional[AST[T]]:
    matches = [m for m in match(ctx, tuple(xs), symbol) if not m.remainder]
    return matches[0].ast if matches else None


if __name__ == '__main__':
    import doctest
    doctest.testmod()

