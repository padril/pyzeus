from __future__ import annotations

from dataclasses import dataclass

from typing import Tuple, FrozenSet, Dict, Generator, cast

@dataclass(frozen=True)
class Empty: ...

@dataclass(frozen=True)
class Terminal[T]:
    data: T

type Symbol[T]  = Empty | Terminal[T] | str  # str is a key for Context
type Order[T]   = Symbol[T] | Tuple[Order[T]] | FrozenSet[Order[T]]
type Context[T] = Dict[str, NonTerminal[T]]
type AST[T]     = Tuple[T | AST[T]]

@dataclass(frozen=True)
class NonTerminal[T]:
    data: Order[T]

@dataclass
class Success[T]:
    ast: AST[T]
    remainder: Tuple[T]

def match[T](ctx: Context[T], xs: Tuple[T], symbol: Order[T]
             ) -> Generator[Success[T], None, None]:
    """
    Uses a str to non-terminal map to match a symbol in the map to a tuple.

    expr = addexpr
    addexpr = multexpr ( '+' multexpr ) *
    multexpr = number ( '*' number ) *

    >>> number = NonTerminal(frozenset({Terminal(str(i)) for i in range(10)}))
    >>> mestar = NonTerminal(frozenset({Empty(), (Terminal('*'), 'number', 'mestar')}))
    >>> multexpr = NonTerminal(('number', 'mestar'))
    >>> aestar = NonTerminal(frozenset({Empty(), (Terminal('+'), 'multexpr', 'aestar')}))
    >>> addexpr = NonTerminal(('multexpr', 'aestar'))
    >>> ctx = {'number': number, 'mestar': mestar, 'multexpr': multexpr, 'aestar': aestar, 'addexpr': addexpr}
    >>> [m for m in match(ctx, tuple('1+2*3'), 'addexpr') if not m.remainder]
    [Success(ast=(((('1',), ()), ('+', (('2',), ('*', ('3',), ())), ())),), remainder=())]
    >>> [m for m in match(ctx, tuple('1*2+3'), 'addexpr') if not m.remainder]
    [Success(ast=(((('1',), ('*', ('2',), ())), ('+', (('3',), ()), ())),), remainder=())]
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
            for m in match(ctx, xs, ctx[symbol].data):
                m.ast = (m.ast,)
                yield m

# def find[T](ctx: Context[T], xs: Sequence[T], symbol: Order[T]) -> 

if __name__ == '__main__':
    import doctest
    doctest.testmod()

