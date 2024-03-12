visible_ascii = "!" ... "~" ;
whitespace    = " " | "\n" | "\t" | "\r" | "\f" | "\b" ;
character     = visible_ascii | whitespace ;
letter        = "A" ... "Z" | "a" ... "z" ;
digit         = "0" ... "9" ;

S = whitespace * -> discard ;

quantifier = "+" | "*" | "?" ;
identifier = letter 
           | letter , ( letter | digit | "_" | " " ) * ,
             ( letter | digit | "_" )
          -> lambda xs: (("IDENT", *concat(xs)),) ;

single_terminal = "\"" , ( character - "\"" | "\\\"" ) + , "\""
               -> lambda xs: (("TERM", *convert_escaped(concat(xs[1:-1])[0])),) ;

range_terminal = single_terminal , S , "..." , S, single_terminal
              -> lambda xs: (("RANGE", xs[0], xs[-1]),) ;
cut = "!" -> lambda _: (("CUT",),) ;
terminal = single_terminal | range_terminal | cut ;

paren_term  = "(", S, alternation, S, ")" -> lambda xs: xs[1:-1] ;
term = paren_term | terminal | identifier ;

quantified_expression = term, S, quantifier
                     -> lambda xs: (({"*": "ANY", "+": "MANY", "?": "OPT"}[xs[-1]], xs[0]),) ;
difference_expression = term , S , "-" , S , term
                     -> lambda xs: (("DIFF", xs[0], xs[2]),) ;
factor = quantified_expression | difference_expression | term ;

concatenation = factor , ( S, "," , S , factor ) * -> lambda xs: (("SEQ", *[x for i, x in enumerate(xs) if i % 2 == 0]),) if len(xs) > 1 else xs ;
alternation = concatenation , ( S, "|" , S , concatenation ) * -> lambda xs: (("ALT", *[x for i, x in enumerate(xs) if i % 2 == 0]),) if len(xs) > 1 else xs ;

name = identifier -> lambda xs: (("NAME", xs[0]),) ;
pattern = alternation -> lambda xs: (("PATTERN", xs[0]),) ;
conversion = ( "->" , S , ( character - ";" ) * ) ? -> lambda xs: (("CONV", *(concat(xs[2:]) if xs else ('',))),) ;

rule = name , S , "=" , S , pattern , S , conversion , ";", ! -> lambda xs: (("RULE", xs[0], xs[2], xs[3]),) ;

grammar = ( S , rule , S ) * ;
