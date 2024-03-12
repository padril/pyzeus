visible_ascii = "!" ... "~" ;
whitespace    = " " | "\n" | "\t" | "\r" | "\f" | "\b" ;
character     = visible_ascii | whitespace ;
letter        = "A" ... "Z" | "a" ... "z" ;
digit         = "0" ... "9" ;

S { discard } = whitespace * ;

quantifier = "+" | "*" | "?" ;
identifier { lambda xs: (("IDENT", *concat(xs)),) }
           = letter 
           | letter , ( letter | digit | "_" | " " ) * ,
             ( letter | digit | "_" ) ;

single_terminal { lambda xs: (("TERM", *convert_escaped(concat(xs[1:-1])[0])),) }
                = "\"" , ( character - "\"" | "\\\"" ) + , "\"" ;
range_terminal { lambda xs: (("RANGE", xs[0], xs[-1]),) }
                = single_terminal , S , "..." , S, single_terminal ;
cut { lambda _: (("CUT",),) }
                = "!" ;
terminal        = single_terminal | range_terminal | cut ;

paren_term { lambda xs: xs[1:-1] } = "(", S, alternation, S, ")" ;
term = paren_term | terminal | identifier ;

quantified_expression { lambda xs: (({"*": "ANY", "+": "MANY", "?": "OPT"}[xs[-1]], xs[0]),) }
                           = term, S, quantifier ;
difference_expression { lambda xs: (("DIFF", xs[0], xs[2]),) }
                           = term , S , "-" , S , term ; 
factor = quantified_expression | difference_expression | term ;

concatenation { lambda xs: (("SEQ", *[x for i, x in enumerate(xs) if i % 2 == 0]),) if len(xs) > 1 else xs } = factor , ( S, "," , S , factor ) * ;
alternation { lambda xs: (("ALT", *[x for i, x in enumerate(xs) if i % 2 == 0]),) if len(xs) > 1 else xs }= concatenation , ( S, "|" , S , concatenation ) * ;

name { lambda xs: (("NAME", xs[0]),) } = identifier ;
pattern { lambda xs: (("PATTERN", xs[0]),) } = alternation ;
conversion { lambda xs: (("CONV", *(concat(xs[2:]) if xs else ('',))),) }
           = ( "->" , S , ( character - ";" ) * ) ? ;

rule { lambda xs: (("RULE", xs[0], xs[2], xs[3]),) }
     = name , S , "=" , S , pattern , S , conversion , ";", ! ;

grammar = ( S , rule , S ) * ;
