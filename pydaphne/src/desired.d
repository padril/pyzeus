// These aren't in the correct form because there's no standard
// library for this implementation
dialect python;

use ascii.d; 
use parser.py as p;

// The python implementation requires returns to be tuples
blank = ascii.whitespace * -> tuple() ;

// Group capture names are slightly different, using the Python
// variable m (for match), instead of '
identifier = ( ascii.letter | "_" ) ,
             ( ascii.letter | ascii.digit | "_" ) *
          -> ("IDENT", "".join(*m)) ;

single_terminal = "\"" , [ ( ascii.visible - "\"" ) + ] , "\""
               -> ("TERM", "".join(*m)) ;

// By default, the sequence operator pretty much just concatenates
// the results of each match
range_terminal = [ single_terminal ] , S , "..." , S ,
                 [ single_terminal ]
              -> ("RANGE",) + m ;

cut = "!" -> ("CUT",) ;

terminal = single_terminal | range_terminal | cut ;

paren_term = "(" , S , [ pattern ] , S , ")" ;

term = paren_term | terminal | identifier ;

quantifier = "+" | "*" | "?" -> p.map_quantifiers *m ;

quantified_expression = [ term ] , S , [ quantifier ]
                     -> (m[1][0], m[0]) ;

difference_expression = [ term ] , S , "-" , S , [ term ] 
                     -> ("DIFF",) + '; 

factor = quantified_expression | difference_expression | term ;

concatenation = [ factor ] , ( S , "," , S , [ factor ] ) *
             -> *m if length m > 1 else ("SEQ",) + m ;
             
alternation = [ concatenation ] , ( S , "|" , S , [ concatenation ] ) * 
           -> *m if length m > 1 else ("ALT",) + m ;

name = identifier -> ("NAME", m) ;

pattern = alternation -> ("PATTERN", m) ;

conversion = ascii.any - ";" * -> ("CONV", "".join(*m)) ;

rule  = [ name ] , S , "=",  ! , S ,
        [ pattern ] , S , "->" , ! , S ,
        [ conversion ] , S, ";" , ! -> m;

grammar = ( S , rule , S ) * ;
