#!/usr/bin/env python
"""
    Reynir: Natural language processing for Icelandic

    Parser test module

    Author: Vilhjalmur Thorsteinsson

    This software is at a very early development stage.
    While that is the case, it is:
    Copyright (c) 2015 Vilhjalmur Thorsteinsson
    All rights reserved
    See the accompanying README.md file for further licensing and copyright information.

"""

import time
from contextlib import closing


# Import the Psycopg2 connector for PostgreSQL
try:
    # For CPython
    import psycopg2.extensions as psycopg2ext
    import psycopg2
except ImportError:
    # For PyPy
    import psycopg2cffi.extensions as psycopg2ext
    import psycopg2cffi as psycopg2

# Make Psycopg2 and PostgreSQL happy with UTF-8
psycopg2ext.register_type(psycopg2ext.UNICODE)
psycopg2ext.register_type(psycopg2ext.UNICODEARRAY)


from tokenizer import TOK, tokenize
from grammar import Nonterminal, Terminal, Token, Production, Grammar, GrammarError
from parser import Parser, ParseError, ParseForestPrinter
from binparser import BIN_Parser
from fastparser import Fast_Parser
from settings import Settings, ConfigError


class Test_DB:

    """ Encapsulates a database of test sentences and results """

    def __init__(self):
        """ Initialize DB connection instance """
        self._conn = None # Connection
        self._c = None # Cursor

    @classmethod
    def open_db(cls):
        """ Return an open instance of the database on the default host """
        return cls().open(Settings.DB_HOSTNAME)

    def open(self, host):
        """ Open and initialize a database connection """
        self._conn = psycopg2.connect(dbname="test", user="reynir", password="reynir",
            host=host, client_encoding="utf8")
        if not self._conn:
            raise Exception("Unable to open connection to database at host " + host)
        # Ask for automatic commit after all operations
        # We're doing only reads, so this is fine and makes things less complicated
        self._conn.autocommit = True
        self._c = self._conn.cursor()
        return None if self._c is None else self

    def close(self):
        """ Close the DB connection and the associated cursor """
        self._c.close()
        self._conn.close()
        self._c = self._conn = None

    def create_sentence_table(self):
        """ Create a fresh test sentence table if it doesn't already exist """
        assert self._c is not None
        self._c.execute("CREATE TABLE sentences (id serial PRIMARY KEY, sentence varchar, numtrees int, best int, target int);")
        return True

    def add_sentence(self, sentence, numtrees = 0, best = -1, target = 1):
        """ Add a sentence to the test sentence table """
        assert self._c is not None
        self._c.execute("INSERT INTO sentences (sentence, numtrees, best, target) VALUES (%s, %s, %s, %s);",
            [ sentence, numtrees, best, target ])
        return True

    def update_sentence(self, identity, sentence, numtrees = 0, best = -1, target = 1):
        """ Update a sentence and its statistics in the table """
        assert self._c is not None
        self._c.execute("UPDATE sentences SET (sentence, numtrees, best, target) = (%s, %s, %s, %s) WHERE id = %s;",
            [ sentence, numtrees, best, target, identity ])
        return True

    def delete_sentence(self, identity):
        """ Delete a sentence from the table """
        assert self._c is not None
        self._c.execute("DELETE FROM sentences WHERE id = %s;", [ identity ])
        return True

    def sentences(self):
        """ Return a list of all test sentences in the database """
        assert self._c is not None
        m = None
        try:
            self._c.execute("SELECT id, sentence, numtrees, best, target FROM sentences ORDER BY id;")
            t = self._c.fetchall()
            m = [ dict(
                    identity = r[0],
                    sentence = r[1],
                    numtrees = r[2],
                    best = r[3],
                    target = r[4])
                for r in t]
        except psycopg2.DataError as e:
            # Fall through with m set to None
            pass
        return m


def test1():
    # Test grammar 1

    print("------ Test 1 ---------")

    # Abbreviations
    NT = Nonterminal
    TERM = Terminal

    # Hard-coded test case - grammar not read from file

    E = NT ('E')
    T = NT ('T')
    P = NT ('P')
    plus = TERM ('+')
    mult = TERM ('*')
    ident = TERM ('ident')

    g = {
        E: [Production(rhs=[E,plus,T]), Production(rhs=[T])],
        T: [Production(rhs=[T,mult,P]), Production(rhs=[P])],
        P: [Production(rhs=[ident])],
    }

    p = Parser(g, E)
    s = [
        Token('ident', 'a'),
        Token('*', '*'),
        Token('ident', 'b'),
        Token('+', '+'),
        Token('ident', 'c'),
        Token('*', '*'),
        Token('ident', 'd'),
        Token('+', '+'),
        Token('ident', 'e'),
        Token('+', '+'),
        Token('ident', 'f')
    ]

    forest = p.go(s)

    print("Parse combinations: {0}".format(Parser.num_combinations(forest)))

    ParseForestPrinter.print_forest(forest)


def test2():

    print("\n\n------ Test 2 ---------")

    # Test grammar 2 - read from file

    g = Grammar()
    g.read("Reynir.test.grammar")

    #print("Grammar:")
    #print(str(g))
    #print()

    # s = "Villi leit út eða Anna og köttur komu beint heim og kona eða maður fóru snemma inn"
    s = "kona með kött myrti mann með hálsbindi með hund og Páll fór út"
    # s = "kona með kött myrti mann með hund og Villi fór út"
    # s = "Villi leit út"

    class NameToken(Token):

        NÖFN_NF = ["Villi", "Anna", "Hlín", "Páll"]
        NÖFN_ÞF = ["Villa", "Önnu", "Hlín", "Pál"]
        NÖFN_ÞGF = ["Villa", "Önnu", "Hlín", "Páli"]

        def matches(self, terminal):
            """ Does this token match the given terminal? """
            if not terminal.name().startswith("nafn_"):
                return False
            if terminal.name().endswith("_nf"):
                return self._val in NameToken.NÖFN_NF
            if terminal.name().endswith("_þf"):
                return self._val in NameToken.NÖFN_ÞF
            if terminal.name().endswith("_þgf"):
                return self._val in NameToken.NÖFN_ÞGF
            return False

    def make_token(w):
        if w[0].isupper():
            return NameToken('nafn', w)
        return Token('orð', w)

    toklist = [make_token(w) for w in s.split()]

    p = Parser.for_grammar(g)

    forest = p.go(toklist)

    print("Parse combinations: {0}".format(Parser.num_combinations(forest)))

    ParseForestPrinter.print_forest(forest)


def run_test(p, fast_p):
    """ Run a test parse on all sentences in the test table """

    with closing(Test_DB.open_db()) as db:

        slist = db.sentences()

        for s in slist:

            txt = s["sentence"]
            target = s["target"] # The ideal number of parse trees (1 or 0)

            tokens = tokenize(txt)

            tlist = list(tokens)

            # Run the all-Python parser
            err = ""
            try:
                t0 = time.time()
                forest = p.go(tlist)
            except ParseError as e:
                err = "{0}".format(e)
                forest = None
            finally:
                t1 = time.time()

            # Run the C++ parser
            try:
                tf0 = time.time()
                forest2 = fast_p.go(tlist)
            except ParseError as e:
                err = "{0}".format(e)
                forest2 = None
            finally:
                tf1 = time.time()

            num = 0 if forest is None else Parser.num_combinations(forest)
            num2 = Fast_Parser.num_combinations(forest2)

            print("Python: Parsed in {0:.4f} seconds, {1} combinations".format(t1 - t0, num))
            print("C++:    Parsed in {0:.4f} seconds, {1} combinations".format(tf1 - tf0, num2))

            # Clean up the returned forest from Fast_Parser
            Fast_Parser.delete_forest(forest2)

            best = s["best"]
            if best <= 0 or abs(target - num) < abs(target - best):
                # We are closer to the ideal number of parse trees (target) than
                # the best parse so far: change the best one
                best = num

            db.update_sentence(s["identity"], s["sentence"], num, best, target)

            yield dict(
                identity = s["identity"],
                sentence = txt,
                numtrees = num,
                best = best,
                target = target,
                parse_time = t1 - t0,
                err = "" if target == 0 else err, # Don't bother showing errors that are expected
                forest = forest
            )

            # break # !!! DEBUG: only do one loop


def test3():

    print("\n\n------ Test 3 ---------")

    p = BIN_Parser(verbose = False) # Don't emit diagnostic messages
    fast_p = Fast_Parser(verbose = False)

    try:
        g = p.grammar

        print("Reynir.grammar has {0} nonterminals, {1} terminals, {2} productions"
            .format(g.num_nonterminals, g.num_terminals, g.num_productions))

        # Dump the grammar
        # print("\n" + str(g))

        def create_sentence_table():
            """ Only used to create a test fresh sentence table if one doesn't exist """
            with closing(Test_DB.open_db()) as db:

                try:
                    db.create_sentence_table()

                    TEXTS = [
                        "Páll fór út með stóran kött og Jón keypti heitan graut.",
                        "Unga fallega konan frá Garðabæ elti ljóta og feita karlinn rösklega og fumlaust í svörtu myrkrinu",
                        "Kötturinn sem strákurinn átti veiddi feitu músina",
                        "Gamla bláa kommóðan var máluð fjólublá með olíumálningu",
                        "Landsframleiðslan hefur aukist frá því í fyrra",
                        "Guðmundur og Guðrún kusu Framsóknarflokkinn",
                        "Þú skalt fara til Danmerkur.",
                        "Ég og þú fórum til Frakklands í utanlandsferð",
                        "Stóru bláu könnunni mun hafa verið fleygt í ruslið",
                        "Már Guðmundsson segir margskonar misskilnings gæta hjá Hannesi Hólmsteini",
                        "Már Guðmundsson seðlabankastjóri Íslands segir þetta við Morgunblaðið í dag.",
                        "Það er náttúrlega einungis í samfélögum sem eiga við býsna stór vandamál að stríða að ný stjórnmálaöfl geta snögglega sveiflast upp í þriðjungs fylgi.",
                        "Áætlaður kostnaður verkefnisins var tíu milljónir króna og áætluð verklok eru í byrjun september næstkomandi.",
                        "Pakkinn snerist um að ábyrgjast innlán og skuldabréfaútgáfu danskra fjármálafyrirtækja.",
                        "Kynningarfundurinn sem ég hélt í dag fjallaði um lausnina á þessum vanda.",
                        "Kynningarfundurinn sem haldinn var í dag fjallaði um lausnina á þessum vanda.",
                        "Það sakamál sé til meðferðar við Héraðsdóm Suðurlands."
                    ]

                    for t in TEXTS:
                        db.add_sentence(t)

                    slist = db.sentences()
                    for s in slist:
                        print("{0}".format(s))

                except Exception as e:
                    print("{0}".format(e))

        for test in run_test(p, fast_p):

            print("\n'{0}'\n{1} parse trees found in {2:.3f} seconds\n"
                .format(test["sentence"], test["numtrees"], test["parse_time"]))

            if test["numtrees"] > 0:
                # ParseForestPrinter.print_forest(test["forest"])
                # print("{0}".format(Parser.make_schema(test["forest"])))
                pass
            elif test["err"]:
                print("Error: {0}".format(test["err"]))

    finally:
        # Clean up the Fast_Parser to avoid C++ memory leaks
        fast_p.cleanup()


if __name__ == "__main__":

    # Read the configuration settings file

    try:
        Settings.read("Reynir.conf")
    except ConfigError as e:
        print("Configuration error: {0}".format(e))
        quit()

    # Run the tests

    # test1()

    # test2()

    import cProfile as profile
    import pstats

    filename = 'Reynir.profile'

    profile.run('test3()', filename)

    stats = pstats.Stats(filename)

    # Clean up filenames for the report
    stats.strip_dirs()

    # Sort the statistics by the total time spent in the function itself
    stats.sort_stats('tottime')

    stats.print_stats(100) # Print 100 most significant lines
