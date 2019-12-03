# -*- coding: utf-8 -*-

"""
    Reynir: Natural language processing for Icelandic

    Neural Network Parsing Utilities

    Copyright (C) 2018 Miðeind ehf

       This program is free software: you can redistribute it and/or modify
       it under the terms of the GNU General Public License as published by
       the Free Software Foundation, either version 3 of the License, or
       (at your option) any later version.
       This program is distributed in the hope that it will be useful,
       but WITHOUT ANY WARRANTY; without even the implied warranty of
       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
       GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see http://www.gnu.org/licenses/.


    This module contains a handful of useful functions for parsing the
    output from an IceParsing neural network model and working with the
    result.

"""

from __future__ import print_function
from enum import IntEnum
import logging
from pprint import pprint
import inspect

import tokenizer
from reynir import matcher
from reynir import bintokenizer


class GRAMMAR:
    CASES = {"nf", "þf", "þgf", "ef"}
    GENDERS = {"kk", "kvk", "hk"}
    NUMBERS = {"et", "ft"}
    PERSONS = {"p1", "p2", "p3"}

    TENSE = {"þt", "nt"}
    DEGREE = {"mst", "esb", "evb"}  # fst

    VOICE = {"mm", "gm"}
    MOOD = {"fh", "lhþt", "lhnt", "vh", "bh"}

    MISC = {"sagnb", "subj", "abbrev", "op", "none"}


class Node:
    """ Generic tree implementation """

    def __init__(self, name, is_terminal=False, data=None):
        self.name = name
        self.data = data
        self.children = []
        self.is_terminal = is_terminal

    def add_child(self, node):
        self.children.append(node)
        return self

    def has_children(self):
        return bool(self.children)

    def to_dict(self):
        if not self.children:
            json_node = _json_terminal_node(self.name, self.data)
        else:
            json_node = _json_nonterminal_node(self.name)
            json_node[KEY.children] = [c.to_dict() for c in self.children]
        return json_node

    def to_simple_tree(self):
        return matcher.SimpleTree([[self.to_dict()]])

    def to_postfix(self):
        """ Export tree to postfix ordering
            with node-labels as keys """
        result = []
        for child in self.children:
            result.extend(child.to_postfix())
        result.append(self.name)
        return result

    def __str__(self):
        text = (" " + self.data) if self.data is not None else ""
        return "<" + self.name + text + ">"

    def __repr__(self):
        if self.is_terminal:
            return self.__str__()
        result = [self.__str__()]
        for child in self.children:
            result.extend(child.__repr__())
        result.append(Node("/" + self.name).__str__())
        return "".join(result)

    def _pprint(self, _prefix="  ", depth=0):
        """ Inner function for pretty printing tree """
        print("%s%s" % (_prefix * depth, self.__str__()))
        for c in self.children:
            c._pprint(_prefix=_prefix, depth=depth + 1)

    def pprint(self, indent=4):
        """ Pretty print tree """
        self._pprint(_prefix=" " * indent)

    def width(self):
        """ Returns number of leaf nodes,
            assumes a correctly formed tree """
        if self.children:
            return sum([c.width() for c in self.children])
        return 1

    def height(self):
        """ Returns height of tree,
            assumes a correctly formed tree """
        if not self.children:
            return 1
        return 1 + max([c.height() for c in self.children])

    @staticmethod
    def contains(forest, name):
        if isinstance(forest, Node):
            forest = [forest]
        for tree in forest:
            if tree.name == name:
                return True
            if any(Node.contains(c, name) for c in tree.children):
                return True
        return False


class ParseResult(IntEnum):
    """ Result types for the parsing of flat parse trees that are returned by
        the NMT parsing model and the corresponding natural language source text
        into a (non) flat parse tree """

    # Successful parse
    SUCCESS = 0
    # A text token was not consumed before reaching end of parse tok stream
    INCOMPLETE = 1
    # A corresponding left token is missing before a right token is encountered
    UNOPENED_RTOKEN = 2
    # A left token is not closed before the surrounding nonterminal token is closed
    UNCLOSED_LTOKEN = 3
    # Terminal descends from pseudo root instead of a proper nonterminal
    TERM_DESC_ROOT = 5
    # Nonterminal has no terminal or leaf token
    EMPTY_NONTERM = 6
    # Multiple successful parse trees
    MULTI = 7
    # Undocumented parse failure
    FAILURE = 8


def flat_is_nonterminal(string):
    return string.isupper() and "_" not in string


def flat_is_terminal(string):
    return string.islower()


def flat_is_left_nonterminal(string):
    return flat_is_nonterminal(string) and "/" not in string


def flat_is_right_nonterminal(string):
    return flat_is_nonterminal(string) and "/" not in string


def flat_matching_nonterminal(string):
    if "/" in string:
        return string[1:]
    return "/" + string


def parse_flat_tree_to_nodes(parse_toks, text_toks=None, verbose=False):
    """Parses list of toks (parse_toks) into a legal tree structure or None,
       If the corresponding tokenized source text is provided, it is
       included in the tree"""

    vprint = logging.debug if verbose else (lambda *ar, **kw: None)

    if not parse_toks or not flat_is_nonterminal(parse_toks[0]):
        raise ValueError("Invalid parse tokens.")

    root = Node(name="ROOT", is_terminal=False)  # Pseudo root
    stack = []
    parent = root
    txt_count = 0

    for idx, tok in enumerate(parse_toks):
        if flat_is_terminal(tok):
            if parent == root:
                msg = "Error: Tried to parse terminal node {} as descendant of root."
                vprint(msg.format(tok))

                tree = root.children[0] if root.children else None
                return tree, ParseResult.TERM_DESC_ROOT

            if text_toks and txt_count < len(text_toks):
                parse_tok, text_tok = tok, text_toks[txt_count]
                new_node = Node(parse_tok, data=text_tok, is_terminal=True)
                txt_count += 1
            else:
                new_node = Node(tok)

            string = "{}  {}".format(len(stack) * "    ", tok)
            string = string if not new_node.data else string + new_node.data
            vprint(string)

            parent.add_child(new_node)
            continue

        if flat_is_left_nonterminal(tok):
            new_node = Node(tok, is_terminal=False)
            parent.add_child(new_node)
            stack.append(parent)
            parent = new_node

            vprint(len(stack) * "    " + new_node.name)
            continue

        # Token must be a legal right nonterminal since it is not a left token
        # A right token must be matched by its corresponding left token
        if (
            not flat_is_right_nonterminal(tok)
            or flat_matching_nonterminal(parent.name) != tok
        ):
            msg = "Error: Found illegal nonterminal {}, expected right nonterminal"
            vprint(msg.format(tok))

            tree = root.children[0] if root.children else None
            return tree, ParseResult.UNOPENED_RTOKEN
        # Empty NONTERMINALS are not allowed
        if not parent.has_children():
            msg = ["{}{}".format(len(stack) * "    ", tok) for tok in parse_toks[idx:]]
            vprint("\n".join(msg))
            vprint("Error: Tried to close empty nonterminal {}".format(tok))

            tree = root.children[0] if root.children else None
            return tree, ParseResult.EMPTY_NONTERM
        parent = stack.pop()

    if len(stack) > 1:
        vprint("Error: Reached end of parse tokens but stack is not empty")
        vprint([item.name for item in stack])
        tree = root.children[0] if root.children else None
        return tree, ParseResult.INCOMPLETE

    if not root.children:
        return None, ParseResult.FAILURE
    if len(root.children) == 1:
        return root.children[0], ParseResult.SUCCESS

    return root.children, ParseResult.MULTI


def tokenize_flat_tree(parse_str):
    return parse_str.strip().split(" ")


def parse_tree(flat_parse_str):
    return parse_flat_tree_to_nodes(tokenize_flat_tree(flat_parse_str))


def parse_tree_with_text(flat_tree_str, text):
    text_toks, parse_toks = tokenize_and_merge_possible_mw_tokens(text, flat_tree_str)
    return parse_flat_tree_to_nodes(parse_toks, text_toks)


class KEY:
    long_terminal = "a"
    bin_variants = "b"
    bin_category = "c"
    bin_fl = "f"
    nonterminal_tag = "i"  # nonterminal
    token_index = "ix"
    kind = "k"  # token or nonterminal kind
    nonterminal_name = "n"
    lemma = "s"
    short_terminal = "t"  # matchable categories
    text = "x"
    children = "p"


FieldKeyToName = {
    value: key for (key, value) in inspect.getmembers(KEY) if not key.startswith("_")
}


def _json_terminal_node(tok, text="placeholder", token_index=None):
    """ first can be:
            abfn
            ao
            fn
            fs
            gata
            gr
            lo
            no
            person
            pfn
            raðnr
            sérnafn
            so
            so_0
            so_1
            so_2
            tala
            to
            töl"""

    subtokens = tok.split("_")
    first = subtokens[0].split("_", 1)[0]

    tail_start = 1
    if first == "so":
        head = subtokens[0]
        suffix = head[-1]
        if "_" in head and suffix in "012":
            tail_start += int(suffix)
    tail = subtokens[tail_start:]

    if first == "no":
        gender = [t for t in GRAMMAR.GENDERS if t in tail][:1]
        gender = gender.pop() if gender else ""
        case = [t for t in GRAMMAR.CASES if t in tail][:1]
        case = case.pop() if case else ""
        number = [t for t in GRAMMAR.NUMBERS if t in tail][:1]
        number = number.pop() if number else ""
        gr = "gr" if "gr" in tail else ""

        bin_variants = "-".join([t for t in [case, number, gr] if t])

        new_node = {
            KEY.bin_category: gender,
            KEY.short_terminal: tok,
            KEY.text: text,
            KEY.kind: "WORD",
            KEY.bin_variants: bin_variants,
        }

    elif first == "st":
        new_node = {
            KEY.short_terminal: tok,
            KEY.text: text,
            KEY.kind: "WORD",
            KEY.bin_variants: "-",
            KEY.bin_category: "st",
        }

    elif first == "eo":
        new_node = {
            KEY.short_terminal: tok,
            KEY.text: text,
            KEY.kind: "WORD",
            KEY.bin_variants: "ao",
            KEY.bin_category: "ao",
        }

    elif first == "entity":
        new_node = {KEY.short_terminal: tok, KEY.text: text, KEY.kind: "ENTITY"}

    elif first == "person":
        gender = [t for t in GRAMMAR.GENDERS if t in tail][:1]
        gender = gender.pop() if gender else ""
        new_node = {
            KEY.bin_category: gender or "-",
            KEY.short_terminal: tok,
            KEY.kind: "PERSON",
            KEY.lemma: "s",
            KEY.text: text,
        }

    elif first == "p":
        new_node = {KEY.text: text, KEY.kind: "PUNCTUATION"}

    else:
        bin_variants = "-".join([t for t in tail if t])
        new_node = {
            KEY.text: text,
            KEY.short_terminal: tok,
            KEY.bin_category: first,
            KEY.kind: "WORD",
            KEY.bin_variants: bin_variants,
        }

    if KEY.bin_variants in new_node:
        new_node[KEY.bin_variants] = (
            new_node[KEY.bin_variants]
            .upper()
            .replace("GR", "gr")
            .replace("P1", "1P")
            .replace("P2", "2P")
            .replace("P3", "3P")
        )

    if KEY.kind in new_node and new_node[KEY.kind] in ["ENTITY", "WORD", "PERSON"]:
        new_node[KEY.lemma] = text if text else "-"
    new_node[KEY.bin_fl] = "alm"
    new_node[KEY.long_terminal] = tok

    # TODO: refactor flat terminal parsing and use error codesearch
    if KEY.text in new_node and new_node[KEY.text] is None:
        new_node[KEY.text] = ""

    if token_index is not None:
        new_node[KEY.token_index] = token_index

    return new_node


def _json_nonterminal_node(tok):
    new_node = {
        KEY.nonterminal_tag: tok,
        KEY.nonterminal_name: matcher._DEFAULT_ID_MAP[tok]["name"],
        KEY.kind: "NONTERMINAL",
        KEY.children: [],
    }
    return new_node


def tokenize_and_merge_possible_mw_tokens(text, flat_tree):
    mw_tokens = list(bintokenizer.tokenize(text))
    mw_tokens = [tok.txt.split(" ") for tok in mw_tokens if tok.txt is not None]
    sw_tokens = [tok for toks in mw_tokens for tok in toks]  # flatten multiword tokens

    parse_tokens = list(flat_tree.split(" "))
    parse_terminals = filter(lambda x: x[1][0].islower(), enumerate(parse_tokens))
    parse_terminals = list(enumerate(parse_terminals))

    term_idx_to_parse_idx = {
        term_idx: ptok_idx for (term_idx, (ptok_idx, ptok)) in parse_terminals
    }

    offset = 0
    merge_list = []
    for mw_token in mw_tokens:
        weight = len(mw_token)
        idxed_mw_token = [(idx + offset, token) for (idx, token) in enumerate(mw_token)]
        offset += weight
        if weight == 1:
            continue
        merge_info = check_merge_candidate(
            idxed_mw_token, parse_tokens, term_idx_to_parse_idx
        )
        if merge_info is not None:
            merge_list.append(merge_info)

    # merge in reverse order so we don't have to compute offsets
    for (pidx, tidx, weight) in reversed(merge_list):
        parse_tokens[pidx : pidx + weight] = parse_tokens[pidx : pidx + 1]
        sw_tokens[tidx : tidx + weight] = [" ".join(sw_tokens[tidx : tidx + weight])]

    return sw_tokens, parse_tokens


def check_merge_candidate(idxed_mw_token, parse_tokens, term_idx_to_parse_idx):
    # idx_mw_tokens has at least two tokens
    allow_merge = True
    last_ptok = None
    last_pidx = None
    first_pidx = None
    for (idx, token) in idxed_mw_token:
        pidx = term_idx_to_parse_idx[idx]
        last_pidx = pidx - 1 if last_pidx is None else last_pidx
        ptok = parse_tokens[pidx]
        last_ptok = ptok if last_ptok is None else last_ptok

        # parse_tokens must be contiguous and must match
        allow_merge = allow_merge and (last_ptok == ptok) and (last_pidx + 1 == pidx)
        if not allow_merge:
            return None

        first_pidx = pidx if first_pidx is None else first_pidx
        last_pidx = pidx
        last_ptok = ptok
    token_idxs, words = list(zip(*idxed_mw_token))

    return (first_pidx, token_idxs[0], len(idxed_mw_token))


def debug_reynir_dict_format():
    from reynir import Reynir

    parser = Reynir()
    text = "hún fer"
    sent = parser.parse_single(text)

    print("text:", text)
    print()

    print("reynir flat tree:")
    print(sent.tree.flat)
    print()

    res = sent.tree._view(1)
    print("reynir tree as view:")
    print(res)
    print()

    print("reynir tree as dict:")
    pprint(sent.tree._head)
    print()


def debug_reynir_terminal():
    text_tok = "hún"
    from reynir import Reynir
    from reynir.matcher import SimpleTree

    parser = Reynir()
    sent = parser.parse_single(text_tok)

    print("Parsing text:", text_tok)
    print()

    print("flat tree:")
    print(sent.tree.flat)

    print("tree view:")
    print(sent.tree._view(1))

    print("tree._head:")
    dtree = sent.tree._head
    pprint(dtree)

    print("tree._sents:")
    sents_attr = sent.tree._sents
    pprint(sents_attr)

    print("reparse _sents")
    stree = SimpleTree([sents_attr])
    print(stree)
    pprint(stree._head)


def debug_nntree_terminal():
    text_tok = "hún"
    parse_tok = "pfn_kvk_et_nf"
    from reynir.matcher import SimpleTree

    terminal = Node(parse_tok, data=text_tok, is_terminal=True)
    dtree = terminal.to_dict()

    print("parsing flat tree to simple tree:")
    print("flat tree:", parse_tok)
    print("with text", text_tok)
    print()

    print("parsed flat tree as dict:")
    print(dtree)

    print("parsed flat tree as SimpleTree")
    stree = SimpleTree([[dtree]])
    print(stree)
    print()


def debug_parse_with_text():
    from reynir.matcher import SimpleTree

    text = "hún fer"
    nnparser_toks = "P S-MAIN IP NP-SUBJ pfn_et_kvk_nf_p3 /NP-SUBJ VP so_0_et_fh_gm_nt_p3 /VP /IP /S-MAIN /P"
    reynir_toks = (
        "P S-MAIN IP NP-SUBJ pfn_kvk_et_nf /NP-SUBJ VP so_0_et_p3 /VP /IP /S-MAIN /P"
    )
    parse_toks = nnparser_toks
    nntree, presult = parse_tree_with_text(parse_toks, text)

    print("parsing flat tree to simple tree:")
    print("flat tree:", parse_toks)
    print("with text", text)
    print()

    print("parsed flat tree as nntree dict:")
    dtree = nntree.to_dict()
    pprint(dtree)
    # print(presult)
    print()

    print("parsed flat tree as SimpleTree view:")
    print(SimpleTree([[dtree]])._view(1))


def debug_nntree_to_simple_tree():
    from reynir.matcher import SimpleTree

    text = "hún fer"
    # nnparser_toks
    parse_toks = "P S-MAIN IP NP-SUBJ pfn_et_kvk_nf_p3 /NP-SUBJ VP so_0_et_fh_gm_nt_p3 /VP /IP /S-MAIN /P"
    nntree, presult = parse_tree_with_text(parse_toks, text)

    print("parsing flat tree to simple tree:")
    print("flat tree:", parse_toks)
    print("with text", text)
    print()

    print("nntree:")
    nntree.pprint()
    print()

    print("nntree to SimpleTree:")
    print(nntree.to_simple_tree()._view(1))
    print()


def debug_parse():
    from reynir.matcher import SimpleTree

    # flat tree from reynir.Reynir()
    sample = (
        "P S-MAIN IP NP-SUBJ pfn_kvk_et_nf /NP-SUBJ VP so_0_et_p3 /VP /IP /S-MAIN /P"
    )
    res = parse_tree(sample)
    print(res)


def test_merge_person():
    text = "Ingibjörg Sólrún Gísladóttir mun a.m.k. hitta hópinn á morgun"
    flat_tree = "P S-MAIN IP NP-SUBJ person_kvk_nf person_kvk_nf person_kvk_nf /NP-SUBJ VP-SEQ VP so_et_fh_gm_nt_p3 ADVP ao /ADVP so_1_þf_gm_nh NP-OBJ no_et_gr_kk_þf /NP-OBJ /VP ADVP ADVP-DATE-REL ao ao /ADVP-DATE-REL /ADVP /VP-SEQ /IP /S-MAIN /P"
    text_toks, parse_toks = tokenize_and_merge_possible_mw_tokens(text, flat_tree)
    assert len(text_toks) == 6

    # change parse token in for person token
    text = "Ingibjörg Sólrún Gísladóttir mun a.m.k. hitta hópinn á morgun"
    flat_tree = "P S-MAIN IP NP-SUBJ person_kvk_nf no_kvk_nf person_kvk_nf /NP-SUBJ VP-SEQ VP so_et_fh_gm_nt_p3 ADVP ao /ADVP so_1_þf_gm_nh NP-OBJ no_et_gr_kk_þf /NP-OBJ /VP ADVP ADVP-DATE-REL ao ao /ADVP-DATE-REL /ADVP /VP-SEQ /IP /S-MAIN /P"
    text_toks, parse_toks = tokenize_and_merge_possible_mw_tokens(text, flat_tree)
    assert len(text_toks) == 8

    # change parse token for ao
    text = "Ingibjörg Sólrún Gísladóttir mun a.m.k. hitta hópinn á morgun"
    flat_tree = "P S-MAIN IP NP-SUBJ person_kvk_nf no_kvk_nf person_kvk_nf /NP-SUBJ VP-SEQ VP so_et_fh_gm_nt_p3 ADVP ao /ADVP so_1_þf_gm_nh NP-OBJ no_et_gr_kk_þf /NP-OBJ /VP ADVP ADVP-DATE-REL fs ao /ADVP-DATE-REL /ADVP /VP-SEQ /IP /S-MAIN /P"
    text_toks, parse_toks = tokenize_and_merge_possible_mw_tokens(text, flat_tree)
    assert len(text_toks) == 9

    text = "Þetta er í fimmta sinn sem málið er lagt fram en upphaflega flutti málið núverandi hæstv. dómsmálaráðherra, Áslaug Arna Sigurbjörnsdóttir, og hefur hún verið fyrsti flutningsmaður að málinu hingað til, en við sem flytjum það núna eru Vilhjálmur Árnason, Óli Björn Kárason, Páll Magnússon, Njáll Trausti Friðbertsson, Bryndís Haraldsdóttir, Brynjar Níelsson, Ásmundur Friðriksson og Jón Gunnarsson."
    flat_tree = "S0 S-MAIN IP NP-SUBJ fn_et_hk_nf /NP-SUBJ VP VP so_1_nf_et_fh_gm_nt_p3 /VP PP P fs_þf /P NP lo_et_hk_þf no_et_hk_þf CP-REL C stt /C IP S-MAIN VP NP-PRD no_et_gr_hk_nf /NP-PRD VP so_1_nf_et_fh_gm_nt_p3 /VP NP-PRD VP so_et_hk_lhþt_nf_sb /VP /NP-PRD /VP ADVP ao /ADVP /S-MAIN /IP C st /C IP S-MAIN ADVP ao /ADVP VP VP so_1_þf_et_fh_gm_p3_þt /VP NP-OBJ no_et_gr_hk_þf NP-POSS lo_ef_et_kk_sb lo_ef_et_kk no_ef_et_kk p S-MAIN IP NP-SUBJ person_kvk_nf person_kvk_nf person_kvk_nf p S-MAIN IP ADVP ao /ADVP VP VP VP-AUX so_et_fh_gm_nt_p3 /VP-AUX NP-SUBJ pfn_et_kvk_nf_p3 /NP-SUBJ VP VP so_1_nf_gm_sagnb /VP NP-PRD lo_et_kk_nf_vb no_et_kk_nf /NP-PRD /VP /VP PP P fs_þgf /P NP no_et_gr_hk_þgf /NP /PP ADVP ao ao /ADVP /VP /IP /S-MAIN p C st /C pfn_ft_nf_p1 CP-REL C stt /C IP S-MAIN VP VP so_1_þf_fh_ft_gm_nt_p1 /VP NP-OBJ pfn_et_hk_p3_þf /NP-OBJ /VP ADVP ao /ADVP /S-MAIN /IP /CP-REL /NP-SUBJ VP VP so_1_nf_fh_ft_gm_nt_p3 /VP NP-PRD person_kk_nf person_kk_nf /NP-PRD /VP /IP /S-MAIN p /NP-POSS /NP-OBJ /VP /S-MAIN /IP /CP-REL /NP /PP NP-PRD person_kk_nf person_kk_nf person_kk_nf p person_kk_nf person_kk_nf p person_kk_nf person_kk_nf person_kk_nf p person_kvk_nf person_kvk_nf p person_kk_nf person_kk_nf p person_kk_nf person_kk_nf C st /C person_kk_nf person_kk_nf /NP-PRD /VP /IP /S-MAIN p /S0"
    text_toks, parse_toks = tokenize_and_merge_possible_mw_tokens(text, flat_tree)
    assert len(text_toks) == 54


def test_flat_fns():
    nonterms = ["P", "S0", "NP-SUBJ"]
    for nt in nonterms:
        assert flat_is_nonterminal(nt), "{} should be nonterminal".format(nt)
    match = ["/NP-SUBJ", "NP-SUBJ"]
    assert flat_matching_nonterminal(match[0]) == match[1], "{} should match {}".format(
        *match
    )
    assert flat_matching_nonterminal(match[1]) == match[0], "{} should match {}".format(
        *match
    )
    terms = ["so_1_nf_et_fh_gm_nt_p3", "lo_et_hk_þf", "tö"]
    for term in terms:
        assert flat_is_terminal(term), "{} should be terminal".format(term)


if __name__ == "__main__":
    logging.getLogger("").setLevel(logging.DEBUG)
    test_flat_fns()
    test_merge_person()

    # debug_reynir_dict_format()
    # debug_parse_with_text()
    # debug_reynir_terminal()
    # debug_nntree_terminal()
    # debug_nntree_to_simple_tree()

    pass
