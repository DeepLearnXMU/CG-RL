import copy

try:
    from cStringIO import StringIO
except:
    from io import StringIO

import sys

import numpy as np
#sys.path.append("/home/xbb/tranX")
from collections import Iterable

from asdl.asdl import *
from asdl.asdl_ast import AbstractSyntaxTree, RealizedField



def typename(x):
    if isinstance(x, str):
        return x
    return x.__name__

class Node(object):
    def __init__(self, name, children=None):
        self.name = name
        self.parent = None
        self.children = list()

        if children:
            if isinstance(children, Iterable):
                for child in children:
                    self.add_child(child)
            elif isinstance(children, Node):
                self.add_child(children)
            else:
                raise AttributeError('Wrong type for child nodes')

    def add_child(self, child):
        child.parent = self
        self.children.append(child)
    
    def __hash__(self):
        code = hash(self.name)
        return code


    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if hash(self) != hash(other):
            return False

        if self.name != other.name:
            return False

        if len(self.children) != len(other.children):
            return False

        for i in range(len(self.children)):
            if self.children[i] != other.children[i]:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return '%s' % (self.name)

    @property
    def is_leaf(self):
        return len(self.children) == 0
    
    def to_string(self, sb=None):
        is_root = False
        if sb is None:
            is_root = True
            sb = StringIO()

        if self.is_leaf:
            sb.write(self.name)
        else:
            sb.write('( ')
            sb.write(self.name)

            for child in self.children:
                sb.write(' ')
                child.to_string(sb)

            sb.write(' )')

        if is_root:
            return sb.getvalue()    
    
    def __getitem__(self, child_name):
        return next(iter([c for c in self.children if c.name == child_name]))

    def __delitem__(self, child_name):
        tgt_child = [c for c in self.children if c.name == child_name]
        if tgt_child:
            assert len(tgt_child) == 1, 'unsafe deletion for more than one children'
            tgt_child = tgt_child[0]
            self.children.remove(tgt_child)
        else:
            raise KeyError



def ifttt_ast_to_parse_tree_helper(s, offset):
    """
    adapted from ifttt codebase
    """
    if s[offset] != '(':
        name = ''
        while offset < len(s) and s[offset] != ' ':
            name += s[offset]
            offset += 1

        node = Node(name)
        return node, offset
    else:
        # it's a sub-tree
        offset += 2
        name = ''
        while s[offset] != ' ':
            name += s[offset]
            offset += 1

        node = Node(name)
        # extract its child nodes

        while True:
            if s[offset] != ' ':
                raise ValueError('malformed string: node should have either had a '
                                 'close paren or a space at position %d' % offset)

            offset += 1
            if s[offset] == ')':
                offset += 1
                return node, offset
            else:
                child_node, offset = ifttt_ast_to_parse_tree_helper(s, offset)

            node.add_child(child_node)


def ifttt_ast_to_parse_tree(s, attach_func_to_channel=True):
    parse_tree, _ = ifttt_ast_to_parse_tree_helper(s, 0)
    #parse_tree = strip_params(parse_tree)

    #if attach_func_to_channel:
    #    parse_tree = attach_function_to_channel(parse_tree)

    return parse_tree


def ifttt_form_to_ast(grammar, lf_node):
    if lf_node.name == 'IF':
        # expr -> Lambda(var variable, var_type type, expr body)
        prod = grammar.get_prod_by_ctr_name('IfFunction')
        condition_node = lf_node.children[0]
        condition_ast_node = ifttt_form_to_ast(grammar, condition_node)  
        condition_field = RealizedField(prod['condition'], condition_ast_node)
        thenfunc_prod = grammar.get_prod_by_ctr_name('Then')
        thenfunc_field = RealizedField(prod['args'], AbstractSyntaxTree(thenfunc_prod))
        body_node = lf_node.children[2]
        body_ast_node = ifttt_form_to_ast(grammar, body_node)
        body_field = RealizedField(prod['body'], body_ast_node)  
        ast_node = AbstractSyntaxTree(prod,
                                      [condition_field, thenfunc_field, body_field])
    elif lf_node.name == 'FunctionDef':
        prod = grammar.get_prod_by_ctr_name('FunctionDef')
        name_node = lf_node.children[0]
        name_field = RealizedField(prod['name'], name_node)
        condition_node = lf_node.children[1]
        condition_field = RealizedField(prod['condition'], condition_node)
        ast_node = AbstractSyntaxTree(prod,
                                      [name_field, condition_field])
    else:
        raise NotImplementedError
    return ast_node
    #return None



def strip_params(parse_tree):
    if parse_tree.name == 'PARAMS':
        raise RuntimeError('should not go to here!')

    parse_tree.children = [c for c in parse_tree.children if c.name != 'PARAMS' and c.name != 'OUTPARAMS']
    for i, child in enumerate(parse_tree.children):
        parse_tree.children[i] = strip_params(child)

    return parse_tree


def attach_function_to_channel(parse_tree):
    trigger_func = parse_tree['TRIGGER']['FUNC'].children
    assert len(trigger_func) == 1

    trigger_func = trigger_func[0]
    parse_tree['TRIGGER'].children[0].add_child(trigger_func)

    del parse_tree['TRIGGER']['FUNC']

    action_func = parse_tree['ACTION']['FUNC'].children
    assert len(action_func) == 1

    action_func = action_func[0]
    parse_tree['ACTION'].children[0].add_child(action_func)

    del parse_tree['ACTION']['FUNC']

    return parse_tree



def ast_to_ifttt_form(ast_tree):
    constructor_name = ast_tree.production.constructor.name
    if constructor_name == 'IfFunction':
        condition_node = ast_to_ifttt_form(ast_tree['condition'].value)
        args_node = ast_to_ifttt_form(ast_tree['args'].value)
        body_node = ast_to_ifttt_form(ast_tree['body'].value)
        node = Node('IF', [condition_node, args_node, body_node])
    elif constructor_name == 'FunctionDef':
        name_node = ast_tree['name'].value
        condition_node = ast_tree['condition'].value
        node = Node('FunctionDef', [name_node, condition_node])
    elif constructor_name == 'Then':
        node = Node('THEN', [])
        
    return node



if __name__ == '__main__':
    asdl_desc = """
    # define primitive fields
    var, identifier

    expr = IfFunction(stmt condition, thenfunc args, stmt body) 
    
    stmt = FunctionDef(identifier name, var condition)
    
    thenfunc = Then
    """

    grammar = ASDLGrammar.from_text(asdl_desc)
    
    tree_code = "( IF ( FunctionDef Instagram Any_new_photo_by_you ) THEN ( FunctionDef Dropbox Add_file_from_URL ) )"
    parse_tree = ifttt_ast_to_parse_tree(tree_code)
    ast_tree = ifttt_form_to_ast(grammar, parse_tree)
    new_ifttt = ast_to_ifttt_form(ast_tree)
    print(new_ifttt.to_string())
    assert parse_tree.to_string() == new_ifttt.to_string()
    