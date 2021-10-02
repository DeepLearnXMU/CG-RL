# coding=utf-8

from asdl.transition_system import TransitionSystem, GenTokenAction, ReduceAction

from .ifttt_form import ast_to_ifttt_form, ifttt_form_to_ast, Node, ifttt_ast_to_parse_tree

from common.registerable import Registrable


@Registrable.register('ifttt')
class IFTTTTransitionSystem(TransitionSystem):
    def tokenize_code(self, code, mode=None):
        return code.strip().split(' ')

    def surface_code_to_ast(self, code):
        return ifttt_form_to_ast(self.grammar, ifttt_ast_to_parse_tree(code))

    def compare_ast(self, hyp_ast, ref_ast):
        ref_lf = ast_to_ifttt_form(ref_ast)
        hyp_lf = ast_to_ifttt_form(hyp_ast)

        return ref_lf == hyp_lf

    def ast_to_surface_code(self, asdl_ast):
        lf = ast_to_ifttt_form(asdl_ast)
        code = lf.to_string()

        return code

    def get_primitive_field_actions(self, realized_field):
        assert realized_field.cardinality == 'single'
        if realized_field.value is not None:
            return [GenTokenAction(realized_field.value)]
        else:
            return []

    def is_valid_hypothesis(self, hyp, **kwargs):
        return True
