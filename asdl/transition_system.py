# coding=utf-8
from random import shuffle

class Action(object):
    pass


class ApplyRuleAction(Action):
    def __init__(self, production):
        self.production = production

    def __hash__(self):
        return hash(self.production)

    def __eq__(self, other):
        return isinstance(other, ApplyRuleAction) and self.production == other.production

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return 'ApplyRule[%s]' % self.production.__repr__()


class GenTokenAction(Action):
    def __init__(self, token):
        self.token = token

    def is_stop_signal(self):
        return self.token == '</primitive>'

    def __repr__(self):
        return 'GenToken[%s]' % self.token


class ReduceAction(Action):
   def __repr__(self):
       return 'Reduce'


class TransitionSystem(object):
    def __init__(self, grammar):
        self.grammar = grammar

    def get_actions(self, asdl_ast, shuffle_mode='normal'):
        """
        generate action sequence given the ASDL Syntax Tree
        """

        actions = []
        shuffle_ids = []
        parent_action = ApplyRuleAction(asdl_ast.production)
        actions.append(parent_action)
        
        shuffle_fields_idx = [i for i in range(len(asdl_ast.fields))]
        if shuffle_mode == 'reverse':
            shuffle_fields_idx.reverse()
        elif shuffle_mode == 'random':
            shuffle(shuffle_fields_idx)
        else:
            pass
        shuffle_ids.append(shuffle_fields_idx)
        fields = [asdl_ast.fields[i] for i in shuffle_fields_idx]  
        for field in fields:
            # is a composite field
            if self.grammar.is_composite_type(field.type):
                if field.cardinality == 'single':
                    field_actions, other_shuffle_ids = self.get_actions(field.value, shuffle_mode)
                else:
                    field_actions = []
                    other_shuffle_ids = []
                    if field.value is not None:
                        if field.cardinality == 'multiple':
                            for val in field.value:
                                cur_child_actions, other_shuffle_id = self.get_actions(val, shuffle_mode)
                                other_shuffle_ids.extend(other_shuffle_id)
                                field_actions.extend(cur_child_actions)
                        elif field.cardinality == 'optional':
                            field_actions, other_shuffle_id = self.get_actions(field.value, shuffle_mode)
                            other_shuffle_ids.extend(other_shuffle_id)

                    # if an optional field is filled, then do not need Reduce action
                    if field.cardinality == 'multiple' or field.cardinality == 'optional' and not field_actions:
                        field_actions.append(ReduceAction())
                        other_shuffle_ids.extend([None])
            else:  # is a primitive field
                field_actions = self.get_primitive_field_actions(field)
                other_shuffle_ids = [None] * len(field_actions)

                # if an optional field is filled, then do not need Reduce action
                if field.cardinality == 'multiple' or field.cardinality == 'optional' and not field_actions:
                    # reduce action
                    field_actions.append(ReduceAction())
                    other_shuffle_ids.extend([None])
                    
            actions.extend(field_actions)
            shuffle_ids.extend(other_shuffle_ids)

        return actions, shuffle_ids

    def tokenize_code(self, code, mode):
        raise NotImplementedError

    def compare_ast(self, hyp_ast, ref_ast):
        raise NotImplementedError

    def ast_to_surface_code(self, asdl_ast):
        raise NotImplementedError

    def surface_code_to_ast(self, code):
        raise NotImplementedError

    def get_primitive_field_actions(self, realized_field):
        raise NotImplementedError

    def get_valid_continuation_types(self, hyp):
        if hyp.tree:
            if self.grammar.is_composite_type(hyp.frontier_field.type):
                if hyp.frontier_field.cardinality == 'single':
                    return ApplyRuleAction,
                else:  # optional, multiple
                    return ApplyRuleAction, ReduceAction
            else:
                if hyp.frontier_field.cardinality == 'single':
                    return GenTokenAction,
                elif hyp.frontier_field.cardinality == 'optional':
                    if hyp._value_buffer:
                        return GenTokenAction,
                    else:
                        return GenTokenAction, ReduceAction
                else:
                    return GenTokenAction, ReduceAction
        else:
            return ApplyRuleAction,

    def get_valid_continuating_productions(self, hyp):
        if hyp.tree:
            if self.grammar.is_composite_type(hyp.frontier_field.type):
                return self.grammar[hyp.frontier_field.type]
            else:
                raise ValueError
        else:
            return self.grammar[self.grammar.root_type]

    @staticmethod
    def get_class_by_lang(lang):
        if lang == 'python':
            from .lang.py.py_transition_system import PythonTransitionSystem
            return PythonTransitionSystem
        elif lang == 'python3':
            from .lang.py3.py3_transition_system import Python3TransitionSystem
            return Python3TransitionSystem
        elif lang == 'lambda_dcs':
            from .lang.lambda_dcs.lambda_dcs_transition_system import LambdaCalculusTransitionSystem
            return LambdaCalculusTransitionSystem
        elif lang == 'prolog':
            from .lang.prolog.prolog_transition_system import PrologTransitionSystem
            return PrologTransitionSystem
        elif lang == 'wikisql':
            from .lang.sql.sql_transition_system import SqlTransitionSystem
            return SqlTransitionSystem

        raise ValueError('unknown language %s' % lang)
