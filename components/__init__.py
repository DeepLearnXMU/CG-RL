import six

from datasets.django.evaluator import DjangoEvaluator

from datasets.conala.evaluator import ConalaEvaluator
if six.PY3:
    from datasets.wikisql.evaluator import WikiSQLEvaluator
    from datasets.ifttt.evaluator import IFTTTEvaluator

