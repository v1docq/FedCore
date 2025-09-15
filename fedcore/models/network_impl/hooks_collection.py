from itertools import chain 
from typing import Literal

from fedcore.models.network_impl.hooks import BaseHook


class HooksCollection:
    def __init__(self, hooks=None):
        self.hooks = []
        for hook in (hooks or []):
            self.append(hook)

    @property
    def start(self):
        return [hook for hook in self.hooks if hook._hook_place <= 1]
    
    @property
    def end(self):
        return [hook for hook in self.hooks if hook._hook_place >= -1]
        
    def sort(self, place: Literal['start', 'end']):
        getattr(self, place).sort(key=lambda x: x._hook_place)
    
    def append(self, hook):
        assert isinstance(hook, BaseHook)
        self.hooks.append(hook)
        self.hooks.sort(key=lambda x: x._hook_place)

    def clear(self):
        self.hooks.clear()

    def check(self, additional_hooks):
        self._check_specific(additional_hooks)
        # and other checks

    def _check_specific(self, hooks):
        for place in ['start', 'end', ]:
            iterable_hooks = chain(*hooks)
            hook_classes = tuple(hook.__class__ for hook in getattr(self, place))
            for specific_hook in iterable_hooks:
                if specific_hook.value in hook_classes:
                    return True
        return False

    def __repr__(self):
        return "Training Scheme:\nEpoch start:\n\t{}\n<<<Training>>>\nEpoch end\n\t{}".format(
            '\n\t'.join(str(hook) for hook in self.start), 
            '\n\t'.join(str(hook) for hook in self.end))
