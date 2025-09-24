from itertools import chain 
from typing import Literal

from fedcore.models.network_impl.utils.hooks import BaseHook


class HooksCollection:
    def __init__(self, hooks=None):
        self._on_epoch_start = []
        self._on_epoch_end = []
        for hook in (hooks or []):
            self.append(hook)

    @property
    def start(self):
        return self._on_epoch_start
    
    @property
    def end(self):
        return self._on_epoch_end
        
    def sort(self, place: Literal['start', 'end']):
        getattr(self, place).sort(key=lambda x: x._hook_place)
    
    def append(self, hook):
        assert isinstance(hook, BaseHook)
        if hook._hook_place > 0:
            self._on_epoch_end.append(hook)
            self.sort('end')
        elif hook._hook_place < 0:
            self._on_epoch_start.append(hook)
            self.sort('start')
        else:
            self._on_epoch_start.append(hook)
            self._on_epoch_end.append(hook)
            self.sort('start')
            self.sort('end')

    def clear(self):
        self._on_epoch_end.clear()
        self._on_epoch_start.clear()

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
            '\n\t'.join(str(hook) for hook in self._on_epoch_start), 
            '\n\t'.join(str(hook) for hook in self._on_epoch_end))
