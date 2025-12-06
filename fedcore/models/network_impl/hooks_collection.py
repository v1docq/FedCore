from itertools import chain 
from typing import Iterable

from fedcore.models.network_impl.utils.hooks import BaseHook


class HooksCollection:
    def __init__(self, hooks=[]):
        self._on_epoch_start = []
        self._on_epoch_end = []
        for hook in hooks:
            self.append(hook)

    def start(self) -> list[BaseHook]:
        return self._on_epoch_start
    
    def end(self) -> list[BaseHook]:
        return self._on_epoch_end
    
    def all_hooks(self) -> list[BaseHook]:
        return self._on_epoch_start + self._on_epoch_end
    
    def _sort_start(self):
        self.start().sort(key=lambda hook: hook.HOOK_PLACE)

    def _sort_end(self):
        self.end().sort(key=lambda hook: hook.HOOK_PLACE)
    
    def append(self, hook: BaseHook):
        assert isinstance(hook, BaseHook)
        if hook.HOOK_PLACE > 0:
            self._on_epoch_end.append(hook)
            self._sort_end()
        elif hook.HOOK_PLACE < 0:
            self._on_epoch_start.append(hook)
            self._sort_start()
        else:
            self._on_epoch_start.append(hook)
            self._on_epoch_end.append(hook)
            self._sort_end()
            self._sort_start()

    def extend(self, hooks: Iterable[BaseHook]):
        for hook in hooks:
            if (hook.HOOK_PLACE > 0):
                self._on_epoch_end.append(hook)
            elif (hook.HOOK_PLACE < 0):
                self._on_epoch_start.append(hook)
            else:
                self._on_epoch_end.append(hook)
                self._on_epoch_start.append(hook)
        self._sort_end()
        self._sort_start()

    def clear(self):
        self._on_epoch_end.clear()
        self._on_epoch_start.clear()

    def check(self, additional_hooks):
        self._check_specific(additional_hooks)
        # and other checks

    def _check_specific(self, hooks):
        iterable_hooks = chain(*hooks)
        hook_classes = tuple(hook.__class__ for hook in self.all_hooks())
        for specific_hook in iterable_hooks:
            if specific_hook.value in hook_classes:
                return True
        return False

    def __repr__(self):
        return "Training Scheme:\nEpoch start:\n\t{}\n<<<Training>>>\nEpoch end\n\t{}".format(
            '\n\t'.join(str(hook) for hook in self._on_epoch_start), 
            '\n\t'.join(str(hook) for hook in self._on_epoch_end))
