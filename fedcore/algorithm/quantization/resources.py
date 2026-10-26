"""Shared stage ledger; counters include failed work and explicit safe points."""
from dataclasses import dataclass
import time
import psutil
from fedcore.algorithm.low_rank.grammar_execution import ResourceFailure


class StageBudgetExceeded(RuntimeError):
    def __init__(self,code,message):
        self.failure = ResourceFailure(code,message)
        super().__init__(message)


@dataclass(frozen=True)
class StageBudgets:
    max_seconds: float = 60.
    max_training_steps: int = 32
    max_stages: int = 4096
    max_rss_bytes: int = 12*1024**3

    def __post_init__(self):
        import math
        if (type(self.max_seconds) not in (int,float) or not math.isfinite(self.max_seconds) or self.max_seconds<=0
                or any(type(v) is not int or v<0 for v in (self.max_training_steps,self.max_stages,self.max_rss_bytes))):
            raise ValueError('Finite stage budgets required')


class ResourceLedger:
    def __init__(self,budgets=StageBudgets()):
        self.budgets,self.started = budgets,time.perf_counter()
        self.training_steps,self.stages,self.events = 0,0,[]

    def checkpoint(self,stage,*,training_steps=0):
        now = time.perf_counter()
        event = {'stage':stage,'elapsed_seconds':now-self.started,'training_steps':training_steps,'status':'charged'}
        self.events.append(event)
        self.training_steps += training_steps; self.stages += 1
        code = ('StageDeadline' if now-self.started>self.budgets.max_seconds else
                'TrainingBudget' if self.training_steps>self.budgets.max_training_steps else
                'StageCountBudget' if self.stages>self.budgets.max_stages else
                'RSSBudget' if psutil.Process().memory_info().rss>self.budgets.max_rss_bytes else None)
        if code:
            event['status'] = 'refused'
            raise StageBudgetExceeded(code,'Shared stage resource budget exhausted')

    def evidence(self):
        return {'training_steps':self.training_steps,'stages':self.stages,'events':list(self.events),
                'wall_seconds':time.perf_counter()-self.started,'memory_semantics':'RSS admission samples; not measured peak'}
