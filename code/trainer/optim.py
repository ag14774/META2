import torch
import torch.optim
from torch.optim import SGD


class MilestoneSchedulers(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_schedulers, milestones):
        assert len(lr_schedulers) == (len(milestones) + 1)
        milestones.append(float('inf'))
        self.lr_scheds = []
        for lr_sched in lr_schedulers:
            lr_constructor = lr_sched['constructor']
            lr_args = lr_sched['args']

            if lr_constructor:
                lr_sched = lr_constructor(optimizer, **lr_args)
            else:
                lr_sched = None

            self.lr_scheds.append(lr_sched)
        self.milestones = milestones
        self.optimizer = optimizer
        self.last_epoch = -1

    def get_active_sched(self, epoch):
        for i, m in enumerate(self.milestones):
            if epoch < m:
                break
        sched = self.lr_scheds[i]
        return sched

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        sched = self.get_active_sched(epoch)
        if sched:
            sched.step(epoch=epoch)

    def get_lr(self):
        epoch = self.last_epoch
        sched = self.get_active_sched(epoch)
        if sched:
            return sched.get_lr()
        else:
            return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {
            'milestones':
            self.milestones,
            'scheds': [
                sched.state_dict() if sched else None
                for sched in self.lr_scheds
            ]
        }

    def load_state_dict(self, state_dict):
        for p, k, sched, state in zip(self.milestones,
                                      state_dict['milestones'], self.lr_scheds,
                                      state_dict['scheds']):
            assert p == k
            if sched:
                sched.load_state_dict(state)
            else:
                assert state == None


class MultiLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_schedulers):
        assert isinstance(optimizer, MultiOpt)
        self.param_groups = sorted(optimizer.param_groups)
        self.lr_scheds = []
        for k, opt in zip(self.param_groups, optimizer.optimizers):
            if k in lr_schedulers:
                lr_constructor = lr_schedulers[k]['constructor']
                lr_args = lr_schedulers[k]['args']
            else:
                lr_constructor = lr_schedulers['default']['constructor']
                lr_args = lr_schedulers['default']['args']

            if lr_constructor and opt:
                lr_sched = lr_constructor(opt, **lr_args)
            else:
                lr_sched = None

            self.lr_scheds.append(lr_sched)
        self.last_epoch = -1

    def get_lr(self):
        lrs = {
            pg: sched.get_lr()
            for pg, sched in zip(self.param_groups, self.lr_scheds) if sched
        }
        return lrs

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for sched in self.lr_scheds:
            if sched:
                sched.step(epoch=epoch)

    def state_dict(self):
        return {
            'param_groups':
            self.param_groups,
            'scheds': [
                sched.state_dict() if sched else None
                for sched in self.lr_scheds
            ]
        }

    def load_state_dict(self, state_dict):
        for p, k, sched, state in zip(self.param_groups,
                                      state_dict['param_groups'],
                                      self.lr_scheds, state_dict['scheds']):
            assert p == k
            if sched:
                sched.load_state_dict(state)
            else:
                assert state == None


class MultiOpt(torch.optim.Optimizer):
    def __init__(self, params, optimizers):
        assert isinstance(optimizers, dict)
        assert isinstance(params, dict)

        self.param_groups = sorted(list(params.keys()))
        self.optimizers = []
        for k in self.param_groups:
            if k in optimizers:
                op_constructor = optimizers[k]['constructor']
                op_args = optimizers[k]['args']
            else:
                op_constructor = optimizers['default']['constructor']
                op_args = optimizers['default']['args']

            if op_constructor:
                opt = op_constructor(params[k], **op_args)
                self.optimizers.append(opt)
            else:
                self.optimizers.append(None)

    def zero_grad(self):
        for op in self.optimizers:
            if op:
                op.zero_grad()

    def step(self):
        for op in self.optimizers:
            if op:
                op.step()

    def state_dict(self):
        return {
            'param_groups': self.param_groups,
            'optimizers':
            [o.state_dict() if o else {} for o in self.optimizers]
        }

    def load_state_dict(self, state_dict):
        for p, k, opt, state in zip(self.param_groups,
                                    state_dict['param_groups'],
                                    self.optimizers, state_dict['optimizers']):
            assert p == k
            if opt:
                opt.load_state_dict(state)


class SGD_TF(SGD):
    def __init__(self,
                 params,
                 lr=0.1,
                 momentum=0,
                 weight_decay=0,
                 nesterov=False):
        super(SGD_TF, self).__init__(params,
                                     lr=lr,
                                     momentum=momentum,
                                     dampening=0,
                                     weight_decay=weight_decay,
                                     nesterov=nesterov)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state[
                            'momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(-group['lr'], d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(-group['lr'], d_p)
                    if nesterov:
                        p.data.add_(momentum, buf).add_(-group['lr'], d_p)
                    else:
                        p.data.add_(buf)
                else:
                    p.data.add_(-group['lr'], d_p)

        return loss
