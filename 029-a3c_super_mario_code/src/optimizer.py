import torch

class GlobalAdam(torch.optim.Adam):
    def __init__(self, params, lr):
        super(GlobalAdam, self).__init__(params, lr=lr, foreach=False)  # foreach=False 避免新API冲突
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                # 初始化状态
                state['step'] = torch.tensor(0.)  # 改成 tensor
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # 多进程共享
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()