from domain.processes.pena_sigmoid_process import PenaSigmoidProcess


class PenaSigmoid3pProcess(PenaSigmoidProcess):

    def __init__(self, gamma, l, M, initial_state=0):
        super().__init__(gamma, 1, 0, l, M, initial_state=initial_state)
