import unittest

from domain.processes.poisson_process import PoissonProcess
from domain.queue_system import QueueSystem
from exceptions import UninitializedQueueException


class MMnTest(unittest.TestCase):

    # These first two test are most useless,
    # just written for development purposes
    # The correct way to test for stability
    # would be statistically

    def test_m_m_1_queue_not_unstable(self):
        arrivals_process = PoissonProcess(0.1)
        service_process = PoissonProcess(0.4)
        serv_number = 1
        q = QueueSystem(arrivals_process, service_process, serv_number)
        q.simulate_queue(1000)
        self.assertTrue(q.get_final_size() < 10)

    def test_m_m_1_queue_unstable(self):
        arrivals_process = PoissonProcess(0.6)
        service_process = PoissonProcess(0.4)
        serv_number = 1
        q = QueueSystem(arrivals_process, service_process, serv_number)
        q.simulate_queue(1000)
        self.assertTrue(q.get_final_size() > 100)

    def test_non_built_queue_cannot_have_final_size(self):
        arrivals_process = PoissonProcess(0.1)
        service_process = PoissonProcess(0.4)
        serv_number = 1
        q = QueueSystem(arrivals_process, service_process, serv_number)
        err_msg = 'Queue has to be simulated first'
        self.assertRaisesRegex(UninitializedQueueException, err_msg,
                               lambda: q.get_final_size())
