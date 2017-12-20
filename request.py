from enum import Enum
from util import *


# To introduce latency metric,
# we define request object here
class Request:

    ID = Counter()
    INVALID_LOCATION = -1
    INVALID_LATENCY = -2
    # states
    NOT_ENTERED = 0
    ENTERED = 1
    ADMITTED_IN = 2
    ADMITTED_OUT = 3
    FINISHED = 4
    DROPPED = 5

    # N.B. arr_time and finished_time are
    # counted as units of time slots.
    #
    # state indicates the location of current request:
    # - if NOT_ENTERED, then the request does not even enter the system
    # - if ENTERED, then the request enters but is not admitted
    # (in fact, in the simulation, the request is generated only when
    #  entering the system)
    # - if ADMITTED, then the request is ad
    # - if FINISHED, then the request has already been finished
    #
    # loc: [sc, f, s]
    def __init__(self, arr_time):
        self.arr_time = arr_time
        self.finished_time = -1
        self.id = Request.ID.get_and_inc()
        self.state = Request.ENTERED
        self.loc = [-1, -1, -1]

    def is_finished(self):
        return self.finished_time > 0

    def admit(self):
        if self.state is Request.ENTERED:
            self.state = Request.ADMITTED_IN

    def transfer(self):
        if self.state == Request.ADMITTED_IN:
            self.state = Request.ADMITTED_OUT
        elif self.state == Request.ADMITTED_OUT:
            self.state = Request.ADMITTED_IN
        else:
            pass

    def drop(self):
        self.state = Request.DROPPED

    def set_loc(self, sc_id, vnf_id, server_id):
        self.loc = [sc_id, vnf_id, server_id]

    def set_finished_time(self, t):
        self.finished_time = t

        if t > 0:
            self.state = Request.FINISHED

    def get_latency(self):
        if self.is_finished():
            if self.finished_time >= self.arr_time:
                return self.finished_time - self.arr_time + 1
            else:
                return 0
        else:
            return Request.INVALID_LATENCY


class Policy(Enum):
    FIFO = 0
    LIFO = 1


class RequestQueue:

    # Service principle is FIFO by default
    # Actual queue order is reversed
    def __init__(self, srv_policy=Policy.FIFO):
        self.queue = []
        self.policy = srv_policy

    def length(self):
        return len(self.queue)

    def append(self, req):
        self.queue.append(req)

    def append_reqs(self, reqs):
        if reqs is not None and len(reqs) > 0:
            self.queue += reqs
        else:
            return False

    # params:
    # - t: time slot index of service
    # - req_num: number of requests being actually processed
    def serve(self, req_num):
        served_reqs = []

        if req_num > 0 and self.length() > 0:

            for i in range(req_num):
                req = None

                if self.policy is Policy.FIFO:
                    req = self.queue[0]
                    self.queue = self.queue[1:]

                elif self.policy is Policy.LIFO:
                    req = self.queue.pop()

                else:
                    raise Exception("Unknown scheduling policy.")

                assert req is not None
                served_reqs.append(req)

        return served_reqs

    # make a snapshot of existing reqs
    def record_reqs(self):
        return self.queue + []

    def withdraw_all_reqs(self):
        reqs_tmp = self.record_reqs()
        self.queue = []

        return reqs_tmp

    def remove(self, req):
        if req is not None and req in self:
            self.queue.remove(req)
            return True
        else:
            return False

    def __gt__(self, other):
        return self.length() > other.length()

    def __lt__(self, other):
        return self.length() < other.length()

    def __contains__(self, req):
        return req in self.queue


# if __name__ == '__main__':
#     import numpy as np
#
#     q1 = RequestQueue()
#     q2 = RequestQueue()
#     q3 = RequestQueue()
#
#     q1.append(Request(1))
#     q1.append(Request(2))
#     q2.append(Request(3))
#
#     print(q1.length(), q2.length(), q3.length())
#     print(np.argmin([q1, q2, q3]))
