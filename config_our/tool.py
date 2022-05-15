import config_our.cluster_config as config
import threading

class MyThread(threading.Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        try:
            self.result = self.func(*self.args)
        except:
            raise Exception("thread fail!")

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None