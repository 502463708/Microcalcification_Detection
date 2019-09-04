import os


class Logger(object):
    def __init__(self, log_saving_dir):
        assert os.path.isdir(log_saving_dir)
        log_saving_path = os.path.join(log_saving_dir, 'log.txt')
        self.logger = open(log_saving_path, "w", encoding="utf-8")

        return

    def write(self, message):
        self.logger.write(message)
        self.logger.write('\n')

        return

    def flush(self):
        self.logger.flush()

        return
