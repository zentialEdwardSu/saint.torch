from datetime import datetime
import logging
from logging import Logger

CRITICAL = 50
FATAL = CRITICAL
ERROR = 40
WARNING = 30
WARN = WARNING
INFO = 20
DEBUG = 10
NOTSET = 0

# This is a simple log cater
class Logcat():
    def __init__(self, path: str = None,logger:logging.Logger = None):
        self.log_file = None
        self.logger_handler = logger
        if path:
            self.log_file = open(path, 'a+', encoding='utf-8')

    def __del__(self):
        if self.log_file:
            self.log_file.close()

    
    def cat(self, level: int, msg: str):
        '''
        Method to print log message
        
        Args:
            level: 0 as error, 1 as warning, 2 as info, 3 as debug
            msg:   Log message
        '''
        if self.log_file:
            msg_type = {ERROR:'Error', WARNING:'Warning', INFO:'Info', DEBUG:'Debug'}

            msg = str(datetime.now()) + ' ' + msg_type[level] + ': ' + msg
            print(msg)
            print(msg, file=self.log_file)
        elif self.logger_handler:
            self.logger_handler.log(level, msg)


def print_log(level: int, msg: str):
    with open("model.loh",'a+') as log_file:
        msg_type = {ERROR:'Error', WARNING:'Warning', INFO:'Info', DEBUG:'Debug'}

        msg = str(datetime.now()) + ' ' + msg_type[level] + ': ' + msg
        print(msg)
        print(msg, file=log_file)