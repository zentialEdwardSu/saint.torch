# logcat from @bigsaltyfishes(Molyuu)
from datetime import datetime
import os

# This is a simple log cater
# class Logcat():
#     debug_mode:bool
#     def __init__(self, path: str = None,debug_mode:bool=True):
#         self.log_file = None
#         self.debug_mode = debug_mode
#         if path:
#             self.log_file = open(path, 'a+', encoding='utf-8')

#     def __del__(self):
#         if self.log_file:
#             self.log_file.close()
class Logcat():
    debug_mode:bool
    def __init__(self, path: str = None,debug_mode:bool=True, max_log_size=1024*1024):
        self.log_file = path
        self.size_limit = max_log_size
        self.debug_mode = debug_mode

    def __del__(self):
        pass

    def _check_file_size(self):
        if os.path.exists(self.log_file):
            return os.path.getsize(self.log_file)
        return 0

    def _rotate_log_file(self):
        current_size = self._check_file_size()
        if current_size >= self.size_limit:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            backup_file = f"{self.log_file}.{timestamp}"
            os.rename(self.log_file, backup_file)

    
    def cat(self, level: int, msg: str):
        '''
        Method to print log message
        
        Args:
            level: 0 as error, 1 as warning, 2 as info, 3 as debug
            msg:   Log message
        '''
        msg_type = ['Error', 'Warning', 'Info', 'Debug']
        msg = str(datetime.now()) + ' ' + msg_type[level] + ': ' + msg
        with open(self.log_file,"a+",encoding="utf-8") as lf:
            print(msg)
            print(msg, file=lf)

    # handy def for quick log
    def error(self,msg: str):
        self.cat(0,msg)

    def warn(self,msg: str):
        self.cat(1,msg)

    def info(self,msg: str):
        self.cat(2,msg)

    def debug(self,msg: str):
        '''
        Only effect when debug_mode=True
        '''
        if self.debug_mode:
            self.cat(3,msg)