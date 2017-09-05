import time

class clock(object):
    def __enter__(self):
        self.start = time.time()
        
    def __exit__(self, x,y,z):
        end = time.time()
        print('Elapsed time {} seconds'.format(end-self.start))
