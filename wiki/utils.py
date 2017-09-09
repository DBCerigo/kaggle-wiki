import time

def prevYear_shift(period):
    year = 365
    assert period[0]
    new1 = period[0] - year
    if not period[1]:
        new2 = -year
    else:
        new2 = period[1] - year
    return (new1,new2)

class clock(object):
    def __enter__(self):
        self.start = time.time()
        
    def __exit__(self, x,y,z):
        end = time.time()
        print('Elapsed time {} seconds'.format(end-self.start))
