from collections import OrderedDict


class KeyedCache(object):
    def __init__(self, maxsize=1000):
        self.maxsize = maxsize
        self.data = OrderedDict()

    def __setitem__(self, key, value):
        if key in self.data:
            self.data[key] = value
        else:
            self.data[key] = value
            if self.maxsize >= 0 and len(self.data) > self.maxsize:
                self.data.popitem(last=False)

    def __getitem__(self, key):
        return self.data[key]

    def __repr__(self):
        return "{}({})".format(type(self).__name__, self.data)


if __name__ == '__main__':
    cache = KeyedCache(maxsize=-1)
    cache['a'] = 1
    cache['b'] = 2
    cache['c'] = 3
    cache['d'] = 4
    cache['e'] = 5
    print(cache)
    print(cache['e'])
