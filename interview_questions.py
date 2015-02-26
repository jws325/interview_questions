"""simple answers to interview questions"""


#  1. reverse an array

def test_reverse():

    def reverse(a):

        for index, value in enumerate(a):
            if index >= len(a) / 2:
                break

            replace_index = len(a) - 1 - index
            temp = a[replace_index]
            a[replace_index] = value
            a[index] = temp

        return a

    # test
    l = [1, 2, 3]
    assert(reverse(l) == [3, 2, 1])


#  2. create and reverse a singly linked list

def test_ll():

    class LL(object):
        next = None
        val = None

        def __init__(self, val):
            self.val = val
            super(LL, self).__init__()

        def set_val(self, val):
            self.val = val

        def get_val(self):
            return self.val

        def set_next(self, next):
            self.next = next

        def get_next(self):
            return self.next

    def reverse_ll(head):

        prev = None
        cur = head

        while cur is not None:
            temp = cur.get_next()
            cur.set_next(prev)
            prev = cur
            cur = temp

    # test
    first = LL(1)
    second = LL(2)
    third = LL(3)

    first.set_next(second)
    second.set_next(third)

    reverse_ll(first)

    assert(
        third.get_next().get_val() == 2
        and second.get_next().get_val() == 1
        and first.get_next() is None
    )