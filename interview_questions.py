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


#  3. create a function that returns the nth zero-indexed element of the fibonacci sequence
#     use dynamic programming to memoize the sub problems

def test_fibonacci():

    memoized_vals = {0: 0, 1: 1}

    def get_memoized(n):
        if n in memoized_vals:
            return memoized_vals[n]
        else:
            return_val = fibonacci(n)
            memoized_vals[n] = return_val
            return return_val

    def fibonacci(n):
        if n < 0:
            raise ValueError('Negative indices are invalid')
        elif n < 2:
            return n
        else:
            return get_memoized(n-1) + get_memoized(n-2)

    # test

    vals_to_assert = (
        (0, 0),
        (1, 1),
        (2, 1),
        (3, 2),
        (4, 3),
        (5, 5),
        (6, 8),
    )

    for n, return_val in vals_to_assert:
        assert fibonacci(n) == return_val

    try:
        fibonacci(-1)
    except ValueError:
        pass
    else:
        raise RuntimeError('The fibonacci function failed to throw an error for the negative index')

test_fibonacci()