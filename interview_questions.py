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

    memoized_vals = {}

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

# 4. Implement pre-order, in-order, post-order, and breadth-first recursive traversals of a binary tree

def test_binary_tree_recursive_traversals():
    
    class BinaryTree(object):
        def __init__(self, value, left_child=None, right_child=None):
            self.value = value
            self.left_child = left_child
            self.right_child = right_child

        def get_val(self):
            return self.value

        def get_left_child(self):
            return self.left_child

        def get_right_child(self):
            return self.right_child

        def set_left_child(self, left_child):
            self.left_child = left_child

        def set_right_child(self, right_child):
            self.right_child = right_child

    # make the tree
    """
        1
       2 3
      4   5
    """

    head = BinaryTree(
        1,
        left_child=BinaryTree(
            2, left_child=BinaryTree(4)
        ),
        right_child=BinaryTree(
            3, right_child=BinaryTree(5)
        )
    )

    def pre_order_depth_first_search(node, q):
        if node:
            q.append(node.value)
            q.extend(pre_order_depth_first_search(node.get_left_child(), []))
            q.extend(pre_order_depth_first_search(node.get_right_child(), []))

        return q

    assert pre_order_depth_first_search(head, []) == [1, 2, 4, 3, 5]

    def in_order_depth_first_search(node, q):
        if node:
            q.extend(in_order_depth_first_search(node.get_left_child(), []))
            q.append(node.value)
            q.extend(in_order_depth_first_search(node.get_right_child(), []))

        return q

    assert in_order_depth_first_search(head, []) == [4, 2, 1, 3, 5]

    def post_order_depth_first_search(node, q):
        if node:
            q.extend(post_order_depth_first_search(node.get_left_child(), []))
            q.extend(post_order_depth_first_search(node.get_right_child(), []))
            q.append(node.value)

        return q

    assert post_order_depth_first_search(head, []) == [4, 2, 5, 3, 1]

    def breadth_first_search(node, q):

        if node:
            q.append(node.value)
            left_children = breadth_first_search(node.get_left_child(), [])
            right_children = breadth_first_search(node.get_right_child(), [])

            max_child_level = max(len(left_children), len(right_children))

            for x in range(max_child_level):
                for children in (left_children, right_children):
                    if len(children) > x:
                        q.append(children[x])
        return q

    assert breadth_first_search(head, []) == [1, 2, 3, 4, 5]