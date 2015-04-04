"""simple answers to interview questions"""
from collections import namedtuple

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

#5. implement binary search

def test_binary_search():
    def binary_search(iterable, search_val):
        if not iterable:
            return None

        middle_index = len(iterable) / 2
        middle_val = iterable[middle_index]

        if middle_val == search_val:
            return middle_index
        elif middle_val < search_val:
            return binary_search(iterable[middle_index + 1:], search_val)
        else:
            return binary_search(iterable[: middle_index], search_val)

    assert binary_search([1, 2, 3, 4, 5, 6, 7], 4) == 3

#6. Implement Djikstra's algorithm

def test_djikstra():
    Edge = namedtuple('Edge', ('end', 'cost'))

    # straight line directed graph
    # map from node to directed edges

    graph = {
        1: (Edge(end=2, cost=1),),
        2: (Edge(end=3, cost=1),),
        3: (Edge(end=4, cost=1),),
        4: (Edge(end=5, cost=1),),
        5: (Edge(end=6, cost=1),),
        6: (Edge(end=7, cost=1),),
        7: (Edge(end=8, cost=1),),
        8: (Edge(end=9, cost=1),),
        9: (Edge(end=10, cost=1),),
        10: tuple(),
    }

    # given a start node and end node, return nodes on the shortest path and the total cost of that path

    def get_next_node(tentative_costs, unvisited):
        min_cost = float('inf')
        min_cost_node = None

        for node in unvisited:
            temp_cost = tentative_costs[node]
            if temp_cost < min_cost:
                min_cost = temp_cost
                min_cost_node = node

        if min_cost == float('inf'):
            min_cost_node = None

        return min_cost_node

    def get_previous_nodes(graph, current):
        previous_nodes = []

        for node, edges in graph.iteritems():
            for edge in edges:
                if edge.end == current:
                    previous_nodes.append(node)
                    break

        return previous_nodes


    def get_previous_node_on_path(graph, current, tentative_costs):
        previous_nodes = get_previous_nodes(graph, current)
        return get_next_node(tentative_costs, previous_nodes)


    def djikstra(graph, start, end):
        # map nodes to tentative costs
        tentative_costs = {node: 0 if node == start else float('inf') for node in graph.keys()}
        unvisited = [node for node in graph.keys()]

        while True:
            node = get_next_node(tentative_costs, unvisited)

            if node is None:
                break

            node_index = unvisited.index(node)
            node = unvisited.pop(node_index)
            node_cost = tentative_costs[node]
            edges = graph[node]

            for end, edge_cost in edges:
                current_cost = tentative_costs[end]
                new_cost = node_cost + edge_cost
                if new_cost < current_cost:
                    tentative_costs[end] = new_cost

        total_cost = tentative_costs[end]
        path = [end]

        current = end
        while current != start:
            current = get_previous_node_on_path(graph, current, tentative_costs)
            path.insert(0, current)

        return total_cost, path

    assert djikstra(graph, 1, 10) == (9, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

#7. implement a DFA state machine

def test_dfa_state_machine():
    """
    This machine is idle by default.

    If the button is pressed once, it starts flashing.
    If the button is pressed twice, it buzzes.
    If the button is pressed a third time, it goes back to idle.

    If the emergency switch is pressed at any time, the machine resets its state and goes idle.
    """

    dfa = {
        'idle': lambda action: 'flashing' if action == 'button_press' else 'idle',
        'flashing': lambda action: 'buzzing' if action == 'button_press' else 'idle',
        'buzzing': lambda action: 'idle',
    }

    class dfa_machine(object):
        def __init__(self, dfa, state):
            self.dfa = dfa
            self.state = state

        def transition(self, action):
            if action:
                self.state = dfa[self.state](action)

        def get_state(self):
            return self.state

    action_state_sequence = (
        (None, 'idle'),
        ('button_press', 'flashing'),
        (None, 'flashing'),
        ('button_press', 'buzzing'),
        ('button_press', 'idle'),
        ('button_press', 'flashing'),
        ('emergency_switch', 'idle')
    )

    machine = dfa_machine(dfa, 'idle')

    for action, result_state in action_state_sequence:
        machine.transition(action)
        assert machine.get_state() == result_state()