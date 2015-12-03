# Practice for Facebook's phone interview.
from collections import defaultdict


class ListNode(object):
    def __init__(self, x, n=None):
        self.val = x
        self.next = n


class TreeNode():
    def __init__(self, x, l=None, r=None):
        self.val = x
        self.left = l
        self.right = r


buttons = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', 
           '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz', '0': ' '}

def digit_to_text(digits):
    # How to handle exception?
    # O(m^n) m = averge letter (3), n = |digits|.
    if len(digits) == 1:
        try:
            return [char for char in buttons[digits[0]]]
        except KeyError:
            return []

    ans = []
    combinations = digit_to_text(digits[1:])
    try:
        for combination in combinations:
            for char in buttons[digits[0]]:
                ans.append(char + combination)
        return ans
    except KeyError:
        return combinations

def digit_to_text2(digits):
    # Iterative.
    # O(m^n)
    results = ['']
    if not digits or not digits.isdigit():
        return results

    for digit in digits:
        if digit == '1':
            continue

        temp = []
        for letter in buttons[digit]:
            for result in results:
                temp.append(result + letter)
        results = temp
    return results
# digits = '223'
# print digit_to_text2(digits)

def two_sum(nums, target):
    # O(n), no need to sort.
    exist = defaultdict(list)
    for i, n in enumerate(nums):
        exist[n].append(i)

    ans = []
    for i, n in enumerate(nums):
        key = target - n
        if key in exist:
            possible_indices = exist[key]
            for j in possible_indices:
                if i != j:
                    candidate = sorted([i, j])
                    if candidate not in ans:
                        ans.append(candidate)
    return [(nums[i], nums[j]) for (i, j) in ans]

def three_sum(nums, target):
    # O(nlogn)
    nums.sort()

    ans = []
    for i, n in enumerate(nums):
        if i >= 1 and nums[i] == nums[i-1]:
            continue

        j = i + 1
        k = len(nums) - 1
        while j < k:
            x = n + nums[j] + nums[k]
            if x == target:
                ans.append([n, nums[j], nums[k]])
                j += 1
                k -= 1
                while nums[j] == nums[j-1]:
                    j += 1
                while nums[k] == nums[k+1]:
                    k -= 1
            elif x < target:
                j += 1
            else:
                k -= 1
    return ans

def four_sum(nums, target):
    # O(n^2)
    pairs = defaultdict(list)
    n = len(nums)
    for i in range(n):
        for j in range(i+1, n):
            pairs[nums[i] + nums[j]].append([i, j])

    ans = []
    for value, index_list in pairs.iteritems():
        print value, index_list
        for (k, l) in pairs[target - value]:
            for (i, j) in index_list:
                indice = sorted([i, j, k, l])
                if indice not in ans:
                    ans.append(indice)

    return [[nums[i], nums[j], nums[k], nums[l]] for (i, j, k, l) in ans]
# print four_sum([1, 2, 3, 4, 3, 1], 5)

def point_in_the_most_intervals(intervals):
    # Assume intervals are a list [first, second]
    # O(n)
    points = {}

    for interval in intervals:
        for point in interval:
            if point not in points:
                points[point] = 0

    for interval in intervals:
        for point in points.keys():
            if interval[0] <= point <= interval[1]:
                points[point] += 1

    return max(points.keys(), key=lambda x: points[x])
# intervals = [[0, 1], [1, 2], [5, 10], [9, 11], [5, 100]]
# print point_in_the_most_intervals(intervals)

# Repeat character k times.
def repeat(chars, k):
    # chars is a set of characters.
    # O(n^k)
    if k == 0:
        return [[]]

    ans = []
    for char in chars:
        for prev in repeat(chars - set(char), k - 1):
            ans.append([char] + prev)
    return ans


def convert(string_list):
    ans = []
    for s in string_list:
        ans.append(''.join(s))
    return ans
# chars = {'a', 'b', 'c'}
# k = 3
# print convert(repeat(chars, k))

# Compute the shortest time to finish a task sequence.
def compute_time(tasks, k):
    # Greedy algorithm. O(n)
    # The same tasks have to be seperated by k time.
    time = 0
    last = {}  # Last index of a task
    for i, task in enumerate(tasks):
        if task in last:
            n = i - last[task]
            if n <= k:
                time += k + 1 - n
        else:
            last[task] = i
        time += 1
    return time
# tasks = 'aa'
# k = 2
# print compute_time(tasks, k)

# Does the strategy have reseanable guess?
def catch_theft(n, strategy):
    # O(nm)
    if n <= 1:
        return True

    # Today's free space for the theft
    today = [True] * n
    today[strategy[0]] = False
    for looking_for in strategy[1:]:
        yesterday = today[:]  # Make a new copy.
        for j in range(n):
            # Looking for free space for the theft.
            if j == 0:
                today[j] = yesterday[j + 1]
            elif j == n - 1:
                today[j] = yesterday[j - 1]
            else:
                today[j] = yesterday[j - 1] or yesterday[j + 1]

            # If there is a free space arround, and the police is
            # not looking for the current place j, then j is free.
            today[j] &= looking_for != j
            print today

        if not any(today):
            return True
    return False
# strategy = [0, 1, 2, 3, 4, 4, 3, 2, 1, 0]
# n = 5
# print catch_theft(n, strategy)

# Only check three points is not enought for validate a BST.
#       3
#    2     6
#  1 '4' <- not valid
def is_valid_bst(root, min_node=None, max_node=None):
    # O(n)
    if not root:
        return True
    if ((min_node and min_node.val >= root.val) or
            (max_node and max_node.val <= root.val)):
        return False

    return (is_valid_bst(root.left, min_node, root) and
            is_valid_bst(root.right, root, max_node))

# None repeated products in any subset of prime.
def products(primes):
    # O(2^n)
    subsets = [[]]
    for prime in primes:
        subsets += [s + [prime] for s in subsets]

    ans = []
    for subset in subsets[1:]:
        ans.append(reduce(lambda x, y: x * y, subset))
    return list(set(ans))  # If duplicates are in the input.

# BST to LinkedList
class BST_iterator():
    # O(logn)
    def __init__(self, root):
        self.stack = []
        self.visit_the_left_most(root)

    def visit_the_left_most(self, root):
        while root:
            self.stack.append(root)
            root = root.left

    def has_next(self):
        return True if self.stack else False

    def next(self):
        node = self.stack.pop()
        if node.right:
            self.visit_the_left_most(node.right)
        return node

def tree_to_list(root):
    # O(n)
    iter = BST_iterator(root)
    dummy = curr = ListNode(0)
    while iter.has_next():
        curr.next = ListNode(iter.next().val)  # [BUG] iter.next()
        curr = curr.next
    return dummy.next
# root = TreeNode(3, TreeNode(2, TreeNode(1)), TreeNode(4))
# head = tree_to_list(root)
# while head:
#     print head.val
#     head = head.next

# Judge if a "directive graph" has a cycle.
TO_BE_VISITED = 0
VISITING = 1
DONE = 2

def has_cycle(graph):
    # O(|v| + |e|)
    nodes = set()
    for node in graph:
        nodes.add(node)
        for neighbor in graph[node]:
            nodes.add(neighbor)

    status = [TO_BE_VISITED] * len(nodes)

    for vertex in graph.keys():
        if status[vertex] != DONE:
            if not dfs_visit(vertex, graph, status):
                return False
    return True


def dfs_visit(vertex, graph, status):
    # Return True if no cycle else False
    status[vertex] = VISITING

    for node in graph[vertex]:
        if status[node] == VISITING:
            return False
        if status[node] == TO_BE_VISITED:
            if not dfs_visit(node, graph, status):
                return False

    status[vertex] = DONE
    return True
# graph = defaultdict(list)
# graph[0] = [1, 2]
# graph[2] = [3, 4]
# graph[3] = [1]
# print has_cycle(graph)

# Shortest snippet. Look for a snippet between [snip_head, snip_tail]
def shortest_snippet(string, looking_for):
    # O(n)
    snip_head, snip_tail = looking_for
    head_index, tail_index = -1, -1

    snippets = []
    for i, s in enumerate(string):
        if s == snip_head:
            head_index = i
        elif s == snip_tail and head_index != -1:
            tail_index = i

        if head_index != -1 and tail_index != -1:
            snippets.append(((head_index, tail_index),
                             tail_index - head_index))
            head_index, tail_index = -1, -1
    (head_index, tail_index), length = min(snippets, key=lambda s: s[1])
    return string[head_index:tail_index + 1]

# If a and b can switch.
def shortest_snippet2(string, looking_for):
    # O(n)
    head_index, tail_index = -1, -1
    snippets = []
    i = 0
    while i < len(string):
        s = string[i]
        if s in looking_for:
            if head_index == -1:
                head_index = i
            else:
                tail_index = i

        if head_index != -1 and tail_index != -1:
            snippets.append(((head_index, tail_index),
                             tail_index - head_index))
            head_index, tail_index = -1, -1
            i -= 1  # The tail can be the head again.
        i += 1
    (head_index, tail_index), length = min(snippets, key=lambda s: s[1])
    return string[head_index:tail_index + 1]
# string = 'axxbaxb'
# looking_for = ['a', 'b']
# print shortest_snippet2(string, looking_for)

def reorderList(self, head):
    """
    :type head: ListNode
    :rtype: void Do not return anything, modify head in-place instead.
    """
    # O(n)
    if not head or not head.next:
        return

    # Seperate the list in the midpoint. Midpoint is the last node of the
    # first half.
    fast_curr = curr = head
    while fast_curr.next and fast_curr.next.next:
        curr = curr.next
        fast_curr = fast_curr.next.next
    midpoint = curr

    # Reverse the list after the midpoint.
    new_head = reverse(curr.next)
    midpoint.next = None

    # Alternatively link two lists.
    curr = ListNode(0)
    alternative = False
    while head or new_head:
        if alternative:
            curr.next = new_head
            new_head = new_head.next
        else:
            curr.next = head
            head = head.next
        curr = curr.next
        alternative = not alternative


def reverse(self, head):
    # O(n)
    if not head and not head.next:
        return head

    dummy = ListNode(0)
    dummy.next = head
    prev, last, curr = dummy, head, head.next
    while curr:
        last.next = curr.next
        curr.next = prev.next
        prev.next = curr
        curr = last.next
    return dummy.next

# Determine if a meeting schedule has a conflict.
def has_conflict(intervals):
    # O(nlogn)
    intervals.sort(key=lambda x: x[0])
    candidate = intervals[0]
    for interval in intervals[1:]:
        if candidate[0] <= interval[0] <= candidate[1]:
            return True
    return False

def has_conflict2(intervals):
    # O(nm), if all intervals are interger.
    table = defaultdict(int)
    for interval in intervals:
        for i in range(interval[0], interval[1]):
            table[i] += 1
        if min(table.values()) > 1:
            return False
    return True

def min_room(intervals):
    # O(nlogn)
    times = []
    for interval in intervals:
        times.append(interval[0])
        times.append(-interval[1])  # Encode the leaving.

    times.sort(key=lambda x: abs(x))

    room = 0
    room_need = 0
    for time in times:
        if time > 0:
            room += 1
            room_need = max(room_need, room)
        else:
            room -= 1
    return room_need
# intervals = [[1, 2], [2, 3], [2, 4]]
# print has_conflict(intervals)
# print min_room(intervals)

#   1
#  2 5
# 34  6
# Flatten to right child
def flatten_tree(root):
    stack = [root]
    while stack:
        # Inorder traversal
        node = stack.pop()
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)

        node.right = stack[-1] if stack else None
        node.left = None

def tree_to_doubly_linked_list(curr, prev=None, head=None):
    # O(n)
    if not curr:
        return prev, head

    # Inorder visit the left most node.
    prev, head = tree_to_doubly_linked_list(curr.left, prev, head)

    # 1 <- 2 <- 3 ...
    # current node's left points to the previous node.
    curr.left = prev
    if prev:
        prev.right = curr  # previous node's right points to curr.
    else:
        head = curr

    # head's left always point to the latest (curr) point.
    # The lastest's right point (curr) always point to head.
    right = curr.right  # Wait for traverse later.
    head.left = curr
    curr.right = head

    # Traverse to the next node.
    prev = curr
    prev, head = tree_to_doubly_linked_list(right, prev, head)
    return (prev, head)
root = TreeNode(4, TreeNode(3, TreeNode(1), TreeNode(2)), TreeNode(5))
prev, head = None, None
prev, head = tree_to_doubly_linked_list(root, prev, head)
# prev, head = tree_to_doubly_linked_list(root)
while head:
    print head.val
    head = head.right

# Build a suffix array
# banana -> [5, 3,   1,     0,      4, 2]
# 012345     a  ana  anana  banana  na nana
def build_suffix_array(text):
    # O(nlogn)
    suffix_array = sorted(range(len(text)), key=lambda i: text[i:])
    return suffix_array

def search(text, pattern, suffix_array):
    # O(logn)
    i, j = 0, len(text) - 1
    while i <= j:
        m = i + (j - i) / 2
        s = suffix_array[m]
        if pattern == text[s:s+len(pattern)]:
            return suffix_array[m]
        if pattern > text[s:s+len(pattern)]:
            i = m + 1
        else:
            j = m - 1
    return -1
# text = 'banana'
# pattern = 'nan'
# suffix_array = build_suffix_array(text)
# print search(text, pattern, suffix_array)

# Make invalid parentheses string to valid.
def transform_to_valid_parentheses(string):
    # O(n)
    string = list(string)

    # (A) mutually exclusive (B): To catch more ')'
    l, r = 0, 0
    for i, s in enumerate(string):
        if s == '(':  # Go from the left
            l += 1
        else:
            r += 1
        if r > l:  # Invalid
            string[i] = '*'
            r -= 1
    # print ''.join(string)

    # (B) mutually exclusive (A): To catch more '('
    l, r = 0, 0
    for i, s in enumerate(list(reversed(string)), 1):
        if s == '(':
            l += 1
        else:
            r += 1
        if l > r:
            string[-i] = '*'
            l -= 1
    # print ''.join(string)

    ans = []
    for s in string:
        if s != '*':
            ans.append(s)
    return ''.join(ans)
# string = '(())())'
# print transform_to_valid_parentheses(string)

def random_max_index(nums):
    # O(n) time. O(n) space.
    import random
    max_num = max(nums)
    indices = []  # O(n) space
    for i, n in enumerate(nums):
        if n == max_num:
            indices.append(i)
    return indices[random.randrange(len(indices))]

def random_max_index2(nums):
    # O(n) time. O(1) space.
    import random
    max_num = max(nums)
    count = 0
    for i, n in enumerate(nums):
        if n == max_num:
            count += 1
            discard = random.randrange(count)
            # The probability of returning k-th max is 1 / k is hold
            # for a length n array. If the n + 1 element is the
            # largest element as well, the probablity of returning
            # the same index from the first n element without
            # substututed by the k + 1 max is (1 / k + 1).
            # [k max ...]  [k+1 max]
            # chose k       not choose k + 1
            # (1 / k)    * (k / k + 1)
            if not discard:
                index = i
    return index
# nums = [1, 2, 3, 4, 5, 5, 5, 5]
# print random_max_index2(nums)

def eliminate_comments(code):
    # O(n)
    if len(code) < 2:
        return code

    ans = ''
    comment_flag = False
    i = 0
    while i < len(code):
        if not comment_flag:
            if code[i] == '/' and code[i + 1] == '*':
                comment_flag = True
                i += 1
            else:
                ans += code[i]
        else:
            if code[i] == '*' and code[i + 1] == '/':
                comment_flag = False
                i += 1
        i += 1
    return ans
# code = ('/*this is a comment*/\n' +
#         'int main(int argc, char** argv) {\n'
#         '  printf("Hello Word!");\n' +
#         '  /*\n' +  # Two space here is hard to eliminate
#         '  return 0;\n' +
#         '  */' +
#         '}\n')
# print eliminate_comments(code)

def remove_duplicates(nums):
    # O(n), duplicates are near each other.
    i = 0
    for n in nums:
        if i < 1 or nums[i - 1] != n:
            nums[i] = n
            i += 1
    return i
# nums = [2, 2, 2, 1, 5, 6, 6, 7, 7]
# print remove_duplicates(nums)
# print nums

def remove_duplicates2(nums):
    # O(n), sorted nums.
    ALLOWED_DUPLICATE = 2
    i = 0
    for n in nums:
        if i < ALLOWED_DUPLICATE or n != nums[i - ALLOWED_DUPLICATE]:
            nums[i] = n
            i += 1
    return i
# nums = [1, 1, 1, 2, 2, 3]
# print remove_duplicates2(nums)
# print nums

def move_zeros(nums):
    # O(n)
    i = 0
    for j in range(len(nums)):
        if nums[j] != 0:
            # Move non-zeros to the front.
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
# If swap is expensive,
def move_zeros2(nums):
    # O(n)
    i = 0
    for j in range(len(nums)):
        if nums[j] != 0:
            nums[i] = nums[j]
            i += 1
    for j in range(i, len(nums)):
        nums[j] = 0
# nums = [1, 0, 2]
# move_zeros(nums)
# print nums

# Find a minimum window in text that contains all char in the pattern.
def min_window(text, pattern):
    # Time O(n). Improve the brute-force method O(n^2).
    from collections import Counter
    need = Counter(pattern)
    missing = len(pattern)

    i = 0
    l, r = 0, 0
    for j, char in enumerate(text, 1):  # Check text[i:j]
        missing -= 1 if need[char] > 0 else 0
        need[char] -= 1

        if not missing:  # Have all char included

            # Shirk the window if we don't need a char.
            while i < j and need[text[i]] < 0:
                need[text[i]] += 1
                i += 1

            # Update the result.
            if not r or j - i < r - l:
                l, r = i, j
    return text[l:r]
# text = 'axxxaxxba'
# pattern = 'ab'
# print min_window(text, pattern)

# What to return if we couldn't find an order?
def topological_sort(graph):
    # O(|n| + |e|)
    # Count indegree.
    num_vertex = len(graph.keys())

    in_degree = defaultdict(int)
    for v in range(num_vertex):
        for n in graph[v]:
            in_degree[n] += 1

    # Make a list of zero in-degree.
    zero_in_degree_vertexs = []
    for v in range(num_vertex):
        if v not in in_degree:
            in_degree[v] = 0
            zero_in_degree_vertexs.append(v)

    # Take out a zero in-degree node and remove all edge from that node.
    order = []
    while zero_in_degree_vertexs:
        v = zero_in_degree_vertexs.pop()
        order.append(v)
        for n in graph[v]:
            in_degree[n] -= 1
            if in_degree[n] == 0:
                zero_in_degree_vertexs.append(n)
    return order if len(order) == num_vertex else []


def build_graph(num_course, edges):
    # O(n)
    graph = defaultdict(list)
    for (end, start) in edges:
        graph[start].append(end)

    for n in range(num_course):
        if n not in graph:
            graph[n] = []
    return graph
# num_course = 4
# edges = [[0, 1], [3, 1], [1, 3], [3, 2]]
# graph = build_graph(num_course, edges)
# print topological_sort(graph)

# Best time to buy and sell stock
def max_profits(prices, k=1, fee=0):
    # O(nk)
    if not prices:
        return 0

    balance = [-max(prices)] * (k + 1)
    profits = [0] * (k + 1)
    for price in prices:
        for i in range(1, k + 1):
            # Buy the stock
            balance[i] = max(balance[i], profits[i - 1] - price - fee)
            # Sell the stock
            profits[i] = max(profits[i], balance[i] + price)
    return profits[k]
# prices = [1, 3, 6, 1, 10]
# k = 2
# fee = 3
# print max_profits(prices, k, fee)

# Complexity is O(n), space is O(1).
def one_edit_distance(s, t):
    d = len(s) - len(t)
    for i, (a, b) in enumerate(zip(s, t)):
        if a != b:
            # If d == 0, compare the rest after this different pair.
            # If d > 0, we assume s has only one additional char,
            # but the rest is the same, so we jump 1 char.
            # For d < 0, is the same story but happend on t.
            return s[i + (d >= 0):] == t[i + (d <= 0):]
    # If all the same, s might be t's suffix, vise versa.
    return abs(d) == 1
# s = 'acbd'
# t = 'abd'
# print one_edit_distance(s, t)

def check_palindrome(string):
    # O(n)
    if not string:
        return False

    i, j = 0, len(string) - 1
    while i <= j:
        if string[i] != string[j]:
            return False
        i += 1
        j -= 1
    return True

def palindrome_substrings(string):
    # O(n^2)
    count = 0
    n = len(string)
    ans = []
    for i in range(len(string)):
        l, r = i, i
        # Even length palindrome
        while l >= 0 and r < n and string[l] == string[r]:
            ans.append(string[l:r+1])
            count += 1
            l -= 1
            r += 1
        # Odd length palindrome
        l, r = i, i + 1
        while l >= 0 and r < n and string[l] == string[r]:
            ans.append(string[l:r+1])
            count += 1
            l -= 1
            r += 1
    return ans
# string = 'abba'
# print palindrome_substrings(string)
