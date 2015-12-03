# Digits to words
# Eg. 12 -> ['ab', 'l']

def digit_to_word(digit):
    word = chr(int(digit) + ord('a') - 1)
    return word if 'a' <= word <= 'z' else None


def digits_to_words(digits, index=0, attemp=[], ans=[]):
    if index == len(digits):
        ans.append(attemp)
        return

    if index < len(digits) and '1' <= digits[index] <= '9':
        digits_to_words(digits, index + 1,
                        attemp+[digit_to_word(digits[index])], ans)
    if index < len(digits) - 1 and '10' <= digits[index:index+2] <= '26':
        digits_to_words(digits, index + 2,
                        attemp+[digit_to_word(digits[index:index+2])], ans)
    return ans


def digits_to_words_iterative(digits):
    ans = []
    stack = [(0, '')]
    while stack:
        index, word = stack.pop()
        # print index, word
        if index == len(digits):
            ans.append(word)

        if index < len(digits) and '1' <= digits[index] <= '9':
            stack.append((index+1, word+digit_to_word(digits[index])))

        if (index < len(digits)-1 and '10' <= digits[index:index+2] <= '26'):
            stack.append((index + 2,
                          word+digit_to_word(digits[index:index+2])))
    return ans
# print digits_to_words_iterative('1234')
# print digits_to_words('1234')

# Cound tree width.
class TreeNode():
    def __init__(self, x, l=None, r=None):
        self.val = x
        self.left = l
        self.right = r


def tree_width(root):
    if not root:
        return 0
    return max([1 + tree_height(root.left) + tree_height(root.right),
                tree_width(root.left),
                tree_width(root.right)])


def tree_height(root):
    if not root:
        return 0  # Because two end nodes are counted.
    return 1 + max(tree_height(root.left), tree_height(root.right))

# root = TreeNode(1,
#                 TreeNode(2,
#                          TreeNode(4, TreeNode(5)),
#                          TreeNode(5, TreeNode(6, TreeNode(7)))))
# print tree_width(root)


# Quick select top k elements from numbers
import random


def quick_select(nums, k):
    start, end = 0, len(nums) - 1
    while start <= end:
        rank = partion(nums, start, end)
        if rank > k:
            end = rank - 1
        else:
            start = rank + 1
    return nums[:k]


def partion(nums, start, end):
    pivot = nums[random.randrange(start, end + 1)]
    bottom, middle, upper = start, start, end
    while middle <= upper:
        if nums[middle] > pivot:
            nums[middle], nums[bottom] = nums[bottom], nums[middle]
            middle += 1
            bottom += 1
        elif nums[middle] == pivot:
            middle += 1
        else:
            nums[middle], nums[upper] = nums[upper], nums[middle]
            upper -= 1
    return bottom
# nums = [8, 3, 1, 2, 5, 7, 4, 9, 0, 6]
# print quick_select(nums, 3)

# Find palindrome in subsequnce
# E.g 'cdc'-> 'c', 'd', 'cdc'
def check_palindrome(text, i, j, ans):
    while i >= 0 and j < len(text) and text[i] == text[j]:
        ans.add(text[i:j+1])
        i -= 1
        j += 1


def subsequence(text, ans=set()):
    for index in range(len(text)):
        check_palindrome(text, index, index, ans)
        check_palindrome(text, index, index+1, ans)
        # O(n!)
        subsequence(text[:index] + text[index + 1:], ans)
    return ans
text = 'cdcabc'
print subsequence(text)

# Common ancestor in BST.
def common_ancestor(root, p, q):
    path_p = search(p)
    path_q = search(q)

    common_node = None
    for node_p, node_q in zip(path_p, path_q):
        if node_p.val != node_q.val:
            break
        common_node = node_p
    return common_node.val


def search(root, p):
    if root.val == p.val:
        return [root]
    if root.val < p.val:
        return [root] + search(root.left, p)
    return [root] + search(root.right, p)

# Word break
def is_composited(text, dic):
    n = len(text)
    exist = [False] * (n + 1)
    exist[0] = True
    for i in range(1, n + 1):
        for j in range(i):
            if text[j:i] in dic and exist[j]:
                exist[i] = True
                break
    return exist[-1]
# text = 'northcarolina'
# dic = {'north', 'carolina'}
# print is_composited(text, dic)

# /\ -> 5 regions
# \/
from collections import namedtuple
Point = namedtuple('Point', ['row', 'column', 'side'])


def region(matrix):
    points = set()
    for i, row in enumerate(matrix):
        for j, element in enumerate(row):
            points.add(Point(i, j, 'left'))
            points.add(Point(i, j, 'right'))

    count = 0
    visited = set()
    for point in points:
        if point not in visited:
            visit(point, matrix, visited)
            count += 1
    return count


def move(point, d, matrix):
    m, n = len(matrix), len(matrix[0])
    dy, dx = d
    y, x = dy + point.row, dx + point.column

    if (0 <= y < m) and (0 <= x < n):
        side = point.side
        if d == (0, -1):
            side = 'right'
        elif d == (0, 1):
            side = 'left'
        elif d == (-1, 0):
            side = 'right' if matrix[y][x] == '/' else 'left'
        elif d == (1, 0):
            side = 'left' if matrix[y][x] == '/' else 'right'
        return Point(y, x, side)
    return None


def visit(point, matrix, visited):

    if point in visited:
        return
    visited.add(point)

    if point.side == 'left':
        # Can move left
        new_point = move(point, (0, -1), matrix)
        if new_point:
            visit(new_point, matrix, visited)

        if matrix[point.row][point.column] == '/':
            new_point = move(point, (-1, 0), matrix)
            if new_point:
                visit(new_point, matrix, visited)
        else:
            new_point = move(point, (1, 0), matrix)
            if new_point:
                visit(new_point, matrix, visited)
    else:
        # Can move right:
        new_point = move(point, (0, 1), matrix)
        if new_point:
            visit(new_point, matrix, visited)

        if matrix[point.row][point.column] == '/':
            new_point = move(point, (1, 0), matrix)
            if new_point:
                visit(new_point, matrix, visited)
        else:
            new_point = move(point, (-1, 0), matrix)
            if new_point:
                visit(new_point, matrix, visited)
# matrix = [['/', '\\'],
#           ['\\', '/']]
# print region(matrix)

# Find the number equal to its index.
def index_number(nums):
    # -1, -2, -3, -4
    # 0, 1, 2, 3
    left, right = -1, len(nums)
    while right - left > 1:
        middle = (right - left) / 2 + left
        if nums[middle] == middle:
            return nums[middle]
        if nums[middle] > middle:  # nums[right] > middle > numbers[left]
            right = middle
        else:
            left = middle
    return -1
# nums = [0, 0, 1, 1, 1, 2, 6]
# print index_number(nums)

# Maximum none adjency profits.
def max_profit(nums):
    if not nums:
        return nums

    n = len(nums)
    if n < 2:
        return nums[0]

    profits = [0] * n
    profits[0] = max(nums[0], 0)
    profits[1] = max(profits[0], nums[1])
    for i, num in enumerate(nums[2:], 2):
        profits[i] = max(profits[i - 2] + num, profits[i - 1])
    return profits[-1]
# nums = [-1, -1, -1]
# print max_profit(nums)


def num_unique_path(matrix):
    num_path = [0] * len(matrix[0])

    num_path[0] = 1
    for j, element in enumerate(matrix[0][1:], 1):
        num_path[j] = num_path[j - 1] if element == '0' else 0

    for i, row in enumerate(matrix[1:], 1):
        num_path[j] = 0 if row[0] == '1' else num_path[j]
        for j, element in enumerate(row[1:], 1):
            if element == '1':
                num_path[j] = 0
            else:
                num_path[j] += num_path[j - 1]
    return num_path[-1]

# Return an integer based on their weight.
import random


def weighted_select(pairs):
    table = {}
    start = 0
    for item, weight in pairs:
        for i in range(start, start + weight):
            table[i] = item
        start += weight
    return table[random.randrange(start)]


def weighted_select_streamly(paris):
    select = None
    prev, curr = 0, paris[0][1]
    for item, weight in paris:
        if random.uniform(0, curr) > prev:
            select = item
        prev = curr
        curr += weight
    return select
pairs = [(2, 3), (3, 5), (1, 7)]
print weighted_select_streamly(pairs)

# OOD: Design a class call third party request if our request > 10 in 1 sec.
import time
import datetime

class GoogleMapsClient(object):
    """3rd party maps client; we CANT EDIT THIS."""

    def __init__(self):
        self.requests_made = 0

    def make_request(self):
        self.requests_made += 1
        now = datetime.datetime.now().time()
        return "%d - %s - San Francisco" % (self.requests_made, now)

class RateLimitedClient(object):
    def __init__(self, limit):
        self.requests_limit = limit  # requests limit per second.
        self.last_check = None
        self.requests_made = 0
        self.third_party = GoogleMapsClient()

    def make_request(self):
        now = time.time()  # In second.

        if not self.last_check or now - self.last_check > 1:
            self.last_check = now
            self.requests_made = 0

        self.requests_made += 1

        if self.requests_made > self.requests_limit:
            time.sleep(1)
            return ''  # Drop the request, instead of holding it.

        return self.third_party.make_request()
# client = RateLimitedClient(2)
# for _ in range(10):
#     # 0, 1 sent
#     # 2, drop and wait
#     # 3, 4 sent
#     # 5, drop and wait
#     # 6, 7 sent
#     # 8, drop and wait
#     # 9 sent
#     print client.make_request()

import re

def reverse_text(text):
    tokens = re.findall('\w+|[.,!?;]', text)
    reversed_texts = re.findall('[\w]+', text)[::-1]

    text_index = 0
    for i, token in enumerate(tokens):
        if token not in '.,!?;':
            if i < len(tokens) - 1 and tokens[i + 1] not in '.,!?;':
                tokens[i] = reversed_texts[text_index] + ' '
            else:
                tokens[i] = reversed_texts[text_index]
            text_index += 1

    return ''.join(tokens)
# text = 'I have a dream.'
# reversed_text = 'dream a have I.'
# assert reverse_text(text) == reversed_text
