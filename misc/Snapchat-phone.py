class TreeNode():
    def __init__(self, x, l=None, r=None):
        self.val = x
        self.left = l
        self.right = r


class ListNode():
    def __init__(self, x, n=None):
        self.val = x
        self.next = n


class BigNumber():
    def __init__(self, s1, s2):
        self.sign1 = 1 if s1[0] != '-' else -1
        self.sign2 = 1 if s2[0] != '-' else -1

    def calculate(self, case):
        if case == 'add':
            if self.sign1 == 1 and self.sign2 == 1:
                return self.add_string(s1, s2)
            if self.sign1 == -1 and self.sign2 == -1:
                return '-' + self.add_string(s1[1:], s2[1:])
            if self.sign1 == 1 and self.sign2 == -1:
                return self.minus_string(s1, s2[1:])
            if self.sign1 == -1 and self.sign2 == 1:
                return self.minus_string(s2, s1[1:])
        if case == 'mult':
            if self.sign1 == 1 and self.sign2 == 1:
                return self.multiply_string(s1, s2)
            if self.sign1 == -1 and self.sign2 == -1:
                return '-' + self.multiply_string(s1[1:], s2[1:])
            if self.sign1 == 1 and self.sign2 == -1:
                return self.multiply_string(s1, s2[1:])
            if self.sign1 == -1 and self.sign2 == 1:
                return '-' + self.multiply_string(s2, s1[1:])

    def add_string(self, s1, s2):
        i, j, carry = 1, 1, 0
        ans = ''
        while i <= len(s1) or j <= len(s2) or carry:
            x = (int(s1[-i] if i <= len(s1) else 0) +
                 int(s2[-j] if j <= len(s2) else 0) +
                 carry)
            ans += str(x % 10)
            carry = x // 10
            i += 1
            j += 1

        ans = ans[::-1]
        i = 0
        while ans[i] == '0':
            i += 1
        return ans[i:]

    def minus_string(self, s1, s2):
        i, j, carry = 1, 1, 0
        ans = ''
        while i <= len(s1) or j <= len(s2) or carry:
            x = (int(s1[-i] if i <= len(s1) else 0) -
                 int(s2[-j] if j <= len(s2) else 0) +
                 carry)
            if x < 0:
                x = 10 + x
                carry = -1
            else:
                carry = 0
            ans += str(x % 10)
            i += 1
            j += 1

        ans = ans[::-1]
        i = 0
        while ans[i] == '0':
            i += 1
        return ans[i:]

    def multiply_string(self, s1, s2):
        product = [0] * (len(s1) + len(s2))
        for i, m in enumerate(reversed(s1)):
            for j, n in enumerate(reversed(s2)):
                product[i + j] += int(n) * int(m)
                product[i + j + 1] += product[i + j] / 10
                product[i + j] %= 10
        while len(product) > 1 and product[-1] == 0:
            product.pop()
        return ''.join(map(str, reversed(product)))

#     def divide_string(self, s1, s2):
#         q = [0] * len(s1)
#         i, j = 0, 0
#         temp = 0
#         while i < len(s1) and s1[i]:
#             temp = temp * 10 + int(s1[i])
#             if temp < int(s2):
#                 q[j] = 0
#                 j += 1
#             else:
#                 q[j + 1] = temp / int(s2)
#                 temp %= int(s2)
#             i += 1
#         return ''.join(map(str, reversed(q)))
    # def divide_string(self, s1, s2):
    #     q = [0] * len(s1)
    #     for i in range(len(s1) - 1, 0, -1):
    #     i, j = 0, 0
    #     temp = 0
    #     while i < len(s1) and s1[i]:
    #         temp = temp * 10 + int(s1[i])
    #         if temp < int(s2):
    #             q[j] = 0
    #             j += 1
    #         else:
    #             q[j + 1] = temp / int(s2)
    #             temp %= int(s2)
    #         i += 1
    #     return ''.join(map(str, reversed(q)))
s1 = '1000'
s2 = '10'
big_number = BigNumber(s1, s2)
# print big_number.calculate('mult')
print big_number.divide_string(s1, s2)

# print add_string('09', '1')


def combination_sum(nums, target, attemp=[], ans=[]):
    if target == 0:
        ans.append(attemp)
        return ans

    for num in nums:
        if num <= target:
            combination_sum(nums, target - num, attemp + [num], ans)
    return ans
# print combination_sum([4, 3, 1], 5)


def print_diag(matrix):
    m = len(matrix)
    n = len(matrix[0])
    start_indices = ([(0, j) for j in range(n)] +
                     [(i, n-1) for i in range(1, m)])
    for (i, j) in start_indices:
        while i < m and j >= 0:
            print matrix[i][j],
            i += 1
            j -= 1
        print
# print_diag([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])


def has_anagram(text, pattern):
    from collections import Counter
    need = Counter(pattern)
    missing = n = len(pattern)

    for i, char in enumerate(text):
        missing -= 1 if need[char] >= 1 else 0
        need[char] -= 1
        if i >= n and text[i - n] in need:
            missing += 1
            need[char] += 1

        if not missing:
            return True
    return False
# print has_anagram('baban00a', 'babana')


class CycleDetector():
    def __init__(self, graph):
        self.TO_BE_VISITED = 0
        self.VISITING = 1
        self.DONE = 2

        self.graph = graph

    def has_cycle(self):
        status = [self.TO_BE_VISITED] * len(self.graph)
        for point in self.graph.keys():
            if (status[point] == self.TO_BE_VISITED and
                    self.visit(point, status)):
                return True
        return False

    def visit(self, point, status):
        status[point] = self.VISITING
        for neighbor in self.graph[point]:
            if status[neighbor] == self.VISITING:
                return True
            if (status[neighbor] == self.TO_BE_VISITED and
                    self.visit(neighbor, status)):
                return True
        status[point] = self.DONE
        return False
# graph = {0: [1], 1: [2], 2:[0]}
# detector = CycleDetector(graph)
# print detector.has_cycle()

def BST_to_doubly_linked_list(curr, head=None, prev=None):
    if not curr:
        return head, prev

    head, prev = BST_to_doubly_linked_list(curr.left, head, prev)

    curr.left = prev
    if prev:
        prev.right = curr
    else:
        head = curr

    right = curr.right
    head.left = curr
    curr.right = head

    prev = curr
    head, prev = BST_to_doubly_linked_list(right, head, prev)
    return head, prev
# root = TreeNode(4, TreeNode(3, TreeNode(1), TreeNode(2)),
#                 TreeNode(5))
# head, prev = BST_to_doubly_linked_list(root)
# for _ in range(5):
#     print head.val
#     head = head.right


def two_sum(nums, target):
    indices = {}
    for i, num in enumerate(nums):
        if target - num in indices:
            return indices[target - num], i
        indices[num] = i
    return -1, -1
# nums = [8, 4, 3, 2, 1, 5, 6, 7, 9]
# target = 4
# print two_sum(nums, target)


def three_sum(nums, target):
    original_indices = [i for i, num in sorted(enumerate(nums),
                                               key=lambda x: x[1])]
    nums.sort()
    for i, num in enumerate(nums):
        j = i + 1
        k = len(nums) - 1
        while j <= k:
            x = num + nums[j] + nums[k]
            if x == target:
                return (original_indices[i],
                        original_indices[j],
                        original_indices[k])
            if x > target:
                k -= 1
            else:
                j += 1
    return -1, -1, -1
# nums = [8, 4, 3, 2, 1, 5, 6, 7, 9]
# target = 12
# print three_sum(nums, target)
