from collections import defaultdict
from heapq import *
import random


class Square():
    def order_tasks(self, edges):
        """
        Topological Sort, but order the task with their priorities.
        O(v + e) time
        """
        # Count in-degree
        in_degree = defaultdict(int)

        # For nodes with non-zero in-degree
        nodes = set()
        for key, values in edges.iteritems():
            nodes.add(key)
            for value in values:
                in_degree[value] += 1
                nodes.add(value)

        # For nodes with zero in-degree
        zero_indegree = []
        for node in nodes:
            if node not in in_degree:
                zero_indegree.append(-node)  # For max heap

        # Use a max heap to maintain the largest proity task
        heapify(zero_indegree)  # O(n)

        order = []
        while zero_indegree:
            top_priority = -heappop(zero_indegree)
            order.append(top_priority)
            if top_priority in edges:
                for node in edges[top_priority]:
                    in_degree[node] -= 1
                    if in_degree[node] == 0:
                        heappush(zero_indegree, -node)
        return order

    def search_strings(self, matrix, dic):
        """
        DFS search a two dimensional matrix in 8 direction
        """
        m, n = len(matrix), len(matrix[0])

        ans = []
        for word in dic:
            for i in range(m):
                for j in range(n):
                    if self.found(word, i, j, matrix):
                        ans.append((i, j))
        return ans

    def found(self, word, i, j, matrix):
        if not word:
            return True

        if i < 0 or i >= len(matrix) or j < 0 or j >= len(matrix[0]):
            return False

        if matrix[i][j] == word[0]:
            return (self.found(word[1:], i-1, j, matrix) or
                    self.found(word[1:], i, j-1, matrix) or
                    self.found(word[1:], i+1, j, matrix) or
                    self.found(word[1:], i, j+1, matrix) or
                    self.found(word[1:], i-1, j-1, matrix) or
                    self.found(word[1:], i+1, j+1, matrix) or
                    self.found(word[1:], i-1, j+1, matrix) or
                    self.found(word[1:], i+1, j-1, matrix))
        return False

    def rotate_matrix(self, matrix):
        # 1 2 3    7 4 1
        # 4 5 6 -> 8 5 2
        # 7 8 9    9 6 3
        n = len(matrix)

        # Swap horizontally (|)
        for i in range(n):
            for j in range(n // 2):
                matrix[i][j], matrix[i][-j-1] = \
                    matrix[i][-j-1], matrix[i][j]

        # Sway diaganally (/)
        for i in range(n):
            for j in range(n):
                if i + j < n:
                    matrix[i][j], matrix[-j-1][-i-1] = \
                        matrix[-j-1][-i-1], matrix[i][j]

    def primes(self, n):
        if n < 2:
            return []

        primes, num = [2], 3
        while num <= n:
            if all(num % prime != 0 for prime in primes):
                primes.append(num)
            num += 1

        return primes

    def pair_socks(self, socks, color_diff=0):
        color_buckets = {}
        for item, side, color in socks:
            if color not in color_buckets:
                color_buckets[color] = {'left': [], 'right': []}
                color_buckets[color][side].append(item)
            else:
                color_buckets[color][side].append(item)

        pairs = []
        for color, buckets in color_buckets.iteritems():
            num_item = 0
            for left, right in zip(buckets['left'], buckets['right']):
                pairs.append((left, right))
                # Remove the pair socks
                num_item += 1
            buckets['left'] = buckets['left'][num_item:]
            buckets['right'] = buckets['right'][num_item:]

        # If we want to pair the reminder
        for color, buckets in color_buckets.iteritems():
            # merge the unpair socks together
            left_socks = buckets['left']
            right_socks = buckets['right']
            if not left_socks and not right_socks:
                continue

            for c in range(color-color_diff, color+color_diff+1):
                if c in color_buckets:
                    left_socks += color_buckets[c]['left']
                    right_socks += color_buckets[c]['right']
                    # Get rid of all socks
                    color_buckets[c]['left'] = []
                    color_buckets[c]['right'] = []

                for left, right in zip(left_socks, right_socks):
                    pairs.append((left, right))

        return pairs

    def monty_hall(self, num_doors, choosen, switch):
        doors = range(num_doors)

        # Put the prize to a safe place, so we won't reveal it.
        import random
        prize = random.randrange(num_doors)
        doors[prize], doors[-1] = doors[-1], doors[prize]
        print '[Prize is behind door %s.]' % doors[-1]

        # Put your choose to a safe place as well (if you aren't right).
        was_wrong = prize != choosen
        if was_wrong:
            doors[choosen], doors[-2] = doors[-2], doors[choosen]
        else:
            index = random.randrange(num_doors - 1)
            doors[index], doors[-2] = doors[-2], doors[index]

        # Random reveal the rest of doors
        n = num_doors - 3
        while n >= 0:
            index = random.randrange(n + 1)  # Start from 1.
            print 'Reveal door %s.' % doors[index]
            doors[index], doors[n] = doors[n], doors[index]
            n -= 1

        # Just play the game.
        print 'Now, we have door %s and door %s.' % (doors[-2],
                                                     doors[-1])
        print 'Your choice was door %s.' % choosen
        print 'Do you want to change?'
        print 'Yes.' if switch else 'No.'

        if was_wrong and switch:
            print 'You got a car!'
        else:
            print 'You got a goat.'

    def seperate_text(text, dic, start=0, words=[], sentences=[]):
        # No the same as the Work Break on Leetcode
        if start == len(text):
            sentences.append(' '.join(words))
            return

        for end in range(start, len(text) + 1):
            word = text[start:end]
            if word in dic:
                seperate_text(text, dic, end, words + [word], sentences)
        return sentences

    def quick_select(nums, k):
        start, end = 0, len(nums) - 1
        while start < end:
            pivot_index = random.randrange(start, end + 1)

            new_pivot_index = partion(nums, start, end, pivot_index)

            if new_pivot_index < k:
                start = new_pivot_index + 1
            else:
                end = new_pivot_index - 1
        return nums[k - 1]

    def partion(nums, start, end, pivot_index):
        # The pivot will always be placed in its final position.
        pivot = nums[pivot_index]

        # bottom: nums[:smaller]
        # middle: nums[smaller:equal]
        # unknow: nums[equal:larger+1]
        # top: nums[larger+1:]
        smaller, equal, larger = start, start, end
        while equal <= larger:
            if nums[equal] > pivot:  # Change to larger to the problem.
                nums[smaller], nums[equal] = nums[equal], nums[smaller]
                smaller += 1
                equal += 1
            elif nums[equal] == pivot:
                equal += 1
            else:
                nums[equal], nums[larger] = nums[larger], nums[equal]
                larger -= 1
        return smaller

    def partition_palindrome(self, text):
        # Return list of segmented text. Each text is a palidrome.
        # DFS
        ans = []
        for i in range(1, len(text) + 1):
            if text[:i] == text[i-1::-1]:
                for rest in self.partition_palindrome(text[i:]):
                    ans.append([text[:i]] + rest)

        if not ans:
            return [[]]
        return ans

    def min_cut_palindrome(self, text):
        # Find the minimum cut for in all possible partition palidrome
        # DP
        n = len(text)

        # Cut stores the minimum cut we need for k character.
        cut = [0] * (n + 1)  # For corner case, one character
        for i in range(n + 1):
            cut[i] = i - 1

        for i in range(n):
            left, right = i, i
            while 0 <= left and right < n and text[left] == text[right]:
                # left is the index of min cut not including this palindrome.
                cut[right + 1] = min(cut[right + 1], 1 + cut[left])
                left -= 1
                right += 1

            left, right = i, i + 1
            while 0 <= left and right < n and text[left] == text[right]:
                cut[right + 1] = min(cut[right + 1], 1 + cut[left])
                left -= 1
                right += 1

        return cut[n]


class LRUCache():
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return -1

        value = self.cache.pop(key)
        self.cache[key] = value
        return value

    def set(self, key, value):
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) == self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = value


class Tetris():
    def __init__(self):
        self.planes = []
        self.height = 0

    def drop(self, position, length):
        new_left, new_right, = position, position + length
        place = False
        for i, (left, right, height) in enumerate(self.planes):
            # Greedy place the new square from the heightest place.

            # Add the new segment in the original height.
            if left <= new_left <= new_right <= right:
                self.planes.append((left, new_left, height))
                self.planes.append((new_right, right, height))
                place = True
            elif left <= new_left <= right <= new_right:
                self.planes.append((left, new_left, height))
                place = True
            elif new_left <= left <= new_right <= right:
                self.planes.append((new_right, right, height))
                place = True
            elif new_left <= left <= right <= new_right:
                place = True

            # Update the under cover old one.
            if place:
                new_height = height + length
                self.planes[i] = (new_left, new_right, new_height)
                self.height = max(self.height, new_height)
                break

        if not place:
            self.planes.append((new_left, new_right, length))
            self.height = max(self.height, length)

        self.planes.sort(key=lambda x: -x[-1])

    def get_height(self):
        return self.height


class Graph():
    TO_BE_VISITED = 0
    VISITING = 1
    DONE = 2

    def __init__(self, edges):
        self.edges = edges

        self.nodes = set()
        for node in edges.keys():
            self.nodes.add(node)
            for neighbor in self.edges[node]:
                self.nodes.add(neighbor)

    def has_cycle(self):
        status = {node: Graph.TO_BE_VISITED for node in self.nodes}
        for node in self.edges.keys():
            if (status[node] == Graph.TO_BE_VISITED and
                    self.visit(node, status)):
                return True
        return False

    def visit(self, node, status):
        status[node] = Graph.VISITING
        for neighbor in self.edges[node]:
            if status[neighbor] == Graph.VISITING:
                return True
            if self.visit(neighbor, status):
                return True
        status[node] = Graph.DONE
        return False


if __name__ == '__main__':
    # Task order
    # graph = {5: [6],
    #          6: [7],
    #          7: [4],
    #          3: [7, 8],
    #          6: [4]}
    # print Square().order_task(graph)

    # Find string
    # matrix = ['godh',
    #           'efci',
    #           'dagg',
    #           'tggg']
    # dic = ['hi', 'god', 'cat']
    # print Square().search_strings(matrix, dic)

    # Rotate matrix
    # matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # Square().rotate_matrix(matrix)
    # print matrix

    # Sock pairing
    # socks = [(1, 'left', 0),
    #          (2, 'left', 1),
    #          (3, 'right', 2),
    #          (4, 'right', 0)]
    # print Square().pair_socks(socks, 1)

    # Monty hall
    # monty_hall(10, 1, 1)

    # Place the suqare.
    # tetris = Tetris()
    # tetris.drop(0, 1)
    # print tetris.get_height()
    # tetris.drop(2, 2)
    # print tetris.get_height()
    # tetris.drop(1, 3)
    # print tetris.get_height()
    # tetris.drop(0, 2)
    # print tetris.get_height()
    # tetris.drop(1, 2)
    # print tetris.get_height()

    # Seperate text
    # dic = {'there', 'are', 'four', 'word', words'}
    # text = 'therearefourwords'
    # print seperate_text(text, dic)

    # Graph has cycle.
    # edges = {'a': ['b'],
    #          'b': ['c'],
    #          'c': ['a']}
    # graph = Graph(edges)
    # print graph.has_cycle()

    # Quick select
    # nums = [3, 1, 2, 3]
    # k = 2
    # print quick_select(nums, k)

    # Partion palindrome
    # text = 'abac'
    # print Square().partition_palindrome(text)
    # print Square().min_cut_palindrome(text)

