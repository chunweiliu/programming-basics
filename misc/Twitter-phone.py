from collections import defaultdict


class Twitter():
    def __init__(self):
        self.board = [[1, 2, 3], [4, 5, 6], [7, 0, 8]]
        self.zero = (2, 2)
        self.direction = {'u': (1, 0), 'd': (-1, 0), 'l': (0, 1), 'r': (0, -1),
                          's': (0, 0)}

    def move(self, d):
        dy, dx = self.direction[d]
        y, x = self.zero
        if 0 <= y + dy <= 2 and 0 <= x + dx <= 2:
            self.board[y][x], self.board[y + dy][x + dx] = \
                self.board[y + dy][x + dx], self.board[y][x]
        return ''.join([str(self.board[i][j])
                        for i in range(3) for j in range(3)])

    def solve(self):
        queue = [self.move('s')]
        # attemp = set(queue[0])
        step = 0
        while queue:
            status = queue.pop(0)
            if status == '123456780':
                return step
            step += 1
            attemp = set()
            for m in ['u', 'd', 'l', 'r']:
                new_status = self.move(m)
                if new_status not in attemp:
                    attemp.add(new_status)
                else:
                    queue.append(new_status)
            queue = list(attemp)
        return -1

        # BFS


    def delete_substring(self, text, pattern):
        """
        What's the maximum time the pattern can be deleted from the text.
        Return int
        """
        # Naive method O(n ** 2)
        i = 0
        while pattern in text:
            text = text.replace(pattern, '', 1)  # Only replace one each time.
            i += 1
        return i

        # Find all positions of the pattern
        # import re
        # positions = [m.start() for m in re.finditer(pattern, text)]
        # Try all order to see which one lead to the maximum time deletion.

    def find_the_first_repeating_char(self, text):
        from collections import OrderedDict
        exist = OrderedDict()
        for char in text:
            if char in exist:
                exist[char] += 1
            else:
                exist[char] = 1

        for key, value in exist.iteritems():
            if value > 1:
                return key
        return None

    #     """
    #     if not text:
    #         return text
    #     exist = {}
    #     first_i, first_c = len(text), text[-1]
    #     for i, char in enumerate(text):
    #         if char in exist:
    #             if exist[char] < first_i:
    #                 first_i = exist[char]
    #                 first_c = char
    #         else:
    #             exist[char] = i
    #     return first_c

    def minimum_range_query(self, nums, indices):
        """
        Find the minimum number in a range of query
        """
        ans = []
        for index in indices:
            ans.append(min(nums[index[0]:index[1]+1]))
        return ans

    def find_the_first_non_repeating_char(self, text):
        """
        Given a string, find the first non-repeating character in it.
        For example, if the input string is 'GeeksforGeeks', then output should
        be 'f' and if input string is 'GeeksQuiz', then output should be 'G'.
        """
        from collections import OrderedDict

        repeat = OrderedDict()
        for char in text:
            if char in repeat:
                repeat[char] += 1
            else:
                repeat[char] = 1

        for key, value in repeat.iteritems():
            if value == 1:
                return key
        return None

    def alien_order(self, words):
        if not words:
            return words
        if len(words) == 1:
            return words[0]

        # Build graph and topological sort
        graph = defaultdict(list)
        # The relationship only in vertical position
        # wr
        # er -> w first than e. (r has no relationship with w or e.)
        for v, w in zip(words[:-1], words[1:]):  # For each pair
            for vi, wi in zip(v, w):
                if vi != wi:
                    graph[vi].append(wi)
                    break  # The first different gave the order.

        for word in words:
            for w in word:
                if w not in graph:
                    graph[w] = []
        return self.topological_sort(graph)

    def topological_sort(self, graph):
        # O(|n| + |e|)
        # Count indegree.
        num_vertex = len(graph.keys())

        in_degree = defaultdict(int)
        for v in graph.keys():
            for n in graph[v]:
                in_degree[n] += 1

        # Make a list of zero in-degree.
        zero_in_degree_vertexs = []
        for v in graph.keys():
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

# Deep Iterator
class ListIterator():
    def __init__(self, nums):
        self.queue = [nums]
        self.flatten()

    def has_next(self):
        return True if self.queue else False

    def next(self):
        curr = self.queue.pop(0)
        self.flatten()
        return curr

    def flatten(self):
        while self.queue and isinstance(self.queue[0], list):
            curr = self.queue.pop(0)
            self.queue = curr + self.queue
# nums = [1, 2, [3, 4, [5, 6]], [[], 7], [[8], 9]]
# iterator = ListIterator(nums)
# while iterator.has_next():
#     print iterator.next()

class Game():
    def __init__(self, board):
        self.board = board
        self.attemp = {board}
        self.direction = {'u': (1, 0), 'd': (-1, 0),
                          'l': (0, 1), 'r': (0, -1),
                          's': (0, 0)}

    def move(self, board, d):
        dy, dx = self.direction[d]
        y, x = self.find_zero(board)

        board = list(board)
        if 0 <= y + dy <= 2 and 0 <= x + dx <= 2:
            board[y * 3 + x], board[(y + dy) * 3 + x + dx] = \
                board[(y + dy) * 3 + x + dx], board[y * 3 + x]
        return ''.join(board)

    def find_zero(self, board):
        index = board.find('0')
        return (index // 3, index % 3)

    def solve(self):
        queue = [self.move(self.board, 's')]

        step = 0
        while queue:
            new_queue = []
            for status in queue:
                if status == '123456780':
                    return step

                for m in ['u', 'd', 'l', 'r']:
                    new_status = self.move(status, m)
                    if new_status not in self.attemp:
                        self.attemp.add(new_status)
                        new_queue.append(new_status)
            queue = new_queue
            step += 1
        return -1
# board = '123405786'
# twitter = Game(board)
# print twitter.solve()

if __name__ == '__main__':
    # text = 'aabbab'
    # pattern = 'ab'
    # print Twitter().delete_substring(text, pattern)

    # text = 'abcba'
    # print Twitter().find_the_first_repeating_char(text)

    # nums = [10, 20, 33, 44, 55, 11]
    # indices = [(0, 2), [1, 5]]
    # print Twitter().minimum_range_query(nums, indices)

    # text = 'GeeksQuiz'
    # print Twitter().find_the_first_non_repeating_char(text)
    # print Twitter().find_the_first_repeating_char(text)

    words = ["wrt", "wrf", "er", "ett", "rftt"]
    print Twitter().alien_order(words)

    # print Twitter().move('s')
    # print Twitter().solve()
