# Return the sum of all possible combination in an array.
def subset(nums):
    # O(n 2^n)
    ans = [[]]
    for n in nums:  # O(n)
        ans += [s + [n] for s in ans]  # O(2^n)
    return ans


def subset_sum(sets):   
    ans = {}  # [BUG] set()
    for s in sets:  # O(n)
        x = sum(s)
        if x not in ans:
            ans.add(x)
    return list(ans)
# nums = [1, 2, 3, 4]
# print subset_sum(subset(nums))

# Flip an False element in a 2D array, the probablity of flipping each
# False element should be the same. (1 / |false_current_found|)
# array = [[False, False], [False, False]]
# flip_array(array)
# print array
def flip_array(array):
    # O(mn)
    import random
    candidates = range(0, len(array) * len(array[0]))
    random.shuffle(candidates)
    while candidates:
        flip(candidates, array)


def flip(candidates, array):
    x = candidates.pop()
    i, j = x // len(array[0]), x % len(array[0])
    print 'flip (%d, %d)' % (i, j)
    array[i][j] = True

#  A  is   B's subtree
#          4
#  1      1 5
# 2 3    2 3
def trace(root_a, root_b):
    # O(mn)
    # O(m + n) if do inorder and use strstr (kmp) for string comparision.
    if not root_a:
        return True
    if not root_b:
        return False
    if compare(root_a, root_b):
        return True
    return (compare(root_a.left, root_b.left) or
            compare(root_a.right, root_b.right))


def compare(root_a, root_b):
    if not root_a and not root_b:
        return True
    if not root_a or not root_b:
        return False

    return (root_a.val == root_b.val and
            compare(root_a.left, root_b.left) and
            compare(root_a.right, root_b.right))

# total_pass(x, nums) is the number of elements in nums that is greater than x
# Q: Find the maximum total_pass(x, nums) for x in nums.
# A: If duplicates are in nums, find how many minimum x in nums. Return n - m.
#    If nums is sorted, binary search for the next nums[0]. O(log(n))
#    If nums isn't sorted, use a dict to count each element, find the min key.
def max_total_pass1(nums):
    # Binary search if sorted. O(log n)
    target = nums[0]
    candidate = 0
    i, j = 0, len(nums) - 1
    while i <= j:
        m = i + (j - i) / 2
        if nums[m] <= target:
            if nums[m] == target:
                candidate = m
            i = m + 1
        else:
            j = m - 1
    return len(nums) - (candidate + 1)


def max_total_pass2(nums):
    # O(n)
    from collections import defaultdict
    hist = defaultdict(int)
    for n in nums:
        hist[n] += 1

    return len(nums) - hist[min(hist.keys())]
# nums = [1, 2, 1, 4, 3, 2]
# print max_total_pass2(nums)


# left_pass(x, nums) is the number of elements in num[:i+1] is greater than x
# Q: Find the maximum left_pass(x, nums)
# 1 2 3 2 1
# 0 0 0 1 3
# A: If the nums are sorted, trivial return 0.
#    No assumption. Brute force O(n^2). Use self-balance-BST O(nlogn)
def max_left_pass(nums):
    # O(n^2)
    left_pass = [0] * len(nums)
    for i, n in enumerate(nums):
        count = 0
        for m in nums[:i]:
            count += (m > n)
        left_pass[i] = count
    return max(left_pass)
# nums = [1, 2, 3, 2, 1]
# print max_left_pass(nums)

# Encode a list of string to a string, and then decode it to the same strings.
def encode(strings):
    # 3#...2#..0#1#.
    encoded_string = ''
    for s in strings:
        encoded_string += str(len(s)) + '#' + s  # Need guard to identify n.
    return encoded_string


def decode(string):
    decoded_strings = []
    i = 0
    while i < len(string):
        n = ''
        while string[i] != '#':
            n += string[i]
            i += 1
        n = int(n)

        i += 1  # Move over the guard.

        s = string[i:i+n]
        i += n
        decoded_strings.append(s)
    return decoded_strings
# strings = ['012', '', '#a81b']
# print encode(strings)
# print decode(encode(strings))

# Print tree
def print_tree(root, path=''):
    if not root:
        return
    if not root.left and not root.right:
        print path + str(root.val)

    if root.left:
        print_tree(root.left, path + str(root.val) + '->')
    if root.right:
        print_tree(root.right, path + str(root.val) + '->')

# Good numbers = a ** 3 + b ** 3
def good_numbers(n):
    cubics = [number ** 3 for number in range(1, n)]  # Shouldn't contain 0.

    good_numbers = []
    for i in range(n):
        if two_sum(cubics, i):
            good_numbers.append(i)
    return good_numbers


def two_sum(nums, target):
    # O(n)
    exist = set()
    for n in nums:
        exist.add(n)

    for n in nums:
        x = target - n
        if x != n and x in exist:  # x != n to remove the number itself.
            return True
    return False
# print good_numbers(10)

# Wiggle sort
# nums[0] <= nums[1] >= nums[2] <= nums[3] >= ...
def wiggle_sort(nums):
    for i, n in enumerate(nums[1:], 1):
        # If something wrong, fix it.
        if ((i % 2 == 1 and nums[i] < nums[i - 1]) or
                (i % 2 == 0 and nums[i] > nums[i - 1])):
            nums[i], nums[i - 1] = nums[i - 1], nums[i]
    return nums
# nums = [1, 2, 4, 3, 6, 5, 8, 7]
# print wiggle_sort(nums)

# Longest continuous numbers
# 1, 2, 3, 4, 6, 7, 8, 9, 10 -> 6, 7, 8, 9, 10
def longest_continous(nums):
    if not nums:
        return nums

    i = 0
    start, end, length = i, i, 0
    for j in range(1, len(nums)):
        if abs(nums[j] - nums[j - 1]) > 1:
            if length < j - i:
                start, end = i, j - 1
                length = end - start
            i = j
    if length < j - i:
        start, end = i, j
    return nums[start:end + 1]
# nums = [1, 2, 3, 1, 2, 3, 4, 6, 4, 3, 2, 1, 0]
# print longest_continous(nums)

# Valid UTF-8
# How many bytes are valid?
# 000 -> length 1, each other byte should start with 0b10
# 100 -> length 2
# 110 -> length 3
def valid_utf8(bytes):
    expected_length = 0
    if (bytes[0] & 0b10000000) == 0b00000000:
        expected_length = 1
    elif (bytes[0] & 0b11100000) == 0b11000000:
        expected_length = 2
    elif (bytes[0] & 0b11110000) == 0b11100000:
        expected_length = 3
    elif (bytes[0] & 0b11111000) == 0b11110000:
        expected_length = 4
    elif (bytes[0] & 0b11111100) == 0b11111000:
        expected_length = 5
    elif (bytes[0] & 0b11111110) == 0b11111100:
        expected_length = 6
    else:
        return False

    if expected_length != len(bytes):
        return False

    for i in range(1, expected_length):
        if (bytes[i] & 0b11000000) != 0b10000000:
            return False
    return True


# a divide b
# 1 / 2 = 0.5
# 1 / 3 = 0.(3)
# div 0
# Negative sign
from collections import defaultdict
def devide(a, b):
    """
    :type numerator (a): int
    :type denominator (b): int
    :rtype: str
    """
    if b == 0:
        return None
    if a * b < 0:
        return '-' + devide(abs(a), abs(b))
    if a % b == 0:
        return str(a // b)

    s = str(a // b) + '.'
    d = ''
    a, existed_r = a % b, defaultdict(int)
    while a:
        if a in existed_r:
            break
        else:
            existed_r[a] = len(d)  # Record index first.
            a *= 10
            d += str(a // b)
            a %= b
    if a:
        return s + d[:existed_r[a]] + '(' + d[existed_r[a]:] + ')'
    return s + d

# Summary missing range in an array.
# [1, 2, 3, 5, 99] -> [4-4, 6-98].
#                     [1-3], [5-5], [99-99]

# What if the missing range only has one number?
# Any negative inputs? No
def summary(nums):
    if not nums:
        return nums

    existed_intervals = []
    start, end = nums[0], nums[0]
    for i, n in enumerate(nums[1:], 1):
        if n - nums[i - 1] == 1:
            end = n
        else:
            existed_intervals.append([start, end])
            start, end = n, n
    existed_intervals.append([start, end])

    if len(existed_intervals) <= 1:
        # No missing.
        return None

    missing_intervals = []
    for i, interval in enumerate(existed_intervals[1:], 1):
        start = existed_intervals[i - 1][1] + 1
        end = interval[0] - 1
        missing_intervals.append([start, end])
    return missing_intervals
# nums = [1, 2, 3, 5, 99]
# print summary(nums)

# Buy and sell stock at most k times.
def max_profit(prices, k=1):
    # balance[j] hold the max balance for at most j transcation.
    # profits[j] hold the max profits for at most j transcation.
    if not prices:
        return 0

    if k >= len(prices) / 2:
        profit = 0
        for i, price in prices:
            balance = price - prices[i - 1]
            if balance:
                profit += balance
        return profit

    balance = [-max(prices)] * (k + 1)
    profits = [0] * (k + 1)
    for i, price in enumerate(prices):
        for j in range(1, k + 1):
            # Because balance[j] gain from the profits[j - 1], the
            # more k, the profits should be larger.

            # Whether to buy a stock at price.
            balance[j] = max(balance[j], profits[j - 1] - price)

            # Whether to sell a stock at price.
            profits[j] = max(profits[j], balance[j] + price)
    return profits[k]

# Buy and sell have to by seperated by 1 day.
def max_profits2(prices):
    balance = [-max(prices)] * len(prices)
    profits = [0] * len(prices)

    # Preconditions
    balance[0] = -prices[0]  # Either buy on the first day.
    balance[1] = -prices[1]  # Or buy on the second day.
    profits[1] = balance[0] + prices[1]

    for i, price in enumerate(prices[2:], 2):
        # Don't buy today, or buy today if you don't buy 2 days ago.
        balance[i] = max(balance[i - 1], profits[i - 2] - price)  # Buy

        # Don't sell today or sell today.
        profits[i] = max(profits[i - 1], balance[i - 1] + price)  # Sell
    return profits[-1]
# prices = [1, 10, 100]
# print max_profits2(prices)

def maximum_rectangle_area_in_histogram(heights):
    max_area = 0
    heights.append(0)

    i = 0
    stack = []
    while i < len(heights):
        if not stack or heights[i] > heights[stack[-1]]:
            stack.append(i)
            i += 1
        else:
            top = stack.pop()
            if not stack:
                # All heights are larger than the current top.
                area = heights[top] * i
            else:
                # width is the region not includes two heads.
                area = heights[top] * (i - stack[-1] - 1)
            max_area = max(max_area, area)
    return max_area
# heights = [2, 3, 5, 6, 2, 3]
# heights = [1]
# print maximum_rectangle_area_in_histogram(heights)

def maximum_rectangle_area(matrix):
    # The 2D area coule be viewed as a histogram in somehow.
    # 1 1 0 1 -> 1 1 0 1
    # 0 1 1 1 -> 0 2 1 2
    # 0 1 1 1 -> 0 3 2 3 -> max_hist 6
    # 0 1 0 0 -> 0 4 0 0
    def reset_heights(heights, row):
        for j, c in enumerate(row):
            if c == '1':
                heights[j] += 1
            else:
                heights[j] = 0
        return heights

    max_area = 0
    heights = [1 if c == '1' else 0 for c in matrix[0]]
    max_area = maximum_rectangle_area_in_histogram(heights)
    for row in matrix[1:]:
        heights = reset_heights(heights, row)
        area = maximum_rectangle_area_in_histogram(heights)
        max_area = max(max_area, area)
    return max_area
matrix = ['00', '00']
# print maximum_rectangle_area(matrix)

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

# Find the k largest elements from N sorted list
def find_k_largest_elements(sorted_nums, k):

    from heapq import heapify
    from heapq import heappush
    from heapq import heappop

    heap = []
    for i, nums in enumerate(sorted_nums):
        heap.append((-nums[-1], i))
    # min-heap heap[0] is the smallest element.
    # max-heap when you add a "-" in the front.
    heapify(heap)
    print heap

    indices_for_sorted_nums = [1] * len(sorted_nums)
    ans = []
    for _ in range(k):
        value, index = heappop(heap)
        ans.append(-value)

        indices_for_sorted_nums[index] += 1
        heappush(heap,
                 (-sorted_nums[index][-indices_for_sorted_nums[index]], index))
    return ans
# sorted_nums = [[100, 200, 300, 400], [5, 6, 7, 8],
#                [1, 2, 3, 4], [9, 10, 11, 12]]
# k = 3
# print find_k_largest_elements(sorted_nums, k)
