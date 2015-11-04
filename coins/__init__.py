"""
Coin change problems

With limited coins:
1. How many way for making the change?
2. What's the minimum number of coins for making the change?
3. What's the combination of making the change?

Special cases:
1) For unlimited coins, the problems above can be solved with bottom up DP.
2) For canonical coin systems, the problems above can be solved with greedy.
"""

from collections import namedtuple
from collections import defaultdict

CoinSet = namedtuple('CoinSet', ['value', 'number'])


class ChangeMachine():
    def __init__(self, coin_supplies):
        self.coin_supplies = defaultdict(int)
        for value, number in coin_supplies:
            self.coin_supplies[value] = number

    def change(self, money):
        ways = self._change(money, None, defaultdict(int), [])
        if ways:
            min_coin_way, min_coins = ways[0], sum(ways[0].values())
            for way in ways:
                x = sum(way.values())
                if x < min_coins:
                    min_coins = x
                    min_coin_way = way

            # Update current supplies and format output
            coin_set = []
            for value, number in min_coin_way.iteritems():
                self.coin_supplies[value] -= number
                coin_set.append(CoinSet(value, number))

            coin_set.sort(reverse=True)
            return coin_set
        return None

    def _change(self, money, coin_supplies=None,
                attemp=defaultdict(int), ways=[]):
        if coin_supplies is None:
            coin_supplies = self.coin_supplies

        if money == 0:
            ways.append(attemp)
            return 1

        if money < 0 or money > 0 and sum(coin_supplies.values()) <= 0:
            return 0

        # 1) Not choose the coin.
        next_coin_supplies = coin_supplies.copy()  # Use a copy in recursion.
        value, number = next_coin_supplies.popitem()
        self._change(money, next_coin_supplies, attemp, ways)

        # 2) Choose the coin. 1) and 2) are mutual exclusive. So we won't have
        # two same search paths in the recursive tree.
        if number > 0:
            # Using mutable data structure in the recursion function. Since, we
            # will need copies for recusive search. Mutable data structure is
            # usually easier to update.
            next_coin_supplies = coin_supplies.copy()
            next_coin_supplies[value] -= 1

            next_attemp = attemp.copy()
            next_attemp[value] += 1
            self._change(money - value, next_coin_supplies,
                         next_attemp, ways)
        return ways

    def way_to_change(self, money):
        return len(self._change(money))

    def minimum_coins_to_change(self, money):
        ways = self._change(money)
        return min(sum(way.values()) for way in ways)

change_machine = ChangeMachine([CoinSet(1, 10),
                                CoinSet(5, 2),
                                CoinSet(10, 1)])
print change_machine.way_to_change(10)
print change_machine.minimum_coins_to_change(10)
print change_machine.change(30)
print change_machine.change(10)
