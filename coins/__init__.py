"""
Minimum coin change with limited ammount of coins

With limited coins:
1. How many way for making the change?
2. What's the minimum number of coins for making the change?
3. What's the combination of making the change?

Special cases:
1) For unlimited coins, the problems above can be solved with bottom up DP.
2) For canonical coin systems, the problems above can be solved with greedy.
"""
from collections import defaultdict
import unittest


class ChangeMachine(object):
    def __init__(self, coin_supplies):
        self.coin_supplies = coin_supplies

    def change(self, money):
        if not self.coin_supplies:
            return None

        ways = []
        self._change(money, self.coin_supplies, ways=ways)
        if not ways or not ways[0]:  # Empty change
            return None

        ways.sort(key=lambda way: sum(way.values()))

        # Update coin supplies
        for coin_value, coin_number in ways[0].iteritems():
            self.coin_supplies[coin_value] -= coin_number
        return ways[0]

    def _change(self, money, coin_supplies, attemp=defaultdict(int), ways=[]):
        if money == 0:
            ways.append(attemp)
            return 1

        if money < 0 or money > 0 and not any(coin_supplies.values()):
            return 0

        new_coin_supplies = coin_supplies.copy()
        coin_value, coin_number = new_coin_supplies.popitem()  # Random one
        self._change(money, new_coin_supplies, attemp, ways)

        if coin_number > 0:
            new_coin_supplies = coin_supplies.copy()
            new_coin_supplies[coin_value] -= 1

            new_attemp = attemp.copy()
            new_attemp[coin_value] += 1
            self._change(money - coin_value, new_coin_supplies,
                         new_attemp, ways)


class TestChangeMachine(unittest.TestCase):
    def test_no_supplies(self):
        coin_supplies = {}
        change_machine = ChangeMachine(coin_supplies)

        best_changes = change_machine.change(10)
        expected_changes = None
        self.assertEqual(expected_changes, best_changes)

    def test_one_coin(self):
        coin_supplies = {10: 1}
        change_machine = ChangeMachine(coin_supplies)

        best_changes = change_machine.change(10)
        expected_changes = {10: 1}
        self.assertEqual(expected_changes, best_changes)

    def test_canonical(self):
        coin_supplies = {10: 2, 5: 4, 1: 20}
        change_machine = ChangeMachine(coin_supplies)

        best_changes = change_machine.change(20)
        expected_changes = {10: 2}
        self.assertEqual(expected_changes, best_changes)

    def test_uncanonical(self):
        coin_supplies = {4: 2, 3: 2, 1: 2}
        change_machine = ChangeMachine(coin_supplies)

        best_changes = change_machine.change(6)
        expected_changes = {3: 2}
        self.assertEqual(expected_changes, best_changes)

    def test_too_many_money(self):
        coin_supplies = {10: 2, 5: 4, 1: 20}
        change_machine = ChangeMachine(coin_supplies)

        best_changes = change_machine.change(100)
        expected_changes = None
        self.assertEqual(expected_changes, best_changes)

    def test_unable_change(self):
        coin_supplies = {10: 2, 5: 4, 1: 1}
        change_machine = ChangeMachine(coin_supplies)

        best_changes = change_machine.change(22)
        expected_changes = None
        self.assertEqual(expected_changes, best_changes)

    def test_changes(self):
        coin_supplies = {10: 2, 5: 4, 1: 2}
        change_machine = ChangeMachine(coin_supplies)

        best_changes = change_machine.change(41)
        expected_changes = {10: 2, 5: 4, 1: 1}
        self.assertEqual(expected_changes, best_changes)

        best_changes = change_machine.change(1)
        expected_changes = {1: 1}
        self.assertEqual(expected_changes, best_changes)

        best_changes = change_machine.change(1)
        expected_changes = None
        self.assertEqual(expected_changes, best_changes)

    def test_zero_change(self):
        coin_supplies = {10: 2, 5: 4, 1: 2}
        change_machine = ChangeMachine(coin_supplies)

        best_changes = change_machine.change(0)
        expected_changes = None
        self.assertEqual(expected_changes, best_changes)


if __name__ == '__main__':
    unittest.main()
