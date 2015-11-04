#!/usr/bin/env python
from collections import Counter
import random
import sys


class Card(object):
    def __init__(self, suite, number):
        self.suite = suite
        self.number = number

    def __repr__(self):
        return "[%s %s]" % (self.suite, self.number)


class Game(object):
    def __init__(self, players, dealer_index, deck):
        self.players = players
        self.dealer_index = dealer_index
        self.deck = deck

    def play(self):
        all_hole_cards, flop, turn, river = self.deal_cards()
        for player_index in xrange(len(self.players)):
            print "%s: %s" % (self.players[player_index],
                              all_hole_cards[player_index])
        print " ".join(str(x) for x in (flop + [turn] + [river]))
        winner_index = self.determine_winner(all_hole_cards, flop, turn, river)
        print "winner is %s\n" % (self.players[winner_index])
        return self.players[winner_index]

    def deal_cards(self):
        num_players = len(self.players)
        all_hole_cards = [[] for i in xrange(num_players)]
        for i in xrange(num_players * 2):
            all_hole_cards[(self.dealer_index + i) % num_players].append(
                self.deck.pop())
        # burn
        self.deck.pop()
        self.deck.pop()
        flop = [self.deck.pop() for x in xrange(3)]
        # burn
        self.deck.pop()
        turn = self.deck.pop()
        # burn
        self.deck.pop()
        river = self.deck.pop()
        return [all_hole_cards, flop, turn, river]

    def determine_winner(self, all_hole_cards, flop, turn, river):
        # the winning player's index.
        # in the case of a draw, return the smallest index amongst winners'
        # indices.

        # IMPLEMENT ME
        return (self.check_cards(all_hole_cards, flop, turn, river,
                                 self.is_straight_flush, 5) or
                self.check_cards(all_hole_cards, flop, turn, river,
                                 self.is_four_of_a_kind, 5) or
                self.check_cards(all_hole_cards, flop, turn, river,
                                 self.is_full_house, 5) or
                self.check_cards(all_hole_cards, flop, turn, river,
                                 self.is_flush, 5) or
                self.check_cards(all_hole_cards, flop, turn, river,
                                 self.is_straight, 5) or
                self.check_cards(all_hole_cards, flop, turn, river,
                                 self.is_three_of_a_kind, 5) or
                self.check_cards(all_hole_cards, flop, turn, river,
                                 self.is_two_pairs, 5) or
                self.check_cards(all_hole_cards, flop, turn, river,
                                 self.is_one_pair, 5) or
                self.check_cards(all_hole_cards, flop, turn, river,
                                 self.is_no_pair, 5) or
                0)

    def check_cards(self, all_hole_cards, flop, turn, river, method, k):
        competiters = []
        for player_index in xrange(len(self.players)):
            cards = all_hole_cards[player_index] + flop + [turn] + [river]
            cards.sort(key=lambda card: card.number, reverse=True)
            for card_index in range(7 - k + 1):
                k_cards = cards[card_index:card_index + k]
                if method(k_cards):
                    competiters.append((player_index, k_cards))
                    break
        if competiters:
            competiters.sort(key=self.compare)
            print '%s: %s' % (method.__name__, competiters[0][1])
            return (-len(self.players) if competiters[0][0] == 0 else
                    competiters[0][0])
        return False

    def is_straight_flush(self, cards):
        return self.is_straight(cards) and self.is_flush(cards)

    def is_four_of_a_kind(self, cards):
        counter = Counter([card.number for card in cards])

        if any(count >= 4 for count in counter.values()):
            # Move the kind to the front.
            cards[0], cards[2] = cards[2], cards[0]
            return True
        return False

    def is_full_house(self, cards):
        counter = Counter([card.number for card in cards])
        if len(counter) == 2 and any(count == 3 for count in counter.values()):
            if cards[0].number == cards[2].number:
                cards[:3].sort(key=lambda card: card.suite)
            else:
                cards[2:].sort(key=lambda card: card.suite)
                cards[0], cards[2] = cards[2], cards[0]
            return True
        return False

    def is_straight(self, cards):
        card_index = 1
        while (card_index < len(cards) and
               cards[card_index - 1].number == cards[card_index].number + 1):
            card_index += 1
        return card_index == 5

    def is_flush(self, cards):
        suite = set()
        for card in cards:
            suite.add(card.suite)
        return len(suite) == 1

    def is_three_of_a_kind(self, cards):
        counter = Counter([card.number for card in cards])
        if any(count >= 3 for count in counter.values()):
            if cards[0].number == cards[2].number:
                cards[:3].sort(key=lambda card: card.suite)
                # cards[0] = cards[:3].sort(key=card.suite)
            elif cards[1].number == cards[4].number:
                cards[1:4].sort(key=lambda card: card.suite)
                cards[0], cards[1] = cards[1], cards[0]
            else:
                cards[2:].sort(key=lambda card: card.suite)
                cards[0], cards[2] = cards[2], cards[0]
            return True
        return False

    def is_two_pairs(self, cards):
        counter = Counter([card.number for card in cards])
        if len(counter) == 3:  # Since it is not a three of a kind
            cards[0], cards[1] = cards[1], cards[0]
            cards[1], cards[3] = cards[3], cards[1]
            cards[:2].sort(key=lambda card: card.number)
            return True
        return False

    def is_one_pair(self, cards):
        counter = Counter([card.number for card in cards])
        if len(counter) == 4:
            for i, card in enumerate(cards[1:], 1):
                if card.number == cards[i - 1].number:
                    temp1, temp2 = cards[0], cards[1]
                    cards[0], cards[1] = cards[i - 1], cards[i]
                    cards[i - 1], cards[i] = temp1, temp2
            return True
        return False

    def is_no_pair(self, cards):
        counter = Counter([card.number for card in cards])
        return len(counter) == 5

    def compare(self, compeitor):
        return (compeitor[1][0].number, compeitor[1][0].suite)


class Player(object):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return "%s" % (self.name)


def main(argv):
    num_players = 4
    num_games = 5
    players = [Player("player-%d" % i) for i in xrange(num_players)]
    dealer_index = 0

    for game_index in xrange(num_games):
        deck = [Card(suite, number)
                for suite in ["spade", "heart", "diamond", "club"]
                for number in xrange(1, 14)]
        random.shuffle(deck)
        game = Game(players, dealer_index, deck)
        game.play()
        dealer_index += 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
