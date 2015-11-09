"""
user1 = {'a', 'b', 'c'}
user2 = {'a', 'b', 'c', 'd'}
user3 = {'d', 'e'}
user4 = {'a', 'c'}

Who is the most similar user with user1?
user2
"""
from collections import defaultdict
import heapq
import unittest


class RecommendationSystem(object):
    def __init__(self):
        self.users = defaultdict(set)
        self.attributes = defaultdict(set)  # From each attribute to users

    def add_user(self, user, attributes):
        if user not in self.users:
            self.users[user] = attributes
            for attribute in attributes:
                self.attributes[attribute].add(user)

    def similar(self, user, top_k=1):
        match = defaultdict(int)
        for attribute in self.users[user]:
            for similar_user in self.attributes[attribute]:
                if user != similar_user:
                    match[(user, similar_user)] += 1

        # Find the top k values' key in a dictionary. O(k log n)
        top_k_match_pairs = heapq.nlargest(top_k, match, key=match.get)
        return [pair[1] for pair in top_k_match_pairs]


class TestSimiliarUser(unittest.TestCase):
    def test_top_similiar(self):
        system = RecommendationSystem()
        system.add_user('user1', {'a', 'b', 'c'})
        system.add_user('user2', {'a', 'b', 'c', 'd'})
        system.add_user('user3', {'d', 'e'})
        system.add_user('user4', {'a', 'c'})

        self.assertEqual(system.similar('user1'), ['user2'])

    def test_top_k_similiar(self):
        system = RecommendationSystem()
        system.add_user('user1', {'a', 'b', 'c'})
        system.add_user('user2', {'a', 'b', 'c', 'd'})
        system.add_user('user3', {'d', 'e'})
        system.add_user('user4', {'a', 'c'})

        self.assertEqual(system.similar('user1', 2), ['user2', 'user4'])


if __name__ == '__main__':
    unittest.main()
