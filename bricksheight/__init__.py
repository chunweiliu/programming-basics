"""
Brick's Height


ovvnnnnbbx-- (bird view)

   nnnn
      bbb
 vvvv bbb
oooo   xxx
------------> float
0123456789
"""

from collections import namedtuple
import unittest

Brick = namedtuple('Brick', ['start', 'end', 'height'])


class Bricks(object):
    def __init__(self):
        self.bird_view = []  # List of bricks

    def drop(self, brick):
        if not self.bird_view:
            self.bird_view.append(brick)
        else:
            # No need to seperate regions. Just overlap different regions.
            overlap = False
            for start, end, height in self.bird_view:
                if max(start, brick.start) <= min(end, brick.end):  # !
                    overlap = True
                    self.bird_view.append(
                        Brick(brick.start, brick.end,
                              height + brick.height))
                    break

            if not overlap:  # !
                self.bird_view.append(brick)

        self.bird_view.sort(key=lambda brick: brick[2], reverse=True)  # !

    def height(self, point):
        for start, end, height in self.bird_view:
            if start <= point <= end:
                return height
        return 0


class TestBricks(unittest.TestCase):
    def test_initial(self):
        bricks = Bricks()
        for i in range(10):
            self.assertEqual(bricks.height(i), 0)

    def test_one_brick(self):
        bird_view = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        bricks = Bricks()
        bricks.drop(Brick(0, 3, 1))
        for i in range(10):
            self.assertEqual(bricks.height(i), bird_view[i])

    def test_two_bricks(self):
        bird_view = [1, 1, 1, 1, 0, 0, 0, 1, 1, 1]
        bricks = Bricks()
        bricks.drop(Brick(0, 3, 1))
        bricks.drop(Brick(7, 9, 1))
        for i in range(10):
            self.assertEqual(bricks.height(i), bird_view[i])

    def test_bricks(self):
        bird_view = [1, 2, 2, 4, 4, 4, 4, 3, 3, 1]
        bricks = Bricks()
        bricks.drop(Brick(0, 3, 1))
        bricks.drop(Brick(7, 9, 1))
        bricks.drop(Brick(1, 4, 1))
        bricks.drop(Brick(6, 8, 2))
        bricks.drop(Brick(3, 6, 1))
        for i in range(10):
            self.assertEqual(bricks.height(i), bird_view[i])

if __name__ == '__main__':
    unittest.main()
