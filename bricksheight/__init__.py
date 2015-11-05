"""
Brick's Height


ooxxkkkkkkkssss-- (bird view)

    kkkkkkk
          sssss
  xxx     sssss
  xxx     sssss
oooovvv   sssss
oooovvv   sssss
-----------------> float
012345678901234
"""

from collections import namedtuple

Region = namedtuple('Region', ['start', 'end', 'height'])


class Bricks(object):
    def __init__(self):
        self.bird_view = []  # List of regions

    def drop(self, region):
        if not self.bird_view:
            self.bird_view.append(region)
        else:
            # No need to seperate regions. Just overlap different regions.
            overlap = False
            for start, end, height in self.bird_view:
                if max(start, region.start) <= min(end, region.end):  # !
                    overlap = True
                    self.bird_view.append(
                        Region(region.start, region.end,
                               height + region.height))
                    break

            if not overlap:  # !
                self.bird_view.append(region)

        self.bird_view.sort(key=lambda region: region[2], reverse=True)  # !

    def height(self, point):
        for start, end, height in self.bird_view:
            if start <= point <= end:
                return height
        return 0

if __name__ == '__main__':
    bricks = Bricks()
    bricks.drop(Region(0, 3, 2))
    bricks.drop(Region(4, 6, 2))
    bricks.drop(Region(10, 14, 5))
    bricks.drop(Region(2, 4, 2))
    bricks.drop(Region(4, 10, 1))
    print bricks.bird_view
    for i in range(15):
        print bricks.height(i)
