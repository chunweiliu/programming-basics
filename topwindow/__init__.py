#!/usr/bin/env python

from collections import namedtuple
import sys

ZWindow = namedtuple('ZWindow', ['window', 'x', 'y', 'z'])


class ScreenCanvas(object):
    def __init__(self, width, height):
        self.canvas_width = width
        self.canvas_height = height
        self.layers = [[0] * width for _ in range(height)]
        self.windows = []

    def add_to_top(self, window, x, y):
        # Adds supplied window at the front of the screen.
        heightest_layer = 0
        for h in range(max(0, y), min(y+window.height, self.canvas_height)):
            for w in range(max(0, x), min(x+window.width, self.canvas_width)):
                self.layers[h][w] += 1
                heightest_layer = max(heightest_layer, self.layers[h][w])
        self.windows.append(ZWindow(window, x, y, heightest_layer))
        # Sorted windows by their z, so we always consider the top window.
        self.windows.sort(key=lambda z_window: z_window.z, reverse=True)

    def find_window_at_position(self, x, y):
        # Returns None if no window detected at position,
        # Returns a Window object otherwise.
        for window, window_x, window_y, window_z in self.windows:
            if (window_x <= x <= window_x + window.width and
                    window_y <= y <= window_y + window.height):
                return window
        return Window(None, None, None)

    def bring_window_at_position_to_front(self, x, y):
        # Brings window at specified position, if any, to the
        # front of the screen.
        for i, (window, window_x, window_y, _) in enumerate(self.windows):
            if (window_x <= x <= window_x + window.width and
                    window_y <= y <= window_y + window.height):
                self.windows[i] = ZWindow(window, window_x, window_y,
                                          self.windows[0].z + 1)
                break
        self.windows.sort(key=lambda z_window: z_window.z, reverse=True)

    # Closes named window. no-op otherwise.
    def close_window(self, window_name):
        for i, (window, _, _, _) in enumerate(self.windows):
            if window.name == window_name:
                del self.windows[i]


class Window(object):
    def __init__(self, name, width, height):
        self.name = name
        self.width = width
        self.height = height

    def __repr__(self):
        return "Window[%s]" % (self.name)


def main(argv):
    canvas = ScreenCanvas(800, 600)
    canvas.add_to_top(Window('A', 20, 15), 12, 11)
    canvas.add_to_top(Window('B', 40, 40), 20, 20)
    canvas.add_to_top(Window('C', 30, 30), 10, 10)
    print canvas.find_window_at_position(22, 22).name  # should return ’C’
    canvas.bring_window_at_position_to_front(50, 50)
    print canvas.find_window_at_position(22, 22).name  # should return ’B’
    canvas.bring_window_at_position_to_front(11, 11)
    print canvas.find_window_at_position(22, 22).name  # should return ’C’
    canvas.close_window('B')
    canvas.close_window('C')
    print canvas.find_window_at_position(22, 22).name  # should return ’A’
    print canvas.find_window_at_position(1, 1).name  # should return None

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
