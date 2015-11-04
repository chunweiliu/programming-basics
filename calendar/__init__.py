#!/usr/bin/env python

from datetime import datetime
from collections import namedtuple

import sys

Event = namedtuple('Event', ['id', 'start_time', 'end_time'])
ConflictedTimeWindow = \
    namedtuple('ConflictedTimeWindow',
               ['start_time', 'end_time', 'conflicted_event_ids'])


class Calendar(object):
    # Should allow multiple events to be scheduled over the same time window.
    def __init__(self):
        self.timestamp = []
        self.events = []

    def schedule(self, event):
        # How to devide intervals?
        for timestamp in [event.start_time, event.end_time]:
            if timestamp not in self.timestamp:
                self.timestamp.append(timestamp)

        self.events.append(event)

    def find_conflicted_time_windows(self):
        # Divide all timestamps. For every regions, check every events. O(n^2)
        self.timestamp.sort()

        conflicted_events = []
        for start_time, end_time in zip(self.timestamp[:-1],
                                        self.timestamp[1:]):
            conflicted_ids = []
            for event in self.events:
                # Region overlaping
                if (max(start_time, event.start_time) <
                        min(end_time, event.end_time)):
                    conflicted_ids.append(event.id)
            if len(conflicted_ids) > 1:
                conflicted_events.append(
                    ConflictedTimeWindow(start_time, end_time, conflicted_ids))
        return conflicted_events


def main(argv):
    calendar = Calendar()
    calendar.schedule(Event(1,
        datetime.strptime('2014-01-01 10:00', '%Y-%m-%d %H:%M'),
        datetime.strptime('2014-01-01 11:00', '%Y-%m-%d %H:%M')))
    calendar.schedule(Event(2,
        datetime.strptime('2014-01-01 11:00', '%Y-%m-%d %H:%M'),
        datetime.strptime('2014-01-01 12:00', '%Y-%m-%d %H:%M')))
    calendar.schedule(Event(3,
        datetime.strptime('2014-01-01 12:00', '%Y-%m-%d %H:%M'),
        datetime.strptime('2014-01-01 13:00', '%Y-%m-%d %H:%M')))

    calendar.schedule(Event(4,
        datetime.strptime('2014-01-02 10:00', '%Y-%m-%d %H:%M'),
        datetime.strptime('2014-01-02 11:00', '%Y-%m-%d %H:%M')))
    calendar.schedule(Event(5,
        datetime.strptime('2014-01-02 09:30', '%Y-%m-%d %H:%M'),
        datetime.strptime('2014-01-02 11:30', '%Y-%m-%d %H:%M')))
    calendar.schedule(Event(6,
        datetime.strptime('2014-01-02 08:30', '%Y-%m-%d %H:%M'),
        datetime.strptime('2014-01-02 12:30', '%Y-%m-%d %H:%M')))

    calendar.schedule(Event(7,
        datetime.strptime('2014-01-03 10:00', '%Y-%m-%d %H:%M'),
        datetime.strptime('2014-01-03 11:00', '%Y-%m-%d %H:%M')))
    calendar.schedule(Event(8,
        datetime.strptime('2014-01-03 09:30', '%Y-%m-%d %H:%M'),
        datetime.strptime('2014-01-03 10:30', '%Y-%m-%d %H:%M')))
    calendar.schedule(Event(9,
        datetime.strptime('2014-01-03 09:45', '%Y-%m-%d %H:%M'),
        datetime.strptime('2014-01-03 10:45', '%Y-%m-%d %H:%M')))

    print calendar.find_conflicted_time_windows()
    # Should print something like the following

    # [ConflictedTimeWindow(start_time=datetime.datetime(2014, 1, 2, 9, 30),
    #                       end_time=datetime.datetime(2014, 1, 2, 10, 0),
    #                       conflicted_event_ids=[5, 6]),
    #  ConflictedTimeWindow(start_time=datetime.datetime(2014, 1, 2, 10, 0),
    #                       end_time=datetime.datetime(2014, 1, 2, 11, 0),
    #                       conflicted_event_ids=[4, 5, 6]),
    #  ConflictedTimeWindow(start_time=datetime.datetime(2014, 1, 2, 11, 0),
    #                       end_time=datetime.datetime(2014, 1, 2, 11, 30),
    #                       conflicted_event_ids=[5, 6]),
    #  ConflictedTimeWindow(start_time=datetime.datetime(2014, 1, 3, 9, 45),
    #                       end_time=datetime.datetime(2014, 1, 3, 10, 0),
    #                       conflicted_event_ids=[8, 9]),
    #  ConflictedTimeWindow(start_time=datetime.datetime(2014, 1, 3, 10, 0),
    #                       end_time=datetime.datetime(2014, 1, 3, 10, 30),
    #                       conflicted_event_ids=[7, 8, 9]),
    #  ConflictedTimeWindow(start_time=datetime.datetime(2014, 1, 3, 10, 30),
    #                       end_time=datetime.datetime(2014, 1, 3, 10, 45),
    #                       conflicted_event_ids=[7, 9])]

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
