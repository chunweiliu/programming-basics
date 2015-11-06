"""
Calendar

Implement a Calendar class which supports two operations:

1. Schedule events

2. Show the organized schedule


|aaaaaaaaaa|
     |bbbbbbbbbb|
  |cccccc|
-------------------
|a|ac|abc|a|b   | -> Output
          b

"""

from collections import namedtuple
from datetime import datetime

import unittest

Event = namedtuple('Event', ['name', 'start', 'end'])
TimeWindow = namedtuple('TimeWindow', ['start', 'end', 'events'])


class Calendar(object):
    def __init__(self):
        self.timestamps = []  # list of time
        self.events = []  # list of events
        self._schedule = []  # list of TimeWindow

    def add_event(self, event):
        # Update events
        self.events.append(event)

        # Update timestamps
        for timestamp in [event.start, event.end]:
            if timestamp not in self.timestamps:
                self.timestamps.append(timestamp)
        self.timestamps.sort()

        # Update schedule
        self._schedule = []
        for start, end in zip(self.timestamps[:-1], self.timestamps[1:]):
            events = []
            for event in self.events:
                if max(event.start, start) < min(event.end, end):
                    events.append(event.name)
            self._schedule.append(TimeWindow(start, end, events))

    def schedule(self):
        return self._schedule


class TestCalendar(unittest.TestCase):
    def test_no_overlap(self):
        calendar = Calendar()

        event_name = 'a'
        start_time = datetime.strptime('2014-01-01 10:00', '%Y-%m-%d %H:%M')
        end_time = datetime.strptime('2014-01-01 11:00', '%Y-%m-%d %H:%M')

        calendar.add_event(Event(event_name, start_time, end_time))
        expected_schedule = [TimeWindow(start_time, end_time, [event_name])]

        self.assertEqual(expected_schedule, calendar.schedule())

    def test_overlap(self):
        calendar = Calendar()

        calendar.add_event(
            Event('a',
                  datetime.strptime('2014-01-01 10:00', '%Y-%m-%d %H:%M'),
                  datetime.strptime('2014-01-01 11:00', '%Y-%m-%d %H:%M')))

        calendar.add_event(
            Event('b',
                  datetime.strptime('2014-01-01 10:30', '%Y-%m-%d %H:%M'),
                  datetime.strptime('2014-01-01 11:30', '%Y-%m-%d %H:%M')))

        expected_schedule = [
            TimeWindow(datetime.strptime('2014-01-01 10:00',
                                         '%Y-%m-%d %H:%M'),
                       datetime.strptime('2014-01-01 10:30',
                                         '%Y-%m-%d %H:%M'), ['a']),
            TimeWindow(datetime.strptime('2014-01-01 10:30',
                                         '%Y-%m-%d %H:%M'),
                       datetime.strptime('2014-01-01 11:00',
                                         '%Y-%m-%d %H:%M'), ['a', 'b']),
            TimeWindow(datetime.strptime('2014-01-01 11:00',
                                         '%Y-%m-%d %H:%M'),
                       datetime.strptime('2014-01-01 11:30',
                                         '%Y-%m-%d %H:%M'), ['b'])]

        self.assertEqual(expected_schedule, calendar.schedule())

    def test_overlap2(self):
        calendar = Calendar()

        calendar.add_event(
            Event('a',
                  datetime.strptime('2014-01-01 10:00', '%Y-%m-%d %H:%M'),
                  datetime.strptime('2014-01-01 11:00', '%Y-%m-%d %H:%M')))

        calendar.add_event(
            Event('b',
                  datetime.strptime('2014-01-01 10:30', '%Y-%m-%d %H:%M'),
                  datetime.strptime('2014-01-01 11:30', '%Y-%m-%d %H:%M')))

        calendar.add_event(
            Event('c',
                  datetime.strptime('2014-01-01 10:15', '%Y-%m-%d %H:%M'),
                  datetime.strptime('2014-01-01 10:45', '%Y-%m-%d %H:%M')))

        expected_schedule = [
            TimeWindow(datetime.strptime('2014-01-01 10:00',
                                         '%Y-%m-%d %H:%M'),
                       datetime.strptime('2014-01-01 10:15',
                                         '%Y-%m-%d %H:%M'), ['a']),
            TimeWindow(datetime.strptime('2014-01-01 10:15',
                                         '%Y-%m-%d %H:%M'),
                       datetime.strptime('2014-01-01 10:30',
                                         '%Y-%m-%d %H:%M'), ['a', 'c']),
            TimeWindow(datetime.strptime('2014-01-01 10:30',
                                         '%Y-%m-%d %H:%M'),
                       datetime.strptime('2014-01-01 10:45',
                                         '%Y-%m-%d %H:%M'), ['a', 'b', 'c']),
            TimeWindow(datetime.strptime('2014-01-01 10:45',
                                         '%Y-%m-%d %H:%M'),
                       datetime.strptime('2014-01-01 11:00',
                                         '%Y-%m-%d %H:%M'), ['a', 'b']),
            TimeWindow(datetime.strptime('2014-01-01 11:00',
                                         '%Y-%m-%d %H:%M'),
                       datetime.strptime('2014-01-01 11:30',
                                         '%Y-%m-%d %H:%M'), ['b'])]

        self.assertEqual(expected_schedule, calendar.schedule())


if __name__ == '__main__':
    unittest.main()
