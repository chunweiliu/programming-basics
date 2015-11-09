"""
Shorten URL

From 'http://myverycoolproduct.myverycoolcompany.com'
To   'sho.rt/zV15utK'

"""
import string
import unittest


class ShortenURL(object):
    def __init__(self):
        self.long_to_short = {}
        self.short_to_long = {}

        self.domain_name = 'sho.rt/'

    def register_url(self, long_url):
        if long_url in self.long_to_short:
            return

        short_url = self._shorten(long_url)
        self.long_to_short[long_url] = short_url
        self.short_to_long[short_url] = long_url
        return short_url

    def look_up(self, short_url):
        return self.short_to_long[short_url]

    def _shorten(self, long_url):
        code = sum(hash(word) for word in long_url)
        # code is a 12 digit numbers

        # Encode the 12 digit code to a 7 digit alphanumerics
        code_words = string.uppercase + string.lowercase + string.digits
        encoded_code = ''
        while code:
            encoded_code += code_words[code % len(code_words)]
            code /= len(code_words)

        return self.domain_name + encoded_code


class TestShortenURL(unittest.TestCase):
    def test_long_to_short(self):
        service = ShortenURL()
        long_url = 'http://myverycoolproduct.myverycoolcompany.com'
        short_url = service.register_url(long_url)
        print short_url
        self.assertEqual(service.look_up(short_url), long_url)

if __name__ == '__main__':
    unittest.main()
