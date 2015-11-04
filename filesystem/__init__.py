"""
In Memory File System

Handle basic commands such as: ls, mkdir, add_file
"""
from collections import defaultdict


class Folder(object):
    def __init__(self):
        self.files = []

        # Trick: Using a defaultdict with self type.
        # Same trick is used for implementing Trie.
        self.folders = defaultdict(Folder)

    def __repr__(self):
        return '\n'.join(['.', '..']+self.files+self.folders.keys())


class FileSystem(object):
    def __init__(self):
        self.root = Folder()

    def ls(self, path):
        # tokens[0] is an empty string
        # tokens[1] is the first level folder
        # tokens[-1] is the file
        tokens = path.split('/')

        curr = self.root
        for i, next_folder in enumerate(tokens[1:], 1):
            if next_folder not in curr.folders:
                break
            curr = curr.folders[next_folder]

        if i == len(tokens) - 1:
            return curr
        else:
            return ''

    def mkdir_p(self, path):
        tokens = path.split('/')

        curr = self.root
        for i, next_folder in enumerate(tokens[1:], 1):
            curr = curr.folders[next_folder]

    def add_file(self, path, file):
        tokens = path.split('/')

        curr = self.root
        for i, next_folder in enumerate(tokens[1:], 1):
            curr = curr.folders[next_folder]
        curr.files.append(file)


if __name__ == '__main__':
    fs = FileSystem()

    # .
    # ..
    print fs.ls('/')

    fs.mkdir_p('/folder1/folder2/')
    fs.mkdir_p('/folder1/another_folder2/')

    # .
    # ..
    # folder1
    print fs.ls('/')

    # .
    # ..
    # folder2
    # another_folder2
    print fs.ls('/folder1')

    fs.add_file("/folder1", "file1.txt")

    # .
    # ..
    # file1.txt
    # folder2
    # another_folder2
    print fs.ls("/folder1")
