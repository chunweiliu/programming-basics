#!/usr/bin/env python


class Table(object):
    def __init__(self, name, column_names, data):
        self.name = name
        self.column_names = column_names
        self.data = data  # List of List

    def select(self, projected_column_names):
        # Columns
        column_indices = [j for j in range(len(self.column_names))
                          if self.column_names[j] in projected_column_names]
        rows = [[row[column_index] for column_index in column_indices]
                for row in self.data]
        return Table("Select", projected_column_names, rows)

    def where(self, column_name, value):
        # Rows
        column_index = self.column_names.index(column_name)
        rows = [row for row in self.data if row[column_index] == value]
        return Table("Where", self.column_names, rows)

    def __repr__(self):
        return ", ".join(self.column_names) + "\n" + \
            "\n".join([", ".join([str(val) for val in row])
                       for row in self.data])


class DB(object):
    def __init__(self):
        self.table_map = {}

    def add_table(self, table):
        self.table_map[table.name] = table

    def table(self, table_name):
        return self.table_map[table_name]

    def inner_join(self, left_table, left_table_key_name,
                   right_table, right_table_key_name):
        left_index = left_table.column_names.index(left_table_key_name)
        right_index = right_table.column_names.index(right_table_key_name)
        rows = []
        for left_row in left_table.data:
            for right_row in right_table.data:
                if left_row[left_index] == right_row[right_index]:
                    rows.append(left_row + right_row)

        return Table("InnerJoin",
                     self.renaming(left_table) + self.renaming(right_table),
                     rows)

    def left_join(self, left_table, left_table_key_name,
                  right_table, right_table_key_name):
        left_index = left_table.column_names.index(left_table_key_name)
        right_index = right_table.column_names.index(right_table_key_name)
        rows = []
        for left_row in left_table.data:
            found = False
            for right_row in right_table.data:
                if left_row[left_index] == right_row[right_index]:
                    rows.append(left_row + right_row)
                    found = True

            if not found:
                rows.append(left_row + [None] * len(right_row))

        return Table("LeftJoin",
                     self.renaming(left_table) + self.renaming(right_table),
                     rows)

    def right_join(self, left_table, left_table_key_name,
                   right_table, right_table_key_name):
        # Do a left join.
        rows = self.left_join(right_table, right_table_key_name,
                              left_table, left_table_key_name).data

        # And swap the left to the right.
        n = len(left_table.data[0])
        for row in rows:
            row[:n], row[n:] = row[n:], row[:n]

        # Be care for the name order.
        return Table("RightJoin",
                     self.renaming(right_table) + self.renaming(left_table),
                     rows)

    def outer_join(self, left_table, left_table_key_name,
                   right_table, right_table_key_name):
        # Do a left join.
        rows = self.left_join(left_table, left_table_key_name,
                              right_table, right_table_key_name).data

        # Find the not matching for the right.
        left_index = left_table.column_names.index(left_table_key_name)
        right_index = right_table.column_names.index(right_table_key_name)
        for right_row in right_table.data:
            found = None
            for left_row in left_table.data:
                if left_row[left_index] == right_row[right_index]:
                    found = True
            if not found:
                rows.append([None] * len(left_row) + right_row)

        return Table("OuterJoin",
                     self.renaming(left_table) + self.renaming(right_table),
                     rows)

    def renaming(self, table):
        return [table.name + '.' + table.column_name
                for table.column_name in table.column_names]


def main(argv):
    department_table = Table('departments', ['id', 'name'], [
        [0, 'engineering'],
        [1, 'finance']])
    user_table = Table('users', ['id', 'department_id', 'name'], [
        [0, 0, 'Ian'],
        [1, 0, 'John'],
        [2, 1, 'Eddie'],
        [3, 1, 'Mark']])
    salary_table = Table('salaries', ['id', 'user_id', 'amount'], [
        [0, 0, 100],
        [1, 1, 150],
        [2, 1, 200],
        [3, 3, 200],
        [4, 3, 300],
        [5, 4, 400]])

    db = DB()
    db.add_table(user_table)
    db.add_table(department_table)
    db.add_table(salary_table)

    # should print something like
    # id, department_id, name
    # 1, 0, John
    print db.table('users') \
            .where('id', 1) \
            .select(['id', 'department_id', 'name'])
    print

    # should print something like
    # users.name, departments.name
    # Ian, engineering
    # John, engineering
    print db.inner_join(
        db.table('users'), 'department_id', db.table('departments'), 'id') \
        .where('departments.name', 'engineering') \
        .select(['users.name', 'departments.name'])
    print

    # should print something like
    # users.name, salaries.amount
    # Ian, 100
    # John, 150
    # John, 200
    # Mark, 200
    # Mark, 300
    # Eddie, None
    print db.left_join(
        db.table('users'), 'id', db.table('salaries'), 'user_id') \
        .select(['users.name', 'salaries.amount'])
    print

    # should print something like
    # users.name, salaries.amount
    # Ian, 100
    # John, 150
    # John, 200
    # Mark, 200
    # Mark, 300
    # None, 400
    print db.right_join(
        db.table('users'), 'id', db.table('salaries'), 'user_id') \
        .select(['users.name', 'salaries.amount'])
    print

    # should print something like
    # users.name, salaries.amount
    # Ian, 100
    # John, 150
    # John, 200
    # Mark, 200
    # Mark, 300
    # Eddie, None
    # None, 400
    print db.outer_join(
        db.table('users'), 'id', db.table('salaries'), 'user_id') \
        .select(['users.name', 'salaries.amount'])

if __name__ == '__main__':
    # sys.exit(main(sys.argv[1:]))
    main(None)
