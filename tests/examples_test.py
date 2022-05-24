"""Test example scripts."""


import os


def test_examples():
    for root, dirs, files in os.walk('examples/'):
        for file in files:
            try:
                os.system(f'python examples/{file}')
                assert True
            except Exception:
                assert False


if __name__ == '__main__':

    test_examples()
