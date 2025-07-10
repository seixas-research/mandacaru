import os

LOGFILES = ['test.log']

def pytest_sessionfinish(session, exitstatus):
    print("\n\n FINISHING TESTS.")

