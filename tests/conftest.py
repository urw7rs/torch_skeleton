def pytest_addoption(parser):
    parser.addoption("--root", action="store", default=".")
    parser.addoption("--num_workers", type=int, default=0)


def pytest_generate_tests(metafunc):
    option_value = metafunc.config.getoption("root")
    if "root" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("root", [option_value])

    option_value = metafunc.config.getoption("num_workers")
    if "num_workers" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("num_workers", [option_value])
