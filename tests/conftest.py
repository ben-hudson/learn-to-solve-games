def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gamut_jar: invokes the real GAMUT jar (needs a Java runtime and $GAMUT_JAR)"
    )
