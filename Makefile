VERSION = "0.0.1"
version:
    @echo 'Using version value defined in Makefile'
    @echo 'Updating to version ${VERSION} in following files:'
    @echo 'src/astropalmerio/__init__.py'
    @./update_version.sh ${VERSION}