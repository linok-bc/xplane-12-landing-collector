# Setup

Let `<PATH_TO_XPLANE>` be the path the X-Plane 12 local files. It should look something like `/home/<USER>/.local/share/Steam/steamapps/common/X-Plane 12`

This package requires some dependencies that are installed via pip. XPPython comes with its own version of Python; we will use their bundled pip installer. The path to this file should be something like `<PATH_TO_XPPLANE>/Resources/plugins/XPPython3/lin_x64/python3.12/bin/python3.12`. Please be cognisant that Python's version may change.

```
cd "<PATH_TO_XPLANE>"
Resources/plugins/XPPython3/lin_x64/python3.12/bin/python3.12 -sm pip install \
omegaconf \
timezonefinder
```

For video saving, we will be replacing the default location that X-Plane saves videos with a symlink to where we want our data to be stored. The base location is `<PATH_TO_XPLANE>/Outputs/screenshots`. Please back up your files if you have any there.
