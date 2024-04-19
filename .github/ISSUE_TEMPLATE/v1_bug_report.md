---
name: v1 Bug Report
about: Report a bug in v1 (will not be fixed)
title: "[v1 BUG]: "
labels: bug, v1-wontfix
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**Example(s)**
Provide some examples of where the current code fails. Feel free to share your actual code for additional context, but a minimal and isolated example is preferred.

**Expected behavior**
A clear and concise description of what you expected to happen. If there is correct, expected output, include that here as well.

**Error Stack Trace**
If the bug is resulting in an error message, provide the _full_ stack trace (not just the last line). This is helpful for debugging, especially in cases where you aren't able to provide a minimum/isolated working example with accompanying files.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment**
- python version
- package versions: `conda list` or `pip list`
- OS

**Checklist**
- [ ] all dependencies are satisifed: `conda list` or `pip list` shows the packages listed in the `pyproject.toml`
- [ ] the unit tests are working: `pytest -v` reports no errors

**Additional context**
Add any other context about the problem here.
