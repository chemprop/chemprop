# How to contribute

We welcome contributions from external contributors, and this document
describes how to merge code changes into this repository.

## Getting Started

* Make sure you have a [GitHub account](https://github.com/signup/free).
* [Fork](https://help.github.com/articles/fork-a-repo/) this repository on GitHub.
* On your local machine,
  [clone](https://help.github.com/articles/cloning-a-repository/) your fork of
  the repository.

## Making Changes

* Add some really awesome code to your local fork.  It's usually a [good
  idea](http://blog.jasonmeridth.com/posts/do-not-issue-pull-requests-from-your-master-branch/)
  to make changes on a
  [branch](https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/)
  with the branch name relating to the feature you are going to add.
* When you are ready for others to examine and comment on your new feature,
  navigate to your fork of `chemprop` on GitHub and open a [pull
  request](https://help.github.com/articles/using-pull-requests/) (PR). Note that
  after you launch a PR from one of your fork's branches, all
  subsequent commits to that branch will be added to the open pull request
  automatically.  Each commit added to the PR will be validated for
  mergability, compilation and test suite compliance; the results of these tests
  will be visible on the PR page.
* The PR description must say who you are. See [Identifying Yourself](#identifying-yourself).
* If you're providing a new feature, you **must** add test cases and documentation.
* When the code is ready to go, run the test suite with `pytest tests` and check
  the formatting as described in [Formatting](#formatting).
* When you're ready to be considered for merging, click the "Ready for review"
  box on the PR page to let the Chemprop devs know that the changes are complete.
  The code will not be merged until the continuous integration returns checkmarks,
  and at least one core developer gives "Approved" reviews.

## Formatting

The continuous integration checks formatting with `black`, `isort`, and `flake8`
before it runs any tests. If the formatting check fails, the tests are not run. To
fix formatting before you push, run these from the root of the repository:

```bash
pip install -e ".[dev]"   # installs black 23.*, isort, and flake8
isort .
black .
flake8 .                  # flake8 only reports problems; fix them by hand
```

## Identifying Yourself

Please give your name in the PR description, along with your institution or employer (if any).

You must also email the maintainers at chemprop@mit.edu from your institution
or work email address, so that we can confirm who you are. Include your GitHub
username and the PR number. If you do not have a school or work email address,
email us anyway and explain. We will not share your email address.

We ask for this for two reasons:

1. So that we can give credit to contributors.
2. To help protect against malicious changes by AI agents.

We will not merge a PR until we can confirm who its author is.

## Use of AI Tools

AI-generated code is welcome if it follows this AI policy. Parts of this policy
are adapted from the
[OpenMM AI policy](https://github.com/openmm/openmm/blob/c403e95b35546bc700c97875c9265a92d1b43ecb/AI_POLICY.md).

* **Disclose it.** If you used AI, say what you used it for. As OpenMM puts it:
  "This requirement is not meant to discourage any particular use of AI. It is
  simply so that everyone understands what they are dealing with."
* **Write the summary yourself.** Every PR must begin with a summary you wrote
  in your own words. AI-generated text may follow if you feel it is accurate
  and useful.
* **Understand what you submit.** You are responsible for the code you submit.
  You must understand the code and be ready to answer questions about it.
* **Take part in the discussion yourself.** You may use AI to help answer a
  maintainer's question, but do not just paste the question into an AI and post
  its reply. Assume the maintainer already asked an AI. They want to hear what
  you tried, why you made a choice, or what you know about your use case.
* **Fix the root cause.** Ensure any bug fixes address the underlying problem,
  not just hide the symptom.
* **Keep it maintainable.** Do not add unreasonable amounts of tech debt. New
  code should match the style of the rest of the code base.
* **Add the tests your change needs, and no more.** Tests should fail without
  your change and pass with it. Avoid many near-duplicate tests and tests of
  trivial behavior. You are responsible for reviewing any AI-generated tests.

## Additional Resources

* [General GitHub documentation](https://help.github.com/)
* [PR best practices](http://codeinthehole.com/writing/pull-requests-and-other-good-practices-for-teams-using-github/)
* [A guide to contributing to software packages](http://www.contribution-guide.org)
