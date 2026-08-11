# Contributing

Contributions to ``photon-mosaic`` are very welcome and appreciated. This could be
fixing a bug, improving the documentation or developing a new feature. To get started, please read this contributing guide and the [introduction to `photon-mosaic` API for developers](./developer-intro.md).

If you're unsure about any part of the contributing process or have any questions, please
get in touch through our [Zulip chat](https://neuroinformatics.zulipchat.com/#narrow/channel/500681-photon-mosaic).
Otherwise, feel free to dive right in and start contributing by
[creating a development environment](#creating-a-development-environment)
and [opening a pull request](#pull-requests).

## Creating a development environment

To install ``photon-mosaic`` for development, first the
[GitHub repository](https://github.com/photon-mosaic/photon-mosaic-pipeline)
should be cloned. Then, you can change-directory
to the cloned repository and run pip install with the developer tag:

```sh
pip install -e .[dev]
```

or if using `zsh`:

```sh
pip install -e '.[dev]'
```

Finally, initialise the pre-commit hooks:

```bash
pre-commit install
```

## Pull requests

In all cases, please submit code to the main repository via a pull request. The developers recommend and adhere
to the following conventions:

- Please submit *draft* pull requests as early as possible (you can still push to the branch once submitted) to
  allow for discussion.
- One approval of a PR (by a repo maintainer) is sufficient for it to be merged.
- If the PR receives approval without additional comments, it will be merged immediately by the approving reviewer.
