---
name: Pull Request
about: Submit a pull request for review
title: '[type]: brief description of changes'
labels: ''
assignees: ''
---

## Description

<!-- Provide a clear and concise description of your changes -->
<!-- What problem does this PR solve? What does it implement? -->

## Type of Change

<!-- Mark the relevant option with an 'x' -->

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Performance improvement
- [ ] Code cleanup/refactoring
- [ ] Documentation update
- [ ] Testing improvements
- [ ] Other (please describe):

## Related Issues

<!-- Link to related issues using 'Fixes #123' or 'Closes #123' syntax -->
<!-- Example: Fixes #42, Closes #123 -->

Fixes #

## Testing

<!-- Describe the tests you ran and how you verified your changes -->

### Tests Added

- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Property-based tests added/updated
- [ ] No new tests needed (explain why):

### Tests Run

```bash
# Commands used to test
pytest tests/ -v
cargo test
```

### Test Results

<!-- Paste or describe test results -->
- All tests passing: [ ] Yes [ ] No
- Coverage maintained: [ ] Yes [ ] No
- Coverage percentage: __%

## Quality Checks

<!-- Confirm that you've run these checks -->

- [ ] Code follows project style guidelines (Rust: `cargo fmt`, Python: `black`)
- [ ] Code passes linting (Rust: `cargo clippy`, Python: `flake8`)
- [ ] All tests pass locally
- [ ] Pre-commit hooks pass
- [ ] Documentation updated (if applicable)
- [ ] No new warnings generated

## Performance Impact

<!-- Describe any performance implications -->

- [ ] Performance tested (if applicable)
- [ ] No performance degradation
- [ ] Performance improvement:
  - Before: ___
  - After: ___
  - Improvement: ___%

## Breaking Changes

<!-- List any breaking changes and migration steps -->

- [ ] No breaking changes
- [ ] Breaking changes documented:
  1.
  2.
  3.

## Screenshots (if applicable)

<!-- Add screenshots for UI changes or before/after comparisons -->

## Checklist

- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

## Additional Notes

<!-- Any additional information or context that reviewers should know -->

<!-- Thanks for contributing! -->
