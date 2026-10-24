# Task: run-length codec

Implement `encode(s)` and `decode(s)` in `pkg/rle.py`.

* `encode` replaces each run of one character `c` repeated `n` times by `c` followed by `n` when `n > 1`, and by
  `c` alone when `n == 1`. Runs are maximal.
* A digit or a backslash in the INPUT is escaped in the output with a backslash before it, before any run
  count: the run `"777"` encodes as `"\73"`, the run `"\\"` (one backslash) as `"\\"` (a backslash then a
  backslash).
* `decode` inverts `encode` exactly: `decode(encode(s)) == s` for every string `s`.
* Run counts may have several digits. Characters are any Unicode code points. The empty string encodes to the
  empty string.
