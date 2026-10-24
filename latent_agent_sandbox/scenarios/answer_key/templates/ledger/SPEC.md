# Task: a shared-expense ledger

Implement `balances(lines)` and `summary(lines)` in `pkg/ledger.py`.

Each line of `lines` is one of:

* `NAME pays NAME AMOUNT`: the first name's balance goes down by AMOUNT, the second's up by AMOUNT.
* `NAME splits AMOUNT among NAME,NAME,...`: the payer's balance goes down by AMOUNT; each listed name's balance
  goes up by an equal share. Shares are in whole cents; when the cents do not divide evenly, the leftover cents
  go one each to the listed names in the order written. The payer may appear in the list.
* A blank line or a line starting with `#` is ignored.

AMOUNT is a decimal with at most two places (`12`, `12.5`, `12.50`). Names are lower-case words. Any other line
raises `ValueError`.

`balances(lines)` returns `{name: cents}` with every name that appeared, as `int` cents: negative means the
person paid out more than they received, positive the reverse.

`summary(lines)` returns one line per name sorted by name, `name: +12.50` or `name: -0.05` (always a sign, two
decimals), joined with `\n`, and an empty string when there are no names.
