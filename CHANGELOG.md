# Change Log

## [1.0.2] - 2023-06-14

### New
- Added basic functionality for simulations. Implemented as method `.simulate()` 
  of the class `StrategyProfile`, e.g. a computed equilibrium. 
- Added the option to create non-generic random games. (`SGame.random_nongeneric_game`).
  Can be used with the `sgamesolver-timings` terminal script by specifying "nongeneric=True" as 
  game parameter in the Excel sheet. 

### Changed
- Parameter 'eta_fix' of Logtracing now defaults to True.

### Fixed
- Fixed some bugs regarding error reporting in `SGame.from_table()`
  (which occurred when the action labels where not natively strings).

## [1.0.1] - 2022-09-04

### New
- `SGame.from_table()` is more flexible now: to_state-column now also accepts a strings 
  formatted as 'state: prob, state: prob, ...'. See documentation for details.

## [1.0.0] - 2022-06-06

### New
- Everything!