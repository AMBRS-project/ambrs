# Native initial-composition validation

MAM4 and PartMC input adapters reject invalid initial compositions even when
the Scenario was constructed manually, without either sampler. Validation at
these boundaries does not change which finite signed weights the sampling API
accepts.

For PartMC, every supplied initial mode must have one mass fraction per declared
species. Values must be finite, real, scalar and nonnegative. Invalid source
components are rejected before CAMP species filtering and name mapping, so a
discarded or overwritten component cannot hide invalid input. A zero particle
number does not exempt an invalid composition. No tolerance, clipping,
resampling, new normalization or inferred species composition is introduced.
Valid tiny/zero proportions and the existing ammonium-sulfate split are retained.

MAM4 additionally has mode-specific supported species sets and requires finite
positive normalization totals, as its own input schema describes. This does
not establish faithful output reconstruction of every native species.

The new PartMC check covers initial AerosolModalSizeState input through
create_input/create_inputs. Low-level directly supplied AeroMode objects and
emission/background time series are separate paths; their scalar versus
distribution-valued composition schemas need separate validation. Do not claim
that this initial-state guard validates every manually serialized native file.
Sampler-level nonnegativity, distribution conditioning and physical sum/closure
policies remain separate compatibility decisions. Native executable/reference
validation remains required.
