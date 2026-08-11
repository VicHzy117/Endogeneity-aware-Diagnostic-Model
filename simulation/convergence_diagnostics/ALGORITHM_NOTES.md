# Algorithm notes

- Q is part of the ordinal-response likelihood. When `q_jk = 0`, the effective loading `Delta_jk` is exactly zero.
- Each Q coordinate uses the partially collapsed exact-Q Bayes-factor update followed by a positive half-normal draw only when active.
- The sampler uses a tail-stable truncated-normal routine to prevent deep-tail rejection stalls.
- Complete-data likelihood and `pi2` traces are deliberately not retained for convergence jobs; neither is needed for Rhat, and omitting them substantially reduces memory and disk use.
- The reported continuous measurement block is `Delta = Q * beta`, not an auxiliary inactive slab coefficient.
