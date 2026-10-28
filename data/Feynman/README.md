# Feynman benchmarks

Thirteen synthetic regression tasks are registered in
[Program.cs](../../HyperOp.Experiments/Program.cs). Inputs are sampled independently
and uniformly in the order shown below. Physical constants are variable inputs;
`h` denotes reduced Planck's constant (ℏ).

| Runner identifier / descriptor | Inputs in descriptor order | Domains in that order | Target expression |
| --- | --- | --- | --- |
| I.6.20 / Feynman2 | sigma, theta | [1,3], [1,3] | exp(-theta²/(2 sigma²))/(sqrt(2 pi) sigma) |
| I.9.18 / Feynman5 | m1, m2, G, x1, x2, y1, y2, z1, z2 | [1,2]³, [3,4], [1,2], [3,4], [1,2], [3,4], [1,2] | G m1 m2 / ((x2-x1)²+(y2-y1)²+(z2-z1)²) |
| I.15.3.x / Feynman17 | x, u, c, t | [5,10], [1,2], [3,20], [1,2] | (x-u t)/sqrt(1-u²/c²) |
| I.30.5 / Feynman31 | lambd, d, n | [1,2], [2,5], [1,5] | asin(lambd/(n d)) |
| I.32.17 / Feynman33 | epsilon, c, Ef, r, omega, omega_0 | [1,2]⁵, [3,5] | (epsilon c Ef²/2)(8 pi r²/3) omega⁴/(omega²-omega_0²)² |
| I.41.16 / Feynman44 | omega, T, h, kb, c | [1,5]⁵ | h omega³/(pi² c² (exp(h omega/(kb T))-1)) |
| I.48.20 / Feynman50 | m, v, c | [1,5], [1,2], [3,20] | m c²/sqrt(1-v²/c²) |
| II.6.15a / Feynman56 | epsilon, p_d, r, x, y, z | [1,3]⁶ | 3 p_d z sqrt(x²+y²)/(4 pi epsilon r⁵) |
| II.11.27 / Feynman64 | n, alpha, epsilon, Ef | [0,1]², [1,2]² | n alpha epsilon Ef/(1-n alpha/3) |
| II.11.28 / Feynman65 | n, alpha | [0,1]² | 1+n alpha/(1-n alpha/3) |
| II.35.21 / Feynman81 | n_rho, mom, B, kb, T | [1,5]⁵ | n_rho mom tanh(mom B/(kb T)) |
| III.9.52 / Feynman90 | p_d, Ef, t, h, omega, omega_0 | [1,3]⁴, [1,5]² | p_d Ef t/h sinc²((omega-omega_0)t/2) |
| III.10.19 / Feynman91 | mom, Bx, By, Bz | [1,5]⁴ | mom sqrt(Bx²+By²+Bz²) |

`sinc(z) = sin(z)/z`, with `sinc(0) = 1`.

## Source conventions

Definitions follow [AI Feynman](https://arxiv.org/abs/1905.11481) and the domains in
[Haider's 2021 preprint](https://arxiv.org/abs/2107.09458), with these explicit choices:

- I.6.20 uses the two-input Gaussian with denominator `sqrt(2 pi) sigma`.
  Haider's preprint prints `sqrt(2 pi sigma)`.
- I.30.5 uses `lambd` in [1,2] so all `asin` arguments are valid.
- III.9.52 includes the squared sinc denominator from AI Feynman and its continuous
  limit at equal frequencies.

## Data and evaluation

The runner fixes the data seed at 42 and reuses the same noiseless observations
across methods and repetitions. Current defaults are 8,000 training, 2,000 validation
and 10,000 test rows.

GP fits on training data; HPO uses validation NMSE. Test data score the selected
expression and fresh-seed runs of the nominated configuration. Fresh GP seeds
measure search variability on the same data.

Run the equation, domain and partition checks from the repository root:

```powershell
dotnet run --project HyperOp.Checks -c Release
```
