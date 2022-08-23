# Debug

## Pasos en Matlab

### Iteración T=1

- $s(1) = W_{k(1)} \cdot d^2_{k(1)}$
- $f(1) = prior \cdot \exp(s(1)) = prior \cdot \exp(W_{k(1)} \cdot d^2_{k(1)})$
- $p(1) = f(1)/\sum f(1)$

### Iteración T=2

- $s(2)= s(1)+ W_{k(1)} \cdot d^2_{k(2)} = W_{k(1)} \cdot d^2_{k(1)} + W_{k(2)} \cdot d^2_{k(2)}$
- $f(2)= cte \cdot p(1) \cdot \exp(s(2)) = cte \cdot \frac{f(1)}{\sum f(1)} \cdot \exp(s(2))$
- $f(2) = cte \cdot \frac{prior \cdot \exp(W_{k(1)} \cdot d^2_{k(1)})}{\sum f(1)} \cdot \exp(W_{k(1)} \cdot d^2_{k(1)} + W_{k(2)} \cdot d^2_{k(2)})$
- $f(2) = \frac{cte}{\sum f(1)} \cdot prior \cdot \exp(W_{k(1)} \cdot d^2_{k(1)}) \cdot \exp(W_{k(1)} \cdot d^2_{k(1)} + W_{k(2)} \cdot d^2_{k(2)})$
- $p(2) = f(2)/\sum f(2)$
