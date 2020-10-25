# Linear OT 
Linear OT implementation for domain adaptation for aligning parallel unpaired scRNA/scATAC

## Dependencies

```
- numpy
- scipy
```

## Usage

Inputs are 
1. ```xs```, a matrix of "source" RNA expression levels from scRNA-seq
2. ```xt```, a matrix of "target" gene activities calcualated from ATAC-seq peak accessibility

Also, optionally:
1. ```ws```: vector of weights representing the size of each cluster/metacell from RNA
2. ```wt```: same as above but for ATAC
3. rho: float in range [0,1] representing whether the final transformation should be closer to RNA distribution (0) or ATAC distribution (1). Default value is 1.
4. reg: small float to make sure covariance matrices are invertible

```xs``` and ```xt``` should be dense matrices representing expression of highly variable genes.

To get transformed aligned ```xs``` and ```xt```, run the following:

```
from barycenter import LinearOT

model = LinearOT()

xs_transformed, xt_transformed = model.fit_transform(xs, xt)
```
