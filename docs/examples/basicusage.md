scDEF takes as input an [AnnData](https://anndata.readthedocs.io/en/latest/) object containing UMI counts. We strongly recommend filtering the data to keep only highly variable genes. After [installing scDEF](https://cbg-ethz.github.io/scDEF/installation), the basic usage is to import it:

```
import scdef
```

Then we create the `scDEF` object, passing an `AnnData` object with a layer containing the raw counts:
```
scd = scdef.scDEF(your_anndata_object, counts_layer='counts')
```
The `scDEF` object will hold a copy of your `AnnData` object and add annotations to it after fitting.

And we then fit it to the data:
```
scd.learn()
```

After fitting, the `scDEF` object is updated with the new variational parameters and a Graphviz graph, and the `AnnData` it contains is updated with new `.obs`, `.obsm` and `.uns` fields. The graph is stored in `scd.graph` and can be generated and updated with `scd.make_graph`.

Please see the example notebooks to learn how scDEF can be used for including hierarchical gene signature learning, clustering, and batch integration.
