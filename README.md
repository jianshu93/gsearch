# A rust classifier based on HNSW for prokaryotic genomes

archaea stands for: **A Rust Classifier Hierarchical Navigable SW graphs**

This package (**currently in development**) compute probminhash signature of  bacteria and archaea genomes and stores the id of bacteria and probminhash signature in a Hnsw structure for searching of new request genomes

This package is developped in collaboration with Jianshu Zhao

# Dependencies and Installation
* Two libraries, zeromq, libsodium are required to successfully compile. 

```
### if you are using Linux (ubuntu for example), install them first
sudo apt-get install libzmq-dev libsodium-dev

### if you are using MacOS
brew install zeromq
brew install libsodium

### if you are using conda or on a server, you can install them first, but remember to add library configuration path and dynamic library config path to you environmental variables
conda install -c anaconda zeromq libsodium
export PKG_CONFIG_PATH="~/miniconda3/lib/pkgconfig:$PKG_CONFIG_PATH"
export LD_CONFIG_PATH="~/miniconda3/lib:$LD_CONFIG_PATH"
```
  
*  Clone hnsw-rs and probminhash or get them from crate.io
    - git clone https://github.com/jean-pierreBoth/hnswlib-rs

    - git clone https://github.com/jean-pierreBoth/probminhash

* Clone kmerutils which is not in crate.io:
  
    - git clone https://github.com/jean-pierreBoth/kmerutils

* Get archaea directly from gitbub: (git clone https://github.com/jean-pierreBoth/archaea) as long as it is not in crate.io and run cargo build --release

* A dependency is provided as a feature. It uses the crate **annembed** that gives some statistics on the hnsw graph constructed (and will provide some visualization of data).
It is activated by running:
    -   *cargo build --release --features annembed_f*

## Sketching of genomes

The sketching and database is done by the module ***tohnsw***.

The sketching of reference genomes can take some time (one or 2 hours for 50000 bacterial genomes of NCBI for parameters giving a correct quality of sketching). Result is stored in 2 structures:
- A Hnsw structure storing rank of data processed and corresponding sketches.
- A Dictionary associating each rank to a fasta id and fasta filename.

The Hnsw structure is dumped *in hnswdump.hnsw.graph* and  *hnswdump.hnsw.data*
The Dictionary is dumped in a json file *seqdict.json*
## Requests

For requests  the module ***request*** is being used. It reloads the dumped files, hnsw and seqdict related
takes a list of fasta files containing requests and for each fasta file dumps the asked number of nearest neighbours.
  
## Classify
 The classify module is used to assign taxonomy information from requested neighbours to query genomes. Average nucleitide identity will be calculated

## compilation single module
```
cargo build --release --bin tohnsw
cargo build --release --bin request

```
