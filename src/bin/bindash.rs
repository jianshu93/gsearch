use clap::{Arg, ArgAction, Command};
use env_logger;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::collections::HashMap;
use std::fmt::Debug;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use needletail::{parse_fastx_file, Sequence};
use num;

use kmerutils::base::{
    alphabet::Alphabet2b,
    kmergenerator::*,
    sequence::Sequence as SequenceStruct,
    CompressedKmerT,
    Kmer16b32bit,
    Kmer32bit,
    Kmer64bit,
    KmerBuilder,
};
use kmerutils::sketcharg::{DataType, SeqSketcherParams, SketchAlgo};
use kmerutils::sketching::setsketchert::{
    OptDensHashSketch,
    RevOptDensHashSketch,
    SeqSketcherT, // trait
};

use anndists::dist::{Distance, DistHamming};

/// Converts ASCII-encoded bases (from Needletail) into our `SequenceStruct`.
fn ascii_to_seq(bases: &[u8]) -> Result<SequenceStruct, ()> {
    let alphabet = Alphabet2b::new();
    let mut seq = SequenceStruct::with_capacity(2, bases.len());
    seq.encode_and_add(bases, &alphabet);
    Ok(seq)
}

/// Reads a list of file paths (one per line) from a text file.
fn read_genome_list(filepath: &str) -> Vec<String> {
    let file = File::open(filepath).expect("Cannot open genome list file");
    let reader = BufReader::new(file);
    reader
        .lines()
        .map(|line| line.expect("Error reading genome list"))
        .collect()
}

/// Generic function that sketches a list of FASTA/Q file paths using a provided
/// `SeqSketcherT` and a k-mer hash function. Returns a `HashMap` from file path
/// to the single sketch vector.
///
/// IMPORTANT:
/// - We still *initialize* OptDens/RevOptDens with `S = f32`
/// - BUT the `Sig` type returned by your kmerutils OptDensHashSketch/RevOptDensHashSketch
///   is `u64`, so we store `Vec<u64>`.
fn sketch_files<Kmer, F>(
    file_paths: &[String],
    sketcher: &(impl SeqSketcherT<Kmer, Sig = u64> + Sync),
    kmer_hash_fn: F,
) -> HashMap<String, Vec<u64>>
where
    Kmer: CompressedKmerT + KmerBuilder<Kmer> + Send + Sync,
    <Kmer as CompressedKmerT>::Val: num::PrimInt + Send + Sync + Debug,
    KmerGenerator<Kmer>: KmerGenerationPattern<Kmer>,
    F: Fn(&Kmer) -> <Kmer as CompressedKmerT>::Val + Send + Sync + Copy,
{
    file_paths
        .par_iter()
        .map(|path| {
            // Read all sequences from this FASTA/Q
            let mut sequences = Vec::new();
            let mut reader = parse_fastx_file(path).expect("Invalid FASTA/Q file");
            while let Some(record) = reader.next() {
                let seq_record = record.expect("Error reading sequence record");
                let seq_seq = seq_record.normalize(false).into_owned();
                let seq = ascii_to_seq(&seq_seq).unwrap();
                sequences.push(seq);
            }

            // Prepare references
            let sequences_ref: Vec<&SequenceStruct> = sequences.iter().collect();

            // sketch_compressedkmer_seqs returns Vec<Vec<Sig>> and inner vec has size 1 (per kmerutils design)
            let signature = sketcher.sketch_compressedkmer_seqs(&sequences_ref, kmer_hash_fn);

            (path.clone(), signature[0].clone())
        })
        .collect()
}

/// Computes the distance between two sketches (as `Vec<u64>`) using DistHamming,
/// then applies your transformation:
///
///    distance = -ln( (2*j) / (1 + j) ) / kmer_size
///
/// where j = 1 - hamming_distance (assuming DistHamming returns normalized distance).
fn compute_distance(query_sig: &[u64], reference_sig: &[u64], kmer_size: usize) -> f64 {
    let dist_hamming = DistHamming;

    // DistHamming::eval returns an f32 in many implementations; keep it generic and cast to f64.
    let h: f64 = dist_hamming.eval(query_sig, reference_sig) as f64;

    // Clamp to [0,1] for safety
    let h = if h <= 0.0 { 0.0 } else if h >= 1.0 { 1.0 } else { h };

    // Jaccard from Hamming similarity
    let mut j = 1.0 - h;

    // guard against j=0 => ln(0)
    if j <= 0.0 {
        j = f64::MIN_POSITIVE;
    }

    let fraction = (2.0 * j) / (1.0 + j);
    -fraction.ln() / (kmer_size as f64)
}

fn write_results(
    output: Option<String>,
    query_genomes: &[String],
    reference_genomes: &[String],
    query_sketches: &HashMap<String, Vec<u64>>,
    reference_sketches: &HashMap<String, Vec<u64>>,
    kmer_size: usize,
) {
    let mut output_writer: Box<dyn Write> = match output {
        Some(filename) => {
            Box::new(BufWriter::new(File::create(&filename).expect("Cannot create output file")))
        }
        None => Box::new(BufWriter::new(io::stdout())),
    };

    writeln!(output_writer, "Query\tReference\tDistance").expect("Error writing header");

    let all_pairs: Vec<(String, String)> = query_genomes
        .iter()
        .flat_map(|q| reference_genomes.iter().map(move |r| (q.clone(), r.clone())))
        .collect();

    let results: Vec<(String, String, f64)> = all_pairs
        .into_par_iter()
        .map(|(query_path, reference_path)| {
            let query_signature = &query_sketches[&query_path];
            let reference_signature = &reference_sketches[&reference_path];

            let mut distance = compute_distance(query_signature, reference_signature, kmer_size);

            let query_basename = Path::new(&query_path)
                .file_name()
                .and_then(|os_str| os_str.to_str())
                .unwrap_or(&query_path);

            let reference_basename = Path::new(&reference_path)
                .file_name()
                .and_then(|os_str| os_str.to_str())
                .unwrap_or(&reference_path);

            if query_basename == reference_basename {
                distance = 0.0;
            }

            (query_path, reference_path, distance)
        })
        .collect();

    for (q_path, r_path, dist) in results {
        writeln!(output_writer, "{}\t{}\t{:.6}", q_path, r_path, dist)
            .expect("Error writing output");
    }
}

/// Runs the pipeline for a given Kmer type and densification type (`dens`).
///
/// - `dens = 0` => `OptDensHashSketch`
/// - `dens = 1` => `RevOptDensHashSketch`
///
/// We instantiate the internal minhash with `S = f32`,
/// but the returned signature type is `u64` (per your kmerutils implementation).
fn sketching_kmerType<Kmer, F>(
    query_genomes: &[String],
    reference_genomes: &[String],
    sketch_args: &SeqSketcherParams,
    kmer_hash_fn: F,
    dens: usize,
    output: Option<String>,
    kmer_size: usize,
) where
    Kmer: CompressedKmerT + KmerBuilder<Kmer> + Send + Sync,
    <Kmer as CompressedKmerT>::Val: num::PrimInt + Send + Sync + Debug,
    KmerGenerator<Kmer>: KmerGenerationPattern<Kmer>,
    F: Fn(&Kmer) -> <Kmer as CompressedKmerT>::Val + Send + Sync + Copy,
{
    match dens {
        0 => {
            let sketcher = OptDensHashSketch::<Kmer, f32>::new(sketch_args);
            println!(
                "Sketching query genomes with OptDens (internal f32, output u64 signature)..."
            );
            let query_sketches = sketch_files(query_genomes, &sketcher, kmer_hash_fn);

            println!(
                "Sketching reference genomes with OptDens (internal f32, output u64 signature)..."
            );
            let reference_sketches = sketch_files(reference_genomes, &sketcher, kmer_hash_fn);

            println!("Performing pairwise comparisons...");
            write_results(
                output,
                query_genomes,
                reference_genomes,
                &query_sketches,
                &reference_sketches,
                kmer_size,
            );
        }
        1 => {
            let sketcher = RevOptDensHashSketch::<Kmer, f32>::new(sketch_args);
            println!(
                "Sketching query genomes with RevOptDens (internal f32, output u64 signature)..."
            );
            let query_sketches = sketch_files(query_genomes, &sketcher, kmer_hash_fn);

            println!(
                "Sketching reference genomes with RevOptDens (internal f32, output u64 signature)..."
            );
            let reference_sketches = sketch_files(reference_genomes, &sketcher, kmer_hash_fn);

            println!("Performing pairwise comparisons...");
            write_results(
                output,
                query_genomes,
                reference_genomes,
                &query_sketches,
                &reference_sketches,
                kmer_size,
            );
        }
        _ => panic!("Only densification = 0 or 1 are supported!"),
    }
}

fn main() {
    println!("\n ************** initializing logger *****************\n");
    let _ = env_logger::Builder::from_default_env().init();

    let matches = Command::new("BinDash")
        .version("0.1.3")
        .about("Binwise Densified MinHash for Genome/Metagenome/Pangenome Comparisons")
        .arg(
            Arg::new("query_list")
                .short('q')
                .long("query_list")
                .value_name("QUERY_LIST_FILE")
                .help("Query genome list file (one FASTA/FNA file path per line, .gz supported)")
                .required(true)
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("reference_list")
                .short('r')
                .long("reference_list")
                .value_name("REFERENCE_LIST_FILE")
                .help("Reference genome list file (one FASTA/FNA file path per line, .gz supported)")
                .required(true)
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("kmer_size")
                .short('k')
                .long("kmer_size")
                .value_name("KMER_SIZE")
                .help("K-mer size")
                .default_value("16")
                .value_parser(clap::value_parser!(usize))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("sketch_size")
                .short('s')
                .long("sketch_size")
                .value_name("SKETCH_SIZE")
                .help("MinHash sketch size")
                .default_value("2048")
                .value_parser(clap::value_parser!(usize))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("dens_opt")
                .short('d')
                .long("densification")
                .value_name("DENS_OPT")
                .help("Densification strategy, 0 = optimal densification, 1 = reverse optimal/faster densification")
                .default_value("0")
                .value_parser(clap::value_parser!(usize))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("threads")
                .short('t')
                .long("threads")
                .value_name("THREADS")
                .help("Number of threads to use in parallel")
                .default_value("1")
                .value_parser(clap::value_parser!(usize))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("OUTPUT_FILE")
                .help("Output file (defaults to stdout)")
                .required(false)
                .action(ArgAction::Set),
        )
        .get_matches();

    let query_list = matches
        .get_one::<String>("query_list")
        .expect("Query list is required")
        .to_string();
    let reference_list = matches
        .get_one::<String>("reference_list")
        .expect("Reference list is required")
        .to_string();
    let kmer_size = *matches.get_one::<usize>("kmer_size").unwrap();
    let sketch_size = *matches.get_one::<usize>("sketch_size").unwrap();
    let dens = *matches.get_one::<usize>("dens_opt").unwrap();
    let threads = *matches.get_one::<usize>("threads").unwrap();
    let output = matches.get_one::<String>("output").cloned();

    ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();

    let query_genomes = read_genome_list(&query_list);
    let reference_genomes = read_genome_list(&reference_list);

    let sketch_args = SeqSketcherParams::new(
        kmer_size,
        sketch_size,
        SketchAlgo::OPTDENS, // actual algo chosen by sketcher type
        DataType::DNA,
    );

    if kmer_size <= 14 {
        let nb_alphabet_bits = 2;
        let kmer_hash_fn_32bit = move |kmer: &Kmer32bit| -> <Kmer32bit as CompressedKmerT>::Val {
            let mask: <Kmer32bit as CompressedKmerT>::Val =
                num::NumCast::from::<u64>((1u64 << (nb_alphabet_bits * kmer.get_nb_base())) - 1)
                    .unwrap();
            kmer.get_compressed_value() & mask
        };

        sketching_kmerType::<Kmer32bit, _>(
            &query_genomes,
            &reference_genomes,
            &sketch_args,
            kmer_hash_fn_32bit,
            dens,
            output,
            kmer_size,
        );
    } else if kmer_size == 16 {
        let nb_alphabet_bits = 2;
        let kmer_hash_fn_16b32bit =
            move |kmer: &Kmer16b32bit| -> <Kmer16b32bit as CompressedKmerT>::Val {
                let canonical = kmer.reverse_complement().min(*kmer);
                let mask: <Kmer16b32bit as CompressedKmerT>::Val =
                    num::NumCast::from::<u64>(
                        (1u64 << (nb_alphabet_bits * kmer.get_nb_base())) - 1,
                    )
                    .unwrap();
                canonical.get_compressed_value() & mask
            };

        sketching_kmerType::<Kmer16b32bit, _>(
            &query_genomes,
            &reference_genomes,
            &sketch_args,
            kmer_hash_fn_16b32bit,
            dens,
            output,
            kmer_size,
        );
    } else if kmer_size <= 32 {
        let nb_alphabet_bits = 2;
        let kmer_hash_fn_64bit = move |kmer: &Kmer64bit| -> <Kmer64bit as CompressedKmerT>::Val {
            let canonical = kmer.reverse_complement().min(*kmer);
            let mask: <Kmer64bit as CompressedKmerT>::Val =
                num::NumCast::from::<u64>((1u64 << (nb_alphabet_bits * kmer.get_nb_base())) - 1)
                    .unwrap();
            canonical.get_compressed_value() & mask
        };

        sketching_kmerType::<Kmer64bit, _>(
            &query_genomes,
            &reference_genomes,
            &sketch_args,
            kmer_hash_fn_64bit,
            dens,
            output,
            kmer_size,
        );
    } else {
        panic!("kmer_size must not be 15 and cannot exceed 32!");
    }
}