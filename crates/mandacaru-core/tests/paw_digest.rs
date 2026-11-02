//! The file digest that guards a dataset handed to a hybrid crystal (`paw::digest`): SHA-256
//! against the NIST vectors and a streamed file.
use mandacaru_core::paw::digest::{sha256, sha256_file};
use std::path::PathBuf;

#[test]
fn the_digest_equals_the_nist_vectors_and_streams_a_file_across_block_boundaries() {
    assert_eq!(
        sha256(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        sha256(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
    // One million 'a', written to a file and streamed in 64 KiB blocks.
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("p2_digest_million_a.bin");
    std::fs::write(&path, vec![b'a'; 1_000_000]).unwrap();
    let digest = sha256_file(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    assert_eq!(
        digest,
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
    );
    assert!(sha256_file(&path).is_err(), "a missing file is an error");
}
