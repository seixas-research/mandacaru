//! The digest that guards a dataset handed to a crystal (`paw::digest`): SHA-256 against the
//! NIST vectors, across block boundaries.
use mandacaru_core::paw::digest::sha256;

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
    // One million 'a': 15625 blocks of 64 bytes.
    assert_eq!(
        sha256(&vec![b'a'; 1_000_000]),
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
    );
}

#[test]
fn a_pinned_entry_gives_its_content_digest_and_an_old_file_pin_is_refused() {
    use mandacaru_core::paw::digest::pinned_content;
    use serde_json::json;
    let digest = "ab".repeat(32);
    assert_eq!(
        pinned_content(&json!({"file": "x", "content_sha256": digest})).unwrap(),
        digest
    );
    assert_eq!(pinned_content(&json!({"file": ""})).unwrap(), "");
    let old = pinned_content(&json!({"file": "x", "sha256": digest}))
        .unwrap_err()
        .to_string();
    assert!(old.contains("save it again"), "{old}");
}
