use mandacaru_core::{
    paw::{
        dataset::Dataset,
        generate::{generate, Options},
    },
    Context,
};

#[test]
fn native_paw_generation_roundtrips_and_retains_physical_operator_contracts() {
    let ctx = Context::new(Some(1), 128 << 20).unwrap();
    for (number, functional, charge) in [
        (1, "lda", 1.),
        (8, "lda", 6.),
        (14, "lda", 4.),
        (1, "pbe", 1.),
    ] {
        {
            let options = Options::parse(&format!(r#"{{"functional":"{functional}"}}"#)).unwrap();
            let dataset = generate(&ctx, number, &options).unwrap();
            assert_eq!(dataset.valence_charge, charge);
            assert!(dataset.reference_energy < 0.);
            for channel in &dataset.channels {
                assert!(channel.channel.duality_error < 1e-8);
                assert!(channel.channel.asymmetry < 5e-5);
                assert!(
                    (&channel.channel.kinetic + &channel.channel.potential
                        - &channel.channel.screened)
                        .amax()
                        < 1e-7
                );
            }
            let document = dataset.document().unwrap();
            let loaded = Dataset::from_document(&ctx, &document).unwrap();
            assert_eq!(dataset.ionic.as_slice(), loaded.ionic.as_slice());
            assert_eq!(loaded.atomic_number, number);
            let (values, slopes) = loaded.local_values(&[0., 100.]).unwrap();
            assert!(values.as_slice()[0].is_finite());
            assert_eq!(slopes.as_slice()[0], 0.);
            assert!((values.as_slice()[1] + charge / 100.).abs() < 1e-14);
            assert!((slopes.as_slice()[1] - charge / 10000.).abs() < 1e-14);
            for channel in &loaded.channels {
                let projectors = loaded.projector_values(channel.l, &[100.]).unwrap();
                assert!(projectors.as_slice().iter().all(|&v| v == 0.));
            }
            let mut malformed = document;
            malformed["r"][1] = malformed["r"][0].clone();
            let before = ctx.memory_used();
            assert!(Dataset::from_document(&ctx, &malformed).is_err());
            assert_eq!(ctx.memory_used(), before);
        }
        assert_eq!(ctx.memory_used(), 0);
    }
}
