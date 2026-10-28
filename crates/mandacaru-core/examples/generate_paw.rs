//! Generate a PAW reference and its projectors without Python.
use mandacaru_core::{
    paw::generate::{self, Options},
    Context,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    let number = args.get(1).map_or(Ok(1), |s| s.parse::<u8>())?;
    let options = Options::parse(args.get(2).map_or("{}", String::as_str))?;
    let ctx = Context::new(Some(1), 512 << 20)?;
    let species = generate::generate(&ctx, number, &options)?;
    println!(
        "{}",
        serde_json::json!({"atomic_number":number,"functional":species.functional,
            "valence_charge":species.valence_charge,"atom_energy_ha":species.reference_energy,
            "one_center_energy_ha":species.one_center_energy,
            "compensation_charge":species.compensation_charge,
            "channels":species.channels.iter().map(|ch|serde_json::json!({"l":ch.l,"radius_bohr":ch.radius,
                "reference_energies_ha":ch.energies,"duality_error":ch.channel.duality_error,
                "coupling_asymmetry":ch.channel.asymmetry,"residuals":ch.smooth.residuals})).collect::<Vec<_>>()
        })
    );
    Ok(())
}
