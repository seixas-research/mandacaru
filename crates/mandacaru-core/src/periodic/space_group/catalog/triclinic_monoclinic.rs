//! Space groups 1-15 (triclinic and monoclinic).
use super::GroupEntry;
use crate::periodic::space_group::classify::CrystalSystem;

pub static GROUPS: &[GroupEntry] = &[
    GroupEntry {
        number: 1,
        symbol: "P1",
        crystal_system: CrystalSystem::Triclinic,
        point_group: "1",
        centering: 'P',
        operations: &[
            ([1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0]),
        ],
    },
    GroupEntry {
        number: 2,
        symbol: "P-1",
        crystal_system: CrystalSystem::Triclinic,
        point_group: "-1",
        centering: 'P',
        operations: &[
            ([1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0]),
            ([-1, 0, 0, 0, -1, 0, 0, 0, -1], [0, 0, 0]),
        ],
    },
    GroupEntry {
        number: 3,
        symbol: "P2",
        crystal_system: CrystalSystem::Monoclinic,
        point_group: "2",
        centering: 'P',
        operations: &[
            ([1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0]),
            ([-1, 0, 0, 0, 1, 0, 0, 0, -1], [0, 0, 0]),
        ],
    },
    GroupEntry {
        number: 4,
        symbol: "P2_1",
        crystal_system: CrystalSystem::Monoclinic,
        point_group: "2",
        centering: 'P',
        operations: &[
            ([1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0]),
            ([-1, 0, 0, 0, 1, 0, 0, 0, -1], [0, 12, 0]),
        ],
    },
    GroupEntry {
        number: 5,
        symbol: "C2",
        crystal_system: CrystalSystem::Monoclinic,
        point_group: "2",
        centering: 'C',
        operations: &[
            ([1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0]),
            ([-1, 0, 0, 0, 1, 0, 0, 0, -1], [0, 0, 0]),
        ],
    },
    GroupEntry {
        number: 6,
        symbol: "Pm",
        crystal_system: CrystalSystem::Monoclinic,
        point_group: "m",
        centering: 'P',
        operations: &[
            ([1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0]),
            ([1, 0, 0, 0, -1, 0, 0, 0, 1], [0, 0, 0]),
        ],
    },
    GroupEntry {
        number: 7,
        symbol: "Pc",
        crystal_system: CrystalSystem::Monoclinic,
        point_group: "m",
        centering: 'P',
        operations: &[
            ([1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0]),
            ([1, 0, 0, 0, -1, 0, 0, 0, 1], [0, 0, 12]),
        ],
    },
    GroupEntry {
        number: 8,
        symbol: "Cm",
        crystal_system: CrystalSystem::Monoclinic,
        point_group: "m",
        centering: 'C',
        operations: &[
            ([1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0]),
            ([1, 0, 0, 0, -1, 0, 0, 0, 1], [0, 0, 0]),
        ],
    },
    GroupEntry {
        number: 9,
        symbol: "Cc",
        crystal_system: CrystalSystem::Monoclinic,
        point_group: "m",
        centering: 'C',
        operations: &[
            ([1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0]),
            ([1, 0, 0, 0, -1, 0, 0, 0, 1], [0, 0, 12]),
        ],
    },
    GroupEntry {
        number: 10,
        symbol: "P2/m",
        crystal_system: CrystalSystem::Monoclinic,
        point_group: "2/m",
        centering: 'P',
        operations: &[
            ([1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0]),
            ([-1, 0, 0, 0, -1, 0, 0, 0, -1], [0, 0, 0]),
            ([-1, 0, 0, 0, 1, 0, 0, 0, -1], [0, 0, 0]),
            ([1, 0, 0, 0, -1, 0, 0, 0, 1], [0, 0, 0]),
        ],
    },
    GroupEntry {
        number: 11,
        symbol: "P2_1/m",
        crystal_system: CrystalSystem::Monoclinic,
        point_group: "2/m",
        centering: 'P',
        operations: &[
            ([1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0]),
            ([-1, 0, 0, 0, -1, 0, 0, 0, -1], [0, 0, 0]),
            ([-1, 0, 0, 0, 1, 0, 0, 0, -1], [0, 12, 0]),
            ([1, 0, 0, 0, -1, 0, 0, 0, 1], [0, 12, 0]),
        ],
    },
    GroupEntry {
        number: 12,
        symbol: "C2/m",
        crystal_system: CrystalSystem::Monoclinic,
        point_group: "2/m",
        centering: 'C',
        operations: &[
            ([1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0]),
            ([-1, 0, 0, 0, -1, 0, 0, 0, -1], [0, 0, 0]),
            ([-1, 0, 0, 0, 1, 0, 0, 0, -1], [0, 0, 0]),
            ([1, 0, 0, 0, -1, 0, 0, 0, 1], [0, 0, 0]),
        ],
    },
    GroupEntry {
        number: 13,
        symbol: "P2/c",
        crystal_system: CrystalSystem::Monoclinic,
        point_group: "2/m",
        centering: 'P',
        operations: &[
            ([1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0]),
            ([-1, 0, 0, 0, -1, 0, 0, 0, -1], [0, 0, 0]),
            ([-1, 0, 0, 0, 1, 0, 0, 0, -1], [0, 0, 12]),
            ([1, 0, 0, 0, -1, 0, 0, 0, 1], [0, 0, 12]),
        ],
    },
    GroupEntry {
        number: 14,
        symbol: "P2_1/c",
        crystal_system: CrystalSystem::Monoclinic,
        point_group: "2/m",
        centering: 'P',
        operations: &[
            ([1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0]),
            ([-1, 0, 0, 0, -1, 0, 0, 0, -1], [0, 0, 0]),
            ([-1, 0, 0, 0, 1, 0, 0, 0, -1], [0, 12, 12]),
            ([1, 0, 0, 0, -1, 0, 0, 0, 1], [0, 12, 12]),
        ],
    },
    GroupEntry {
        number: 15,
        symbol: "C2/c",
        crystal_system: CrystalSystem::Monoclinic,
        point_group: "2/m",
        centering: 'C',
        operations: &[
            ([1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0]),
            ([-1, 0, 0, 0, -1, 0, 0, 0, -1], [0, 0, 0]),
            ([-1, 0, 0, 0, 1, 0, 0, 0, -1], [0, 0, 12]),
            ([1, 0, 0, 0, -1, 0, 0, 0, 1], [0, 0, 12]),
        ],
    },
];
