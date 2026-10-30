//! The Fourier filter of a radial basis function: parity with the frozen numbers of
//! the Python implementation it replaced, the transform identities, input
//! validation and the memory bound (independent of the cutoff).
use mandacaru_core::basis::filter::{
    band_weight, filter, mask_at, window_at, Method, Options, MAX_L,
};
use mandacaru_core::{Context, Error};
use std::alloc::{GlobalAlloc, Layout, System};
use std::f64::consts::PI;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

/// Counts the live heap bytes and their high-water mark.
struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = System.alloc(layout);
        if !pointer.is_null() {
            let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        System.dealloc(pointer, layout);
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let moved = System.realloc(pointer, layout, new_size);
        if !moved.is_null() {
            if new_size >= layout.size() {
                let live = LIVE.fetch_add(new_size - layout.size(), Ordering::Relaxed) + new_size
                    - layout.size();
                PEAK.fetch_max(live, Ordering::Relaxed);
            } else {
                LIVE.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        moved
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// One test at a time: the allocator counts the whole process.
static SERIAL: Mutex<()> = Mutex::new(());
fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn context() -> Context {
    Context::new(Some(2), 256 << 20).unwrap()
}

fn linspace(first: f64, last: f64, n: usize) -> Vec<f64> {
    let step = (last - first) / (n - 1) as f64;
    let mut out = (0..n).map(|i| first + i as f64 * step).collect::<Vec<_>>();
    out[n - 1] = last;
    out
}

fn gaussian(r: &[f64], l: i32, a: f64) -> Vec<f64> {
    r.iter().map(|x| x.powi(l) * (-a * x * x).exp()).collect()
}

#[derive(Clone, Copy)]
enum Grid {
    Uniform,
    Log,
    Zero,
}
impl Grid {
    fn mesh(self) -> Vec<f64> {
        match self {
            Grid::Uniform => linspace(0.02, 25., 700),
            Grid::Log => linspace(0., (25.0f64 / 0.01).ln(), 600)
                .iter()
                .map(|x| 0.01 * x.exp())
                .collect(),
            Grid::Zero => linspace(0., 20., 500),
        }
    }
}

/// A case frozen from the Python filter (`numpy` + `scipy`) before it was deleted:
/// the input is `r^l exp(-a r^2)` on the named mesh; `report` is
/// `[residual_before, residual_after, support_radius, mask_radius, norm_kept, max_change]`.
struct Case {
    tag: &'static str,
    grid: Grid,
    l: usize,
    a: f64,
    k_c: f64,
    method: &'static str,
    rolloff: f64,
    oversample: usize,
    renormalize: bool,
    margin: f64,
    report: [f64; 6],
    masked: bool,
    sum: f64,
    sum_squares: f64,
    samples: [f64; 6],
    indices: [usize; 6],
}

const CASES: &[Case] = &[
    Case {
        tag: "gauss_uni_l0_plain",
        grid: Grid::Uniform,
        l: 0,
        a: 0.8,
        k_c: 4.0,
        method: "plain",
        rolloff: 0.25,
        oversample: 8,
        renormalize: true,
        margin: 1.5,
        report: [
            0.00018266786935761115,
            1.1887220093598216e-05,
            3.129098712446352,
            4.693648068669528,
            0.9967865747120871,
            0.056171232303109116,
        ],
        masked: false,
        sum: 27.049422220531238,
        sum_squares: 18.32473655443976,
        samples: [
            0.9435267908101066,
            -0.0011178419653431898,
            6.818247118951281e-05,
            1.500464276086571e-05,
            3.406642360876929e-06,
            8.560454858419462e-07,
        ],
        indices: [0, 100, 233, 350, 466, 699],
    },
    Case {
        tag: "gauss_uni_l0_switch",
        grid: Grid::Uniform,
        l: 0,
        a: 0.8,
        k_c: 4.0,
        method: "switch",
        rolloff: 0.25,
        oversample: 8,
        renormalize: true,
        margin: 1.5,
        report: [
            0.00018266786935761115,
            2.6328074683057423e-05,
            3.129098712446352,
            4.693648068669528,
            0.9966857882340671,
            0.05612351276411712,
        ],
        masked: true,
        sum: 27.051838057094606,
        sum_squares: 18.32655889285308,
        samples: [
            0.9435744950812891,
            -0.0008918226157329206,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        indices: [0, 100, 233, 350, 466, 699],
    },
    Case {
        tag: "gauss_uni_l0_mask",
        grid: Grid::Uniform,
        l: 0,
        a: 0.8,
        k_c: 4.0,
        method: "mask",
        rolloff: 0.25,
        oversample: 8,
        renormalize: true,
        margin: 1.5,
        report: [
            0.00018266786935761115,
            2.748455691403695e-05,
            3.129098712446352,
            4.693648068669528,
            0.9966703285116782,
            0.05612932155491848,
        ],
        masked: true,
        sum: 27.054098767133496,
        sum_squares: 18.32658212367919,
        samples: [
            0.9435686881490034,
            -0.0006960673664444648,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        indices: [0, 100, 233, 350, 466, 699],
    },
    Case {
        tag: "gauss_log_l0",
        grid: Grid::Log,
        l: 0,
        a: 0.6,
        k_c: 3.0,
        method: "switch",
        rolloff: 0.25,
        oversample: 8,
        renormalize: true,
        margin: 1.5,
        report: [
            0.001788588068804775,
            6.316854959598217e-05,
            3.61728069166264,
            5.425921037493961,
            0.9842532988300745,
            0.12233944918705418,
        ],
        masked: true,
        sum: 314.0728647305235,
        sum_squares: 256.4037024114644,
        samples: [
            0.8776078927596545,
            0.8772781935947955,
            0.870213449073207,
            0.7808106958388354,
            0.14048420090599556,
            0.0,
        ],
        indices: [0, 85, 200, 300, 400, 599],
    },
    Case {
        tag: "gauss_uni_l1_plain",
        grid: Grid::Uniform,
        l: 1,
        a: 1.1,
        k_c: 4.0,
        method: "plain",
        rolloff: 0.25,
        oversample: 8,
        renormalize: true,
        margin: 1.5,
        report: [
            0.012493812418414851,
            1.3952713429077335e-08,
            2.8789413447782546,
            4.318412017167382,
            0.9297405332526574,
            0.2429505548612092,
        ],
        masked: false,
        sum: 11.357826430056452,
        sum_squares: 2.916067722639342,
        samples: [
            0.013393629396414718,
            0.0020519817224266407,
            -8.593503171936839e-07,
            -1.2515016185579708e-05,
            -4.656087734876576e-06,
            -1.1772810064544335e-06,
        ],
        indices: [0, 100, 233, 350, 466, 699],
    },
    Case {
        tag: "gauss_uni_l1_switch",
        grid: Grid::Uniform,
        l: 1,
        a: 1.1,
        k_c: 4.0,
        method: "switch",
        rolloff: 0.25,
        oversample: 8,
        renormalize: true,
        margin: 1.5,
        report: [
            0.012493812418414851,
            0.0003443367826569954,
            2.8789413447782546,
            4.318412017167382,
            0.9274331535413177,
            0.24211444542589045,
        ],
        masked: true,
        sum: 11.385667973034733,
        sum_squares: 2.9229376576974655,
        samples: [
            0.013410280187200791,
            0.0010384759959004533,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
        ],
        indices: [0, 100, 233, 350, 466, 699],
    },
    Case {
        tag: "gauss_uni_l1_mask",
        grid: Grid::Uniform,
        l: 1,
        a: 1.1,
        k_c: 4.0,
        method: "mask",
        rolloff: 0.25,
        oversample: 8,
        renormalize: true,
        margin: 1.5,
        report: [
            0.012493812418414851,
            0.00034593821442086004,
            2.8789413447782546,
            4.318412017167382,
            0.9269357440348112,
            0.241916693964415,
        ],
        masked: true,
        sum: 11.37501409616174,
        sum_squares: 2.924438534629651,
        samples: [
            0.013414270086830282,
            0.0007738863130437258,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
        ],
        indices: [0, 100, 233, 350, 466, 699],
    },
    Case {
        tag: "gauss_log_l1",
        grid: Grid::Log,
        l: 1,
        a: 0.6,
        k_c: 3.0,
        method: "switch",
        rolloff: 0.25,
        oversample: 8,
        renormalize: true,
        margin: 1.5,
        report: [
            0.010337417922697822,
            0.00026766676483180873,
            3.9121755737362958,
            5.868263360604444,
            0.9356284797704956,
            0.228750948360778,
        ],
        masked: true,
        sum: 69.7524546760139,
        sum_squares: 21.48953031882617,
        samples: [
            0.006864120370348483,
            0.020826898295350532,
            0.09289623926432798,
            0.3128282326863489,
            0.28190768221930446,
            -0.0,
        ],
        indices: [0, 85, 200, 300, 400, 599],
    },
    Case {
        tag: "gauss_uni_l2_plain",
        grid: Grid::Uniform,
        l: 2,
        a: 1.4,
        k_c: 4.0,
        method: "plain",
        rolloff: 0.25,
        oversample: 8,
        renormalize: true,
        margin: 1.5,
        report: [
            0.1209928854928044,
            5.4438866414763254e-08,
            2.700257510729614,
            4.050386266094421,
            0.6751087813429811,
            0.49527369913460934,
        ],
        masked: false,
        sum: 5.8265049017272235,
        sum_squares: 0.8088269410162159,
        samples: [
            0.00013020423044168508,
            0.00797700549026618,
            -0.00016237619394802156,
            -5.218463612716485e-05,
            -4.243742427226614e-06,
            -2.775602537860642e-06,
        ],
        indices: [0, 100, 233, 350, 466, 699],
    },
    Case {
        tag: "gauss_uni_l2_switch",
        grid: Grid::Uniform,
        l: 2,
        a: 1.4,
        k_c: 4.0,
        method: "switch",
        rolloff: 0.25,
        oversample: 8,
        renormalize: true,
        margin: 1.5,
        report: [
            0.1209928854928044,
            0.0017252220460322043,
            2.700257510729614,
            4.050386266094421,
            0.6659094152086535,
            0.49222446929080843,
        ],
        masked: true,
        sum: 5.82543102798949,
        sum_squares: 0.8188691698589359,
        samples: [
            0.00013110051444008657,
            0.002062170054316687,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
        ],
        indices: [0, 100, 233, 350, 466, 699],
    },
    Case {
        tag: "gauss_uni_l2_mask",
        grid: Grid::Uniform,
        l: 2,
        a: 1.4,
        k_c: 4.0,
        method: "mask",
        rolloff: 0.25,
        oversample: 8,
        renormalize: true,
        margin: 1.5,
        report: [
            0.1209928854928044,
            0.0018456633131419542,
            2.700257510729614,
            4.050386266094421,
            0.664085043973245,
            0.49162105523741995,
        ],
        masked: true,
        sum: 5.838257790774605,
        sum_squares: 0.8206667583794947,
        samples: [
            0.00013127832410558403,
            0.0014772037664619083,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
        ],
        indices: [0, 100, 233, 350, 466, 699],
    },
    Case {
        tag: "gauss_log_l2",
        grid: Grid::Log,
        l: 2,
        a: 0.6,
        k_c: 3.0,
        method: "switch",
        rolloff: 0.25,
        oversample: 8,
        renormalize: true,
        margin: 1.5,
        report: [
            0.035979898993998116,
            0.0006116863666595362,
            4.122010450208059,
            6.183015675312088,
            0.8362008130057156,
            0.3491965144679534,
        ],
        masked: true,
        sum: 46.32467263834483,
        sum_squares: 15.458415935155195,
        samples: [
            4.7081192069281756e-05,
            0.0004335961337380908,
            0.008694485911486843,
            0.10954030186639635,
            0.4623554258092961,
            -0.0,
        ],
        indices: [0, 85, 200, 300, 400, 599],
    },
    Case {
        tag: "gauss_uni_l3_plain",
        grid: Grid::Uniform,
        l: 3,
        a: 1.7,
        k_c: 4.0,
        method: "plain",
        rolloff: 0.25,
        oversample: 8,
        renormalize: true,
        margin: 1.5,
        report: [
            0.40016775938195914,
            1.2538271332296347e-07,
            2.5930472103004294,
            3.889570815450644,
            0.3340706345451627,
            0.7131378169217765,
        ],
        masked: false,
        sum: 3.291439561799543,
        sum_squares: 0.27024869934096646,
        samples: [
            9.386182011274644e-07,
            0.0045703287749646205,
            -0.00020520476032567493,
            -1.7755457592774461e-06,
            6.784169806166584e-06,
            1.2415809864880508e-06,
        ],
        indices: [0, 100, 233, 350, 466, 699],
    },
    Case {
        tag: "gauss_uni_l3_switch",
        grid: Grid::Uniform,
        l: 3,
        a: 1.7,
        k_c: 4.0,
        method: "switch",
        rolloff: 0.25,
        oversample: 8,
        renormalize: true,
        margin: 1.5,
        report: [
            0.40016775938195914,
            0.0051443466536931215,
            2.5930472103004294,
            3.889570815450644,
            0.3207134237834721,
            0.7078967789634247,
        ],
        masked: true,
        sum: 3.3352253493959494,
        sum_squares: 0.2794572158925472,
        samples: [
            9.57964804530343e-07,
            0.0005742155163657907,
            -0.0,
            -0.0,
            0.0,
            0.0,
        ],
        indices: [0, 100, 233, 350, 466, 699],
    },
    Case {
        tag: "gauss_uni_l3_mask",
        grid: Grid::Uniform,
        l: 3,
        a: 1.7,
        k_c: 4.0,
        method: "mask",
        rolloff: 0.25,
        oversample: 8,
        renormalize: true,
        margin: 1.5,
        report: [
            0.40016775938195914,
            0.005434405160335554,
            2.5930472103004294,
            3.889570815450644,
            0.31752512158298274,
            0.70661111021037,
        ],
        masked: true,
        sum: 3.388967878069981,
        sum_squares: 0.2814707478518341,
        samples: [
            9.627037717251336e-07,
            0.00040620077935654025,
            -0.0,
            -0.0,
            0.0,
            0.0,
        ],
        indices: [0, 100, 233, 350, 466, 699],
    },
    Case {
        tag: "gauss_log_l3",
        grid: Grid::Log,
        l: 3,
        a: 0.6,
        k_c: 3.0,
        method: "switch",
        rolloff: 0.25,
        oversample: 8,
        renormalize: true,
        margin: 1.5,
        report: [
            0.09092330394020043,
            0.0010443920174810772,
            4.343100106674751,
            6.514650160012128,
            0.6899528196956269,
            0.4707844418185014,
        ],
        masked: true,
        sum: 47.89606715792865,
        sum_squares: 22.219923533555765,
        samples: [
            2.8818506736805613e-07,
            8.055701428403285e-06,
            0.0007260290823086804,
            0.034122400239952534,
            0.6336690153288701,
            0.0,
        ],
        indices: [0, 85, 200, 300, 400, 599],
    },
    Case {
        tag: "gauss_zero_l0",
        grid: Grid::Zero,
        l: 0,
        a: 0.9,
        k_c: 3.0,
        method: "switch",
        rolloff: 0.25,
        oversample: 8,
        renormalize: true,
        margin: 1.5,
        report: [
            0.018570977019800594,
            0.0005702829088656269,
            2.9258517034068134,
            4.38877755511022,
            0.9246075106915553,
            0.2550792358909052,
        ],
        masked: true,
        sum: 20.929738280595007,
        sum_squares: 11.685703104612589,
        samples: [
            0.7449207641090948,
            -0.010889376433907031,
            0.0,
            -0.0,
            -0.0,
            0.0,
        ],
        indices: [0, 71, 166, 250, 333, 499],
    },
    Case {
        tag: "gauss_zero_l2",
        grid: Grid::Zero,
        l: 2,
        a: 0.9,
        k_c: 3.0,
        method: "switch",
        rolloff: 0.25,
        oversample: 8,
        renormalize: true,
        margin: 1.5,
        report: [
            0.18858960675416536,
            0.0027449171708323616,
            3.3667334669338675,
            5.050100200400801,
            0.5692709246101151,
            0.5577349643667293,
        ],
        masked: true,
        sum: 9.655641869239396,
        sum_squares: 1.9759124087492246,
        samples: [0.0, 0.019013851843372776, 0.0, 0.0, 0.0, 0.0],
        indices: [0, 71, 166, 250, 333, 499],
    },
    Case {
        tag: "opt_rolloff0",
        grid: Grid::Uniform,
        l: 1,
        a: 0.8,
        k_c: 4.0,
        method: "switch",
        rolloff: 0.0,
        oversample: 8,
        renormalize: true,
        margin: 1.5,
        report: [
            0.0012501790714778416,
            0.00023476040547243393,
            3.379256080114449,
            5.068884120171674,
            0.9982877928767365,
            0.046027221932887656,
        ],
        masked: true,
        sum: 17.172033405066347,
        sum_squares: 5.908812938198817,
        samples: [
            0.018483636835818162,
            -0.001622301151596361,
            -0.0,
            0.0,
            0.0,
            0.0,
        ],
        indices: [0, 100, 233, 350, 466, 699],
    },
    Case {
        tag: "opt_rolloff05",
        grid: Grid::Uniform,
        l: 1,
        a: 0.8,
        k_c: 4.0,
        method: "switch",
        rolloff: 0.5,
        oversample: 8,
        renormalize: true,
        margin: 1.5,
        report: [
            0.0012501790714778416,
            4.997683813456888e-06,
            3.379256080114449,
            5.068884120171674,
            0.8799210046581092,
            0.23751359356868246,
        ],
        masked: true,
        sum: 15.750536315334204,
        sum_squares: 4.709854489148451,
        samples: [
            0.013585642459659753,
            0.0007544095316696964,
            -0.0,
            -0.0,
            0.0,
            -0.0,
        ],
        indices: [0, 100, 233, 350, 466, 699],
    },
    Case {
        tag: "opt_norenorm",
        grid: Grid::Uniform,
        l: 0,
        a: 0.8,
        k_c: 4.0,
        method: "switch",
        rolloff: 0.25,
        oversample: 8,
        renormalize: false,
        margin: 1.5,
        report: [
            0.00018266786935761115,
            2.6328074683057423e-05,
            3.129098712446352,
            4.693648068669528,
            0.9966857882340671,
            0.05768891413825508,
        ],
        masked: true,
        sum: 27.00697309325644,
        sum_squares: 18.265820795741316,
        samples: [
            0.9420095945554509,
            -0.0008903435447241237,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        indices: [0, 100, 233, 350, 466, 699],
    },
    Case {
        tag: "opt_oversample4",
        grid: Grid::Uniform,
        l: 2,
        a: 0.8,
        k_c: 4.0,
        method: "switch",
        rolloff: 0.25,
        oversample: 4,
        renormalize: true,
        margin: 1.5,
        report: [
            0.005575834616685538,
            0.000143020554021156,
            3.5936766809728184,
            5.390515021459228,
            0.9466776215548847,
            0.2020717190184641,
        ],
        masked: true,
        sum: 15.851211218526485,
        sum_squares: 4.8320733969036525,
        samples: [
            0.00025962795222056803,
            0.008860811846699358,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
        ],
        indices: [0, 100, 233, 350, 466, 699],
    },
    Case {
        tag: "opt_margin2",
        grid: Grid::Uniform,
        l: 0,
        a: 0.8,
        k_c: 4.0,
        method: "switch",
        rolloff: 0.25,
        oversample: 8,
        renormalize: true,
        margin: 2.0,
        report: [
            0.00018266786935761115,
            1.7308796431914075e-05,
            3.129098712446352,
            6.258197424892704,
            0.996720924552348,
            0.05614014963615327,
        ],
        masked: true,
        sum: 27.05033720614028,
        sum_squares: 18.325926388059127,
        samples: [
            0.9435578635322003,
            -0.0010581718011988248,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        indices: [0, 100, 233, 350, 466, 699],
    },
    Case {
        tag: "opt_margin2_mask",
        grid: Grid::Uniform,
        l: 1,
        a: 0.8,
        k_c: 4.0,
        method: "mask",
        rolloff: 0.25,
        oversample: 8,
        renormalize: true,
        margin: 2.0,
        report: [
            0.0012501790714778416,
            1.4797496585616798e-05,
            3.379256080114449,
            6.758512160228898,
            0.9833625184642091,
            0.11865649379911566,
        ],
        masked: true,
        sum: 16.618869995609305,
        sum_squares: 5.480303907686815,
        samples: [
            0.016478691413553393,
            0.0025425405552753724,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
        ],
        indices: [0, 100, 233, 350, 466, 699],
    },
    Case {
        tag: "opt_hard",
        grid: Grid::Uniform,
        l: 0,
        a: 3.0,
        k_c: 1.5,
        method: "switch",
        rolloff: 0.25,
        oversample: 8,
        renormalize: true,
        margin: 1.5,
        report: [
            0.8613782315720482,
            0.2303223289900571,
            1.6281545064377683,
            2.4422317596566523,
            0.05330717585664002,
            0.835593276950713,
        ],
        masked: true,
        sum: 7.391789930788709,
        sum_squares: 0.9934393701708794,
        samples: [0.16420955330713363, -0.0, -0.0, 0.0, 0.0, 0.0],
        indices: [0, 100, 233, 350, 466, 699],
    },
];

/// `a` equals `b` to `rel` of `b`, with an absolute floor for the residuals, which are
/// differences `1 - inside` of two numbers near one.
fn close(a: f64, b: f64, rel: f64, floor: f64) -> bool {
    (a - b).abs() <= rel * b.abs() + floor
}

#[test]
fn the_frozen_python_numbers_are_reproduced() {
    let _guard = serial();
    let ctx = context();
    let mut worst = 0.0f64;
    for case in CASES {
        let r = case.grid.mesh();
        let values = gaussian(&r, case.l as i32, case.a);
        let options = Options {
            method: Method::parse(case.method).unwrap(),
            rolloff: case.rolloff,
            oversample: case.oversample,
            renormalize: case.renormalize,
            switch_margin: case.margin,
        };
        let (out, report) = filter(&ctx, &r, &values, case.l, case.k_c, &options).unwrap();
        let tag = case.tag;
        assert_eq!(out.len(), r.len(), "{tag}");
        assert_eq!(report.masked, case.masked, "{tag}");
        let ours = [
            report.residual_before,
            report.residual_after,
            report.support_radius,
            report.mask_radius,
            report.norm_kept,
            report.max_change,
        ];
        for (i, (a, b)) in ours.iter().zip(case.report).enumerate() {
            // the residuals are 1 - (a number near one): the sums differ in order,
            // so an absolute 1e-13 is their noise floor (measured: 7e-15)
            let floor = if i < 2 { 1e-13 } else { 0. };
            assert!(close(*a, b, 1e-9, floor), "{tag}: report[{i}] {a} vs {b}");
        }
        let sum = out.iter().sum::<f64>();
        let squares = out.iter().map(|x| x * x).sum::<f64>();
        assert!(
            close(sum, case.sum, 1e-9, 0.),
            "{tag}: sum {sum} vs {}",
            case.sum
        );
        assert!(close(squares, case.sum_squares, 1e-9, 0.), "{tag}: squares");
        let peak = out.iter().fold(0., |a: f64, x| a.max(x.abs()));
        for (j, expected) in case.indices.iter().zip(case.samples) {
            let error = (out[*j] - expected).abs() / peak;
            worst = worst.max(error);
            assert!(error < 1e-9, "{tag}: node {j}: {} vs {expected}", out[*j]);
        }
    }
    eprintln!(
        "worst relative sample difference over {} cases: {worst:e}",
        CASES.len()
    );
}

#[test]
fn a_gaussian_obeys_parseval_and_the_round_trip() {
    let _guard = serial();
    let ctx = context();
    let r = linspace(0.005, 30., 1500);
    for l in 0..3usize {
        let values = gaussian(&r, l as i32, 1.);
        // Parseval: all of the norm lives below a generous cutoff.
        let inside = band_weight(&ctx, &r, &values, l, 10., 8).unwrap();
        assert!((inside - 1.).abs() < 1e-4, "l = {l}: {inside}");
        // Round trip: a sharp window far above the content returns the function.
        let sharp = Options {
            method: Method::Plain,
            rolloff: 0.,
            renormalize: false,
            ..Options::default()
        };
        let (back, _) = filter(&ctx, &r, &values, l, 10., &sharp).unwrap();
        let peak = values.iter().fold(0., |a: f64, x| a.max(x.abs()));
        let worst = back
            .iter()
            .zip(&values)
            .fold(0., |a: f64, (b, v)| a.max((b - v).abs()));
        assert!(worst < 2e-4 * peak, "l = {l}: {worst}");
    }
}

#[test]
fn the_filtered_function_keeps_its_power_law_at_the_origin() {
    let _guard = serial();
    let ctx = context();
    let r = linspace(0.005, 30., 1500);
    for l in [1usize, 2] {
        let values = gaussian(&r, l as i32, 1.);
        let (out, _) = filter(&ctx, &r, &values, l, 7.5, &Options::default()).unwrap();
        let head = (0..10)
            .map(|i| out[i] / r[i].powi(l as i32))
            .collect::<Vec<_>>();
        assert!(head.iter().all(|x| x.is_finite()));
        assert!(head
            .iter()
            .all(|x| (x - head[0]).abs() < 0.05 * head[0].abs()));
        let ratio = out[1] / out[0];
        assert!((ratio / (r[1] / r[0]).powi(l as i32) - 1.).abs() < 0.01);
    }
}

#[test]
fn a_hard_cutoff_removes_weight_and_the_norm_is_kept_by_default() {
    let _guard = serial();
    let ctx = context();
    let r = linspace(0.005, 30., 1500);
    let values = gaussian(&r, 0, 6.);
    let before = band_weight(&ctx, &r, &values, 0, 4., 8).unwrap();
    assert!(before < 0.99, "there is something to remove: {before}");
    let (out, report) = filter(&ctx, &r, &values, 0, 4., &Options::default()).unwrap();
    assert!((report.residual_before - (1. - before)).abs() < 1e-14);
    assert!(report.residual_after < 0.5 * report.residual_before);
    let norm = |f: &[f64]| {
        (1..r.len())
            .map(|i| {
                0.5 * (f[i] * f[i] * r[i] * r[i] + f[i - 1] * f[i - 1] * r[i - 1] * r[i - 1])
                    * (r[i] - r[i - 1])
            })
            .sum::<f64>()
    };
    assert!(
        (norm(&out) / norm(&values) - 1.).abs() < 1e-3,
        "renormalized to the original norm"
    );
}

#[test]
fn the_windows_are_smooth_and_compact() {
    let k_c = 8.;
    for i in 0..=500 {
        let k = 10. * i as f64 / 500.;
        let w = window_at(k, k_c, 0.25);
        if k <= 0.75 * k_c {
            assert_eq!(w, 1.);
        }
        if k >= k_c {
            assert_eq!(w, 0.);
        }
        assert!((0. ..=1.).contains(&w));
        if i > 0 {
            assert!(w <= window_at(10. * (i - 1) as f64 / 500., k_c, 0.25) + 1e-12);
        }
    }
    assert_eq!(window_at(1., 8., 0.), 1.);
    assert_eq!(window_at(8., 8., 0.), 0.);
    assert_eq!(mask_at(1., 2., 3.), 1.);
    assert_eq!(mask_at(3., 2., 3.), 0.);
    assert!((mask_at(2.5, 2., 3.) - 0.5).abs() < 1e-15);
    assert_eq!(mask_at(0.5, 1., 1.), 1.);
    assert_eq!(mask_at(1.5, 1., 1.), 0.);
}

fn invalid(result: Result<(Vec<f64>, mandacaru_core::basis::filter::Report), Error>) -> String {
    match result {
        Err(Error::Invalid(message)) => message,
        Err(other) => panic!("expected Invalid, got {other:?}"),
        Ok(_) => panic!("expected an error"),
    }
}

#[test]
fn unsupported_input_raises_and_never_panics() {
    let _guard = serial();
    let ctx = context();
    let r = linspace(0.01, 10., 200);
    let v = gaussian(&r, 0, 1.);
    let go = |r: &[f64], v: &[f64], l: usize, k: f64, o: &Options| filter(&ctx, r, v, l, k, o);
    let o = Options::default();
    assert!(invalid(go(&[], &[], 0, 3., &o)).contains("two nodes"));
    assert!(invalid(go(&[0.5], &[1.], 0, 3., &o)).contains("two nodes"));
    assert!(invalid(go(&r, &v[..10], 0, 3., &o)).contains("length"));
    let mut bad = v.clone();
    bad[3] = f64::NAN;
    assert!(invalid(go(&r, &bad, 0, 3., &o)).contains("finite"));
    let mut decreasing = r.clone();
    decreasing.swap(5, 6);
    assert!(invalid(go(&decreasing, &v, 0, 3., &o)).contains("increasing"));
    let mut negative = r.clone();
    negative[0] = -0.1;
    assert!(invalid(go(&negative, &v, 0, 3., &o)).contains("non-negative"));
    assert!(invalid(go(&r, &v, MAX_L + 1, 3., &o)).contains("l <="));
    for k_c in [0., -1., f64::NAN, f64::INFINITY] {
        assert!(invalid(go(&r, &v, 0, k_c, &o)).contains("cutoff"));
    }
    for rolloff in [-0.1, 1.5, f64::NAN] {
        assert!(invalid(go(&r, &v, 0, 3., &Options { rolloff, ..o })).contains("roll-off"));
    }
    for switch_margin in [0., -1., f64::NAN] {
        assert!(invalid(go(&r, &v, 0, 3., &Options { switch_margin, ..o })).contains("margin"));
    }
    assert!(invalid(go(&r, &v, 0, 3., &Options { oversample: 0, ..o })).contains("oversampling"));
    assert!(matches!(Method::parse("nope"), Err(Error::Invalid(_))));
    assert_eq!(Method::NAMES[0], "switch", "the default method comes first");
    assert!(band_weight(&ctx, &r[..1], &v[..1], 0, 3., 8).is_err());
    assert!(band_weight(&ctx, &r, &v, 0, 0., 8).is_err());
}

#[test]
fn a_cutoff_the_context_cannot_hold_is_refused() {
    let _guard = serial();
    let ctx = Context::new(Some(1), 8 << 20).unwrap();
    let r = linspace(0.01, 30., 2000);
    let v = gaussian(&r, 0, 1.);
    // 1e7 wave-vectors at 8 bytes each (three arrays) exceed 8 MiB.
    let result = filter(&ctx, &r, &v, 0, 1e6, &Options::default());
    assert!(matches!(result, Err(Error::Budget { .. })), "{result:?}");
    assert_eq!(ctx.memory_used(), 0, "every reservation is released");
    let absurd = filter(&ctx, &r, &v, 0, 1e30, &Options::default());
    assert!(matches!(absurd, Err(Error::Invalid(_))), "{absurd:?}");
}

/// The memory of the filter is a fixed number of radial-length arrays plus three of
/// wave-vector length, whatever the cutoff, and there is no `n_k x n_r` kernel matrix.
#[test]
fn the_memory_does_not_grow_with_the_cutoff() {
    let _guard = serial();
    let n = 6000usize;
    let r = (0..n)
        .map(|i| 0.001 + 30. * ((i as f64) / (n - 1) as f64).powi(2))
        .collect::<Vec<_>>();
    assert!(r.windows(2).all(|w| w[1] > w[0]));
    let r_max = r[n - 1];
    let values = gaussian(&r, 1, 0.4);
    let mut previous = None;
    for k_c in [2., 5., 10.47] {
        let ctx = Context::new(Some(2), 256 << 20).unwrap();
        let baseline = LIVE.load(Ordering::Relaxed);
        PEAK.store(baseline, Ordering::Relaxed);
        let (out, _) = filter(&ctx, &r, &values, 1, k_c, &Options::default()).unwrap();
        let heap = PEAK.load(Ordering::Relaxed) - baseline;
        let reserved = ctx.memory_peak();
        let n_k = (k_c * 8. * r_max / PI).ceil() as usize + 1;
        let bound = 8 * (8 * n + 4 * n_k);
        let matrix = 8 * n * n_k;
        eprintln!(
            "k_c {k_c}: n_r {n}, n_k {n_k}, reserved {reserved} B, heap {heap} B, bound {bound} B, matrix {matrix} B"
        );
        assert!(reserved <= bound, "reserved {reserved} > bound {bound}");
        assert!(heap <= bound + (1 << 20), "heap {heap} > bound {bound}");
        assert!(
            heap < matrix / 4 || matrix < (1 << 20),
            "no n_k x n_r kernel"
        );
        assert_eq!(out.len(), n);
        // The memory is flat in the cutoff: the radial arrays are the same, only the
        // three wave-vector arrays grow.
        if let Some((last_k, last_reserved)) = previous {
            let allowed = 8 * 4 * (n_k.saturating_sub(last_k));
            assert!(
                reserved <= last_reserved + allowed,
                "grows faster than the wave-vector arrays"
            );
        }
        previous = Some((n_k, reserved));
        assert_eq!(ctx.memory_used(), 0);
    }
}
