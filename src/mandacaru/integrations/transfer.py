"""Controls and result metadata for native molecular ansatz handover."""
import math


def configure(driver, *, transfer, transfer_threshold, transfer_steps,
              update_between_geometries, resume):
    if type(transfer) is not bool or type(update_between_geometries) is not bool:
        raise TypeError("transfer and update_between_geometries must be booleans")
    if not math.isfinite(transfer_threshold) or not 0 <= transfer_threshold <= 1:
        raise ValueError("transfer_threshold must be in [0,1]")
    if transfer_steps is not None and (type(transfer_steps) is not int or transfer_steps < 0):
        raise ValueError("transfer_steps must be a nonnegative integer or None")
    if transfer and driver.method not in ("adapt-vqe", "mcas-vqe", "valqa"):
        raise ValueError("molecular transfer requires ADAPT or an architecture search")
    if transfer and (resume is not None or driver._execute_circuits):
        raise ValueError("molecular transfer requires local native execution and a fresh geometry")
    if transfer_steps is not None and (not transfer or driver.method not in ("mcas-vqe", "valqa")):
        raise ValueError("transfer_steps requires an architecture search with transfer=True")
    if update_between_geometries and driver.method != "valqa":
        raise ValueError("update_between_geometries requires valqa")
    driver._transfer = transfer
    driver._transfer_threshold = transfer_threshold
    driver._transfer_steps = transfer_steps
    driver._update_between_geometries = update_between_geometries
    driver._transfer_seed = None
    driver._transfer_info = None


def capture(driver):
    driver._transfer_seed = None
    previous = driver.result
    if (driver._transfer and previous is not None and previous.scf is not None
            and previous._model is not None and previous._problem is not None):
        driver._transfer_seed = previous.scf.capture_transfer(
            previous._model, previous._problem, previous._raw["indices"],
            previous.optimal_parameters.tolist())


def prepare(driver, reference):
    previous = driver._transfer_seed
    driver._transfer_seed = None
    driver._transfer_info = None
    if not driver._transfer:
        return None
    if previous is None or reference is None:
        driver._transfer_info = dict(accepted=False, confidence=0.,
                                     reason="no previous molecular ansatz")
        return None
    seed = driver.native_reference.transfer_from(
        previous, reference.hamiltonian, driver.problem,
        threshold=driver._transfer_threshold)
    driver._transfer_info = seed
    return seed if seed["accepted"] else None


def record(driver, raw):
    if driver._transfer_info is not None:
        details = dict(driver._transfer_info)
        if details["accepted"] and "seed_relaxation" in raw:
            details["accepted"] = raw["seed_relaxation"]["kept"]
            if not details["accepted"]:
                details["reason"] = "the carried state did not improve the reference objective after relaxation"
        raw["transfer"] = details
