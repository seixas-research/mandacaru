#!/usr/bin/env python
"""Refuse to start a heavy job while the Mac is short of memory or swapping.

usage: memguard.py [--need-gb 1.5] [--wait 900] && <heavy command>

Exit 0 when the job may start; exit 1 (after waiting up to --wait seconds, polling
every 20 s) when free memory stays under 30 percent, when swap-outs are rising, or
when swap in use exceeds 4.5 GB. It never kills anything.
"""
import argparse, re, subprocess, sys, time

def vm():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    d = {}
    for line in out.splitlines():
        m = re.match(r'(.+?):\s+(\d+)\.?$', line.strip())
        if m:
            d[m.group(1).strip().strip('"')] = int(m.group(2))
    return d

def free_percent():
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    m = re.search(r"free percentage:\s*(\d+)%", out)
    return int(m.group(1)) if m else 0

def swap_used_gb():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.,]+)M", out)
    return float(m.group(1).replace(",", ".")) / 1024 if m else 99.0

def ok(need_gb):
    a = vm(); time.sleep(5); b = vm()
    rising = (b.get("Swapouts", 0) - a.get("Swapouts", 0)) * 16384 > 8 * 2**20   # more than 8 MB in 5 s
    free = free_percent(); swap = swap_used_gb()
    free_gb = free / 100 * 16
    good = (free >= 30) and (not rising) and (swap <= 4.5) and (free_gb >= need_gb + 2)
    return good, f"free={free}% (~{free_gb:.1f} GB) swap_used={swap:.1f} GB swapouts_rising={rising} need={need_gb} GB"

p = argparse.ArgumentParser()
p.add_argument("--need-gb", type=float, default=1.0)
p.add_argument("--wait", type=int, default=900)
a = p.parse_args()
deadline = time.time() + a.wait
while True:
    good, msg = ok(a.need_gb)
    if good:
        print("memguard: ok -", msg); sys.exit(0)
    if time.time() >= deadline:
        print("memguard: REFUSED -", msg, file=sys.stderr); sys.exit(1)
    print("memguard: waiting -", msg, file=sys.stderr, flush=True); time.sleep(20)
