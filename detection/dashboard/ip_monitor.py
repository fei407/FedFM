from concurrent.futures import ThreadPoolExecutor
import subprocess, os, functools

DEFAULT_TIMEOUT = 1                         #
MAX_THREADS_FACTOR = 4                      #
MAX_GLOBAL_TIMEOUT = 3.0                    #

IS_WIN = os.name == "nt"
BASE_CMD = ["ping", "-n", "1"] if IS_WIN else ["ping", "-c", "1", "-n"]

def _build_cmd(ip: str, timeout: float) -> list[str]:
    return (BASE_CMD
            + (["-w", str(int(timeout*1000))] if IS_WIN else ["-W", str(timeout)])
            + [ip])

def ping_device(ip: str, timeout=DEFAULT_TIMEOUT) -> bool:
    return subprocess.run(
        _build_cmd(ip, timeout),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0

def _emoji(ok: bool) -> str:
    return "🟢" if ok else "🔴"

def check_device_status(device_table: list, timeout=DEFAULT_TIMEOUT) -> list:
    n = len(device_table)
    max_workers = min(os.cpu_count() * MAX_THREADS_FACTOR, n)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        statuses = list(pool.map(
            functools.partial(ping_device, timeout=timeout),
            (d["ip"] for d in device_table),
            timeout=MAX_GLOBAL_TIMEOUT
        ))

    print(f"\n")
    for dev, ok in zip(device_table, statuses):
        print(f"[PING] {dev['ip']:<15} → {_emoji(ok)}")
    print(f"\n")

    return [
        {**dev, "status": _emoji(ok)}
        for dev, ok in zip(device_table, statuses)
    ]
