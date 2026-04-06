"""
Persistent mpchecker daemon server.

Keeps the asteroid catalog, KD-tree index, and SPICE kernels loaded in memory,
accepting check_observations requests over a Unix-domain socket.  Eliminates
the ~1-2 s cold-start cost on every mpchecker invocation.

Usage via CLI
-------------
  mpchecker --start-daemon          # fork daemon into background, wait for ready
  mpchecker --stop-daemon           # send SIGTERM to running daemon
  mpchecker --serve                 # foreground server (for debugging)
  mpchecker obs.txt                 # auto-uses daemon if running, else standalone
  mpchecker --no-daemon obs.txt     # skip daemon, always run standalone

Protocol
--------
Length-prefixed binary frames over a Unix socket:
  [4-byte big-endian uint32 = payload length] [payload = pickle bytes]
Request  payload : (observations: list[Observation], params: dict)
Response payload : results: list[CheckResult]
"""

import logging
import os
import pickle
import signal
import socket
import struct
import sys
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths (per-user so multiple users on the same node don't collide)
# ---------------------------------------------------------------------------

def _sock_path() -> Path:
    tmpdir = Path(os.environ.get('TMPDIR', '/tmp'))
    return tmpdir / f'mpchecker-{os.getuid()}.sock'


def _pid_path() -> Path:
    tmpdir = Path(os.environ.get('TMPDIR', '/tmp'))
    return tmpdir / f'mpchecker-{os.getuid()}.pid'


def _log_path() -> Path:
    tmpdir = Path(os.environ.get('TMPDIR', '/tmp'))
    return tmpdir / f'mpchecker-{os.getuid()}.log'


# ---------------------------------------------------------------------------
# Socket message framing
# ---------------------------------------------------------------------------

def _recvn(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError('socket closed prematurely')
        buf.extend(chunk)
    return bytes(buf)


def _recv_msg(sock: socket.socket) -> bytes:
    header = _recvn(sock, 4)
    n = struct.unpack('>I', header)[0]
    return _recvn(sock, n)


def _send_msg(sock: socket.socket, data: bytes) -> None:
    sock.sendall(struct.pack('>I', len(data)) + data)


# ---------------------------------------------------------------------------
# Daemon server
# ---------------------------------------------------------------------------

def serve(
    mag_limit: float = 25.0,
    n_workers: int = 1,
    mpcat_dir: Optional[Path] = None,
) -> None:
    """
    Start the daemon server (blocking).

    Loads the asteroid catalog, SPICE kernels, and KD-tree index once, then
    listens for incoming requests on a Unix socket indefinitely.  Rebuilds
    the index automatically when it goes stale.

    Parameters
    ----------
    mag_limit  : faint limit used when loading the asteroid catalog
    n_workers  : parallel workers forwarded to check_observations
    mpcat_dir  : if provided, load MPCATIndex for fo orbit refitting
    """
    from .mpcorb import load_mpcorb, load_comets, load_obscodes
    from .index import get_or_build_index
    from .config import CACHE_DIR
    from .satellites import _load_base_kernels

    print('[daemon] Loading asteroid catalog …', flush=True)
    asteroids = load_mpcorb()
    comets    = load_comets()
    obscodes  = load_obscodes()
    print(f'[daemon] {len(asteroids)} asteroids loaded', flush=True)

    from .checker import build_asteroid_soa
    print('[daemon] Pre-extracting asteroid SOA …', flush=True)
    asteroid_soa = build_asteroid_soa(asteroids)

    print('[daemon] Loading SPICE kernels …', flush=True)
    _load_base_kernels()

    print('[daemon] Building / loading sky index …', flush=True)
    from astropy.time import Time
    sky_index = get_or_build_index(asteroids, obscodes, CACHE_DIR,
                                   t_now_mjd=Time.now().mjd)

    mpcat_index = None
    if mpcat_dir is not None:
        try:
            from .mpcat import MPCATIndex
            mpcat_index = MPCATIndex(mpcat_dir)
            print(f'[daemon] MPCAT index loaded from {mpcat_dir}', flush=True)
        except Exception as exc:
            print(f'[daemon] WARNING: could not load MPCAT index: {exc}', flush=True)

    print('[daemon] Ready — listening on', _sock_path(), flush=True)

    # Set up listening socket
    sock_path = _sock_path()
    if sock_path.exists():
        sock_path.unlink()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sock_path))
    srv.listen(8)
    sock_path.chmod(0o600)

    # Write PID file
    pid_path = _pid_path()
    pid_path.write_text(str(os.getpid()))

    def _shutdown(signum, frame):
        log.info('Daemon shutting down (signal %d)', signum)
        try:
            srv.close()
            sock_path.unlink(missing_ok=True)
            pid_path.unlink(missing_ok=True)
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    while True:
        try:
            conn, _ = srv.accept()
        except OSError:
            break
        try:
            _handle_client(conn, asteroids, comets, obscodes, sky_index,
                           n_workers, mpcat_index, asteroid_soa=asteroid_soa)
        except Exception as exc:
            log.warning('Error handling client: %s', exc)
        finally:
            conn.close()

        # Rebuild index if stale (happens ~every 7-8 days for a long-running daemon)
        t_now = Time.now().mjd
        if not sky_index.is_fresh(t_now):
            log.info('Index stale, rebuilding …')
            try:
                sky_index = get_or_build_index(asteroids, obscodes, CACHE_DIR,
                                               t_now_mjd=t_now)
            except Exception as exc:
                log.error('Index rebuild failed: %s', exc)


def _handle_client(conn, asteroids, comets, obscodes, sky_index, n_workers,
                   mpcat_index=None, asteroid_soa=None):
    data = _recv_msg(conn)
    observations, params = pickle.loads(data)

    from .checker import check_observations
    results = check_observations(
        observations, asteroids, comets, obscodes,
        sky_index=sky_index,
        n_workers=n_workers,
        mpcat_index=mpcat_index,
        asteroid_soa=asteroid_soa,
        **params,
    )
    _send_msg(conn, pickle.dumps(results))


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def query_daemon(observations, params: dict, timeout: float = 60.0):
    """
    Send a check_observations request to the running daemon.

    Returns list[CheckResult] on success, or None if the daemon is not
    reachable (caller should fall back to standalone mode).

    Parameters
    ----------
    observations : list[Observation] to check
    params       : keyword arguments forwarded to check_observations
                   (search_radius_arcmin, mag_limit, dynmodel, check_sats, …)
    timeout      : socket timeout in seconds
    """
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect(str(_sock_path()))
        _send_msg(sock, pickle.dumps((observations, params)))
        data = _recv_msg(sock)
        sock.close()
        return pickle.loads(data)
    except (FileNotFoundError, ConnectionRefusedError, OSError):
        return None


def is_daemon_running() -> bool:
    """Return True if the daemon socket is accepting connections."""
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        sock.connect(str(_sock_path()))
        sock.close()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

def start_daemon_background(mag_limit: float = 25.0, n_workers: int = 1,
                            mpcat_dir: Optional[Path] = None) -> int:
    """
    Fork and start the daemon in the background using the Unix double-fork
    pattern.  Returns the PID of the grandchild (the actual daemon process).

    Logs are written to _log_path().
    """
    sys.stdout.flush()
    sys.stderr.flush()

    # First fork
    pid = os.fork()
    if pid > 0:
        # Parent: wait for first child to exit, then return grandchild PID.
        # The grandchild PID is written to the PID file by serve().
        os.waitpid(pid, 0)
        # Give the grandchild a moment to write the PID file.
        import time
        for _ in range(20):
            time.sleep(0.1)
            pp = _pid_path()
            if pp.exists():
                try:
                    return int(pp.read_text().strip())
                except ValueError:
                    pass
        return -1   # PID unknown but daemon may still be starting

    # First child: become session leader
    os.setsid()

    # Second fork — grandchild is reparented to init, won't become zombie
    pid2 = os.fork()
    if pid2 > 0:
        # Use os._exit() here to skip Python atexit/finalizers.  The first
        # child has already inherited NumPy/Numba thread pools; calling
        # sys.exit() runs their cleanup code on shared resources and triggers
        # SIGABRT in the parent.  os._exit() exits immediately without any
        # Python-level teardown.
        os._exit(0)

    # Grandchild: this IS the daemon process
    log_path = _log_path()
    devnull  = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)                                           # stdin  → /dev/null
    logfd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(logfd, 1)                                             # stdout → log
    os.dup2(logfd, 2)                                             # stderr → log
    os.close(logfd)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    try:
        serve(mag_limit=mag_limit, n_workers=n_workers, mpcat_dir=mpcat_dir)
    except Exception as exc:
        print(f'FATAL: daemon crashed: {exc}', flush=True)
    finally:
        sys.exit(0)


def stop_daemon() -> bool:
    """
    Stop the running daemon by sending SIGTERM.
    Returns True if a process was found and signalled.
    """
    pid_path = _pid_path()
    if not pid_path.exists():
        # Try unlinking a stale socket
        sp = _sock_path()
        if sp.exists():
            sp.unlink(missing_ok=True)
        return False
    try:
        pid = int(pid_path.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        pid_path.unlink(missing_ok=True)
        return True
    except (ValueError, ProcessLookupError, PermissionError) as exc:
        log.warning('stop_daemon: %s', exc)
        pid_path.unlink(missing_ok=True)
        _sock_path().unlink(missing_ok=True)
        return False
